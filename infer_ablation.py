"""
Ablation: test-time timestamp recovery with improved initialization strategies.

Four modes:

  multistart  — N random starts × warmup_steps each → pick lowest-loss init
                → continue full gradient descent from best start

  grid        — Evaluate MSE at n_grid equally-spaced t values (no grad)
                → pick top-K → run full gradient descent from each
                → keep the run with lowest final MSE

  grid-only   — Evaluate MSE at n_grid equally-spaced t values (no grad)
                → return argmin directly, NO gradient descent

  random-only — Evaluate MSE at n_random uniformly-sampled t values (no grad)
                → return argmin directly, NO gradient descent

Results saved to logs/<scene>/infer_<mode>/
(existing logs/<scene>/infer/ from infer.py is untouched)

Usage:
    cd /scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
    export PYTHONPATH='.'

    # Multi-start (5 starts × 5 warmup steps)
    python ../final_experiments/infer_ablation.py --scene mutant --mode multistart --n-starts 5 --warmup-steps 5 --n-steps 50 --lr 0.05

    # Coarse grid + gradient (10 grid points, top-3 → full optimize)
    python ../final_experiments/infer_ablation.py --scene mutant --mode grid --n-grid 10 --top-k 3 --n-steps 50 --lr 0.05

    # Pure grid search (no gradient)
    python ../final_experiments/infer_ablation.py --scene mutant --mode grid-only --n-grid 10

    # Pure random search (no gradient, 75 samples to match multistart budget)
    python ../final_experiments/infer_ablation.py --scene mutant --mode random-only --n-random 75
"""

import argparse, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

KPLANES_DIR = "/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes"
DATA_ROOT   = "/scratch/gpfs/MONA/Toki/Academic/COS526/dnerf/data"
LOGS_ROOT   = "/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/logs"

sys.path.insert(0, KPLANES_DIR)
from plenoxels.runners import video_trainer
from plenoxels.utils.parse_args import parse_optfloat


# ─────────────────────────────────────────────
# CONFIG  (must match train.py)
# ─────────────────────────────────────────────

def make_config(scene, out_dir):
    return {
        'expname':        scene,
        'logdir':         out_dir,
        'device':         'cuda:0',
        'data_downsample': 2.0,
        'data_dirs':      [os.path.join(DATA_ROOT, scene)],
        'dataset_type':   'synthetic_dnerf',
        'contract':       False,
        'ndc':            False,
        'isg':            False,
        'isg_step':       -1,
        'ist_step':       -1,
        'keyframes':      False,
        'scene_bbox':     [[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]],
        'num_steps':      1,
        'batch_size':     1024,
        'scheduler_type': 'warmup_cosine',
        'optim_type':     'adam',
        'lr':             0.01,
        'distortion_loss_weight':              0.00,
        'histogram_loss_weight':               1.0,
        'l1_time_planes':                      0.0001,
        'l1_time_planes_proposal_net':         0.0001,
        'plane_tv_weight':                     0.0001,
        'plane_tv_weight_proposal_net':        0.0001,
        'time_smoothness_weight':              0.01,
        'time_smoothness_weight_proposal_net': 0.001,
        'save_every':    1,
        'valid_every':   1,
        'save_outputs':  False,
        'train_fp16':    True,
        'single_jitter':              False,
        'num_samples':                48,
        'num_proposal_iterations':    2,
        'num_proposal_samples':       [256, 128],
        'use_same_proposal_network':  False,
        'use_proposal_weight_anneal': True,
        'proposal_net_args_list': [
            {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [64, 64, 64, 50]},
            {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 50]},
        ],
        'concat_features_across_scales': True,
        'density_activation':            'trunc_exp',
        'linear_decoder':                False,
        'multiscale_res':                [1, 2, 4, 8],
        'grid_config': [{
            'grid_dimensions':       2,
            'input_coordinate_dim':  4,
            'output_coordinate_dim': 32,
            'resolution':            [64, 64, 64, 75],
        }],
    }


# ─────────────────────────────────────────────
# SHARED RENDER UTIL
# ─────────────────────────────────────────────

def render_batch(model, rays_o, rays_d, bg_color, near_far, t_val, chunk=2048):
    rgbs = []
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        ts = t_val.expand(ro.shape[0])
        nf = near_far.expand(ro.shape[0], -1)
        out = model(ro, rd, bg_color, nf, timestamps=ts)
        rgbs.append(out['rgb'])
    return torch.cat(rgbs, dim=0)


def run_gradient_descent(model, rays_o, rays_d, gt_rgb, bg_color, near_far,
                          device, lr, n_steps, t_raw_init):
    """Run n_steps of Adam on t_raw starting from t_raw_init. Returns (t_history, psnr_history)."""
    t_raw     = torch.nn.Parameter(torch.tensor([t_raw_init], device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([t_raw], lr=lr)

    t_history    = []
    psnr_history = []

    model.eval()
    for _ in range(n_steps):
        optimizer.zero_grad()
        t_val    = torch.tanh(t_raw).squeeze()
        rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
        loss     = F.mse_loss(rgb_pred, gt_rgb)
        loss.backward()
        optimizer.step()
        t_history.append(torch.tanh(t_raw).item())
        psnr_history.append(-10.0 * np.log10(loss.item() + 1e-8))

    return t_history, psnr_history


# ─────────────────────────────────────────────
# MODE 1: MULTI-START
# ─────────────────────────────────────────────

def optimize_t_multistart(model, rays_o, rays_d, gt_rgb, bg_color, near_far,
                           device, lr, n_steps, n_starts, warmup_steps):
    """
    Run n_starts random initializations each for warmup_steps.
    Pick the one with the lowest loss after warmup, then continue for n_steps.
    """
    best_loss   = float('inf')
    best_t_raw  = None

    model.eval()
    for s in range(n_starts):
        t_raw = torch.nn.Parameter(torch.empty(1, device=device).uniform_(-2.0, 2.0))
        opt   = torch.optim.Adam([t_raw], lr=lr)
        loss_val = None
        for _ in range(warmup_steps):
            opt.zero_grad()
            t_val    = torch.tanh(t_raw).squeeze()
            rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
            loss     = F.mse_loss(rgb_pred, gt_rgb)
            loss.backward()
            opt.step()
            loss_val = loss.item()

        if loss_val < best_loss:
            best_loss  = loss_val
            best_t_raw = t_raw.detach().item()

    # Full optimization from best warmup result
    return run_gradient_descent(
        model, rays_o, rays_d, gt_rgb, bg_color, near_far,
        device, lr, n_steps, best_t_raw
    )


# ─────────────────────────────────────────────
# MODE 2: COARSE GRID SEARCH
# ─────────────────────────────────────────────

def optimize_t_grid(model, rays_o, rays_d, gt_rgb, bg_color, near_far,
                    device, lr, n_steps, n_grid, top_k):
    """
    1. Evaluate MSE (no grad) at n_grid equally-spaced t values in [-1, 1].
    2. Pick top_k lowest-loss values.
    3. Run full gradient descent from each; keep the best final result.
    """
    # Step 1: coarse evaluation
    grid_t      = torch.linspace(-1.0, 1.0, n_grid)
    grid_losses = []

    model.eval()
    with torch.no_grad():
        for t_val_np in grid_t.tolist():
            t_val    = torch.tensor(t_val_np, device=device, dtype=torch.float32)
            rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
            loss     = F.mse_loss(rgb_pred, gt_rgb)
            grid_losses.append(loss.item())

    # Step 2: pick top_k
    top_indices = np.argsort(grid_losses)[:top_k]
    print(f"      grid losses: {[f'{v:.4f}' for v in grid_losses]}")
    print(f"      top-{top_k} init t values: {[f'{grid_t[i].item():.3f}' for i in top_indices]}")

    # Step 3: full gradient descent from each top-K init, keep best
    best_t_history    = None
    best_psnr_history = None
    best_final_loss   = float('inf')

    for idx in top_indices:
        t_init    = grid_t[idx].item()
        # Convert t → t_raw via atanh (clip to avoid ±inf at boundary)
        t_raw_init = float(np.arctanh(np.clip(t_init, -0.9999, 0.9999)))

        t_hist, psnr_hist = run_gradient_descent(
            model, rays_o, rays_d, gt_rgb, bg_color, near_far,
            device, lr, n_steps, t_raw_init
        )

        # Evaluate final loss to compare runs
        with torch.no_grad():
            t_val    = torch.tensor(t_hist[-1], device=device, dtype=torch.float32)
            rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
            final_loss = F.mse_loss(rgb_pred, gt_rgb).item()

        if final_loss < best_final_loss:
            best_final_loss   = final_loss
            best_t_history    = t_hist
            best_psnr_history = psnr_hist

    return best_t_history, best_psnr_history


# ─────────────────────────────────────────────
# MODE 3: PURE GRID SEARCH (no gradient)
# ─────────────────────────────────────────────

def optimize_t_grid_only(model, rays_o, rays_d, gt_rgb, bg_color, near_far,
                          device, n_grid, n_steps_for_plot):
    """
    Evaluate MSE (no grad) at n_grid equally-spaced t values in [-1, 1].
    Return the argmin directly — no gradient descent at all.
    History is a flat line (single value repeated) for plot compatibility.
    """
    grid_t      = torch.linspace(-1.0, 1.0, n_grid)
    grid_losses = []

    model.eval()
    with torch.no_grad():
        for t_val_np in grid_t.tolist():
            t_val    = torch.tensor(t_val_np, device=device, dtype=torch.float32)
            rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
            loss     = F.mse_loss(rgb_pred, gt_rgb)
            grid_losses.append(loss.item())

    best_idx  = int(np.argmin(grid_losses))
    best_t    = grid_t[best_idx].item()
    best_psnr = -10.0 * np.log10(grid_losses[best_idx] + 1e-8)

    print(f"      grid losses: {[f'{v:.4f}' for v in grid_losses]}")
    print(f"      best t={best_t:.3f}  (idx={best_idx})")

    return [best_t] * n_steps_for_plot, [best_psnr] * n_steps_for_plot


# ─────────────────────────────────────────────
# MODE 4: PURE RANDOM SEARCH (no gradient)
# ─────────────────────────────────────────────

def optimize_t_random_only(model, rays_o, rays_d, gt_rgb, bg_color, near_far,
                            device, n_random, n_steps_for_plot):
    """
    Evaluate MSE (no grad) at n_random uniformly-sampled t values in [-1, 1].
    Return the argmin directly — no gradient descent at all.
    History is a flat line (single value repeated) for plot compatibility.
    """
    random_t    = torch.empty(n_random).uniform_(-1.0, 1.0)
    random_losses = []

    model.eval()
    with torch.no_grad():
        for t_val_np in random_t.tolist():
            t_val    = torch.tensor(t_val_np, device=device, dtype=torch.float32)
            rgb_pred = render_batch(model, rays_o, rays_d, bg_color, near_far, t_val)
            loss     = F.mse_loss(rgb_pred, gt_rgb)
            random_losses.append(loss.item())

    best_idx  = int(np.argmin(random_losses))
    best_t    = random_t[best_idx].item()
    best_psnr = -10.0 * np.log10(random_losses[best_idx] + 1e-8)

    return [best_t] * n_steps_for_plot, [best_psnr] * n_steps_for_plot


# ─────────────────────────────────────────────
# PLOTS  (same format as infer.py)
# ─────────────────────────────────────────────

def plot_t_vs_step(all_t, gt_times, n_steps, path):
    N    = len(all_t)
    cols = 5
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = axes.flatten()
    steps = np.arange(n_steps)
    for i in range(N):
        ax = axes[i]
        ax.plot(steps, all_t[i], color='steelblue', lw=1.5, label='recovered')
        ax.axhline(gt_times[i], color='crimson', lw=1.5, linestyle='--', label='GT')
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f'img {i}  GT={gt_times[i]:.3f}', fontsize=8)
        ax.set_xlabel('step', fontsize=7)
        ax.set_ylabel('t', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(N, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Recovered t vs Step  (dashed = GT)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_psnr_vs_step(all_psnr, n_steps, path):
    N    = len(all_psnr)
    cols = 5
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = axes.flatten()
    steps = np.arange(n_steps)
    for i in range(N):
        ax = axes[i]
        ax.plot(steps, all_psnr[i], color='darkorange', lw=1.5)
        ax.set_title(f'img {i}', fontsize=8)
        ax.set_xlabel('step', fontsize=7)
        ax.set_ylabel('PSNR (dB)', fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(N, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('PSNR vs Optimization Step', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_spearman_vs_step(all_t, gt_times, n_steps, path):
    gt    = np.array(gt_times)
    corrs = []
    for step in range(n_steps):
        recovered = np.array([all_t[i][step] for i in range(len(all_t))])
        corr, _   = spearmanr(gt, recovered)
        corrs.append(corr)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(n_steps), corrs, color='seagreen', lw=2)
    ax.axhline(1.0, color='gray', lw=1, linestyle='--', alpha=0.5)
    ax.axhline(0.0, color='gray', lw=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Optimization Step', fontsize=11)
    ax.set_ylabel('Spearman Correlation', fontsize=11)
    ax.set_title('Spearman Correlation (all test images) vs Step', fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    final = corrs[-1]
    ax.annotate(f'final: {final:.4f}',
                xy=(n_steps - 1, final),
                xytext=(n_steps * 0.6, final - 0.2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene',   required=True,
                   choices=['bouncingballs','hellwarrior','hook','jumpingjacks',
                            'lego','mutant','standup','trex'])
    p.add_argument('--mode',    required=True, choices=['multistart', 'grid', 'grid-only', 'random-only'],
                   help='Optimization strategy')
    p.add_argument('--n-steps', type=int,   default=50,
                   help='Gradient steps per image (after init phase)')
    p.add_argument('--lr',      type=float, default=0.05)
    p.add_argument('--n-rays',  type=int,   default=2048,
                   help='Random rays per image for optimization')
    p.add_argument('--ckpt',    default=None,
                   help='Path to model.pth  (default: logs/<scene>/<scene>/model.pth)')

    # Multi-start args
    p.add_argument('--n-starts',     type=int, default=5,
                   help='[multistart] Number of random initializations')
    p.add_argument('--warmup-steps', type=int, default=5,
                   help='[multistart] Gradient steps per start during warmup')

    # Grid search args
    p.add_argument('--n-grid', type=int, default=10,
                   help='[grid/grid-only] Number of equally-spaced t values to evaluate')
    p.add_argument('--top-k',  type=int, default=3,
                   help='[grid] Number of top candidates to run full optimization from')

    # Random-only args
    p.add_argument('--n-random', type=int, default=75,
                   help='[random-only] Number of random t samples to evaluate (default=75 matches multistart budget)')

    # Synthetic test set
    p.add_argument('--synthetic-test', default=None,
                   help='Path to .npz from render_synthetic_test.py. '
                        'If set, uses rendered images instead of the D-NeRF test set.')
    p.add_argument('--tag', default='',
                   help='Suffix appended to output dir name (e.g. "_beta0.5" for synthetic runs)')

    args = p.parse_args()

    device    = torch.device('cuda:0')
    log_dir   = os.path.join(LOGS_ROOT, args.scene)
    ckpt_path = args.ckpt or os.path.join(log_dir, args.scene, 'model.pth')
    out_dir   = os.path.join(log_dir, f'infer_{args.mode}{args.tag}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene     : {args.scene}")
    print(f"Mode      : {args.mode}")
    print(f"Checkpoint: {ckpt_path}")
    if args.mode == 'multistart':
        print(f"Starts    : {args.n_starts}  warmup={args.warmup_steps} steps each")
        print(f"Opt steps : {args.n_steps}   lr={args.lr}   rays={args.n_rays}")
    elif args.mode == 'grid':
        print(f"Grid pts  : {args.n_grid}  top-K={args.top_k}")
        print(f"Opt steps : {args.n_steps}   lr={args.lr}   rays={args.n_rays}")
    elif args.mode == 'grid-only':
        print(f"Grid pts  : {args.n_grid}  (no gradient descent)")
        print(f"Rays      : {args.n_rays}")
    elif args.mode == 'random-only':
        print(f"Random samples: {args.n_random}  (no gradient descent)")
        print(f"Rays      : {args.n_rays}")
    print(f"Output    : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    config = make_config(args.scene, log_dir)
    data   = video_trainer.load_data(
        parse_optfloat(config['data_downsample'], 1.0),
        config['data_dirs'],
        validate_only=False,
        render_only=False,
        **{k: v for k, v in config.items()
           if k not in ['data_downsample', 'data_dirs']}
    )
    config.update(data)
    kp_trainer = video_trainer.VideoTrainer(**config)

    ckpt = torch.load(ckpt_path, map_location=device)
    kp_trainer.model.load_state_dict(ckpt['model'])
    model = kp_trainer.model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    # ── Dataset ───────────────────────────────────────────────────────────
    if args.synthetic_test:
        synth      = np.load(args.synthetic_test)
        s_images   = synth['images']      # (N, H*W, 3)
        s_times    = synth['timestamps']  # (N,)
        s_rays_o   = synth['rays_o']      # (N, H*W, 3)
        s_rays_d   = synth['rays_d']      # (N, H*W, 3)
        s_bg       = synth['bg_color']    # (N, 3)
        s_near_far = synth['near_far']
        N        = len(s_times)
        n_pixels = s_images.shape[1]
        nf_base  = torch.tensor(s_near_far, device=device, dtype=torch.float32)
        nf       = nf_base.unsqueeze(0) if nf_base.dim() == 1 else nf_base
        print(f"  [synthetic] loaded {N} images from {args.synthetic_test}")
    else:
        dset     = kp_trainer.test_dataset
        N        = len(dset.poses)
        H, W     = dset.img_h, dset.img_w
        near_far = dset.per_cam_near_fars[0].to(device)
        nf       = near_far.unsqueeze(0) if near_far.dim() == 1 else near_far
        n_pixels = H * W

    print(f"Optimizing t for {N} test images  [mode={args.mode}]...\n")

    all_t    = []
    all_psnr = []
    gt_times = []

    for i in range(N):
        if args.synthetic_test:
            rays_o   = torch.tensor(s_rays_o[i],  device=device, dtype=torch.float32)
            rays_d   = torch.tensor(s_rays_d[i],  device=device, dtype=torch.float32)
            gt_rgb   = torch.tensor(s_images[i],  device=device, dtype=torch.float32)
            bg_color = torch.tensor(s_bg[i],       device=device, dtype=torch.float32)
            gt_t     = float(s_times[i])
        else:
            sample   = dset[i]
            rays_o   = sample['rays_o'].to(device)
            rays_d   = sample['rays_d'].to(device)
            gt_rgb   = sample['imgs'].to(device)
            bg_color = sample['bg_color'].to(device)
            gt_t     = sample['timestamps'].squeeze().item()

        perm   = torch.randperm(n_pixels, device=device)[:args.n_rays]
        ro, rd, gt = rays_o[perm], rays_d[perm], gt_rgb[perm]

        print(f"  img {i:2d} | gt={gt_t:.4f}")

        if args.mode == 'multistart':
            t_hist, psnr_hist = optimize_t_multistart(
                model, ro, rd, gt, bg_color, nf, device,
                args.lr, args.n_steps, args.n_starts, args.warmup_steps
            )
        elif args.mode == 'grid':
            t_hist, psnr_hist = optimize_t_grid(
                model, ro, rd, gt, bg_color, nf, device,
                args.lr, args.n_steps, args.n_grid, args.top_k
            )
        elif args.mode == 'grid-only':
            t_hist, psnr_hist = optimize_t_grid_only(
                model, ro, rd, gt, bg_color, nf, device,
                args.n_grid, args.n_steps
            )
        else:  # random-only
            t_hist, psnr_hist = optimize_t_random_only(
                model, ro, rd, gt, bg_color, nf, device,
                args.n_random, args.n_steps
            )

        all_t.append(t_hist)
        all_psnr.append(psnr_hist)
        gt_times.append(gt_t)

        err = abs(gt_t - t_hist[-1])
        print(f"         recovered={t_hist[-1]:.4f} | err={err:.4f} | psnr={psnr_hist[-1]:.2f} dB")

    # ── Summary ───────────────────────────────────────────────────────────
    opt_times = np.array([h[-1] for h in all_t])
    gt_arr    = np.array(gt_times)
    errs      = np.abs(gt_arr - opt_times)
    corr, _   = spearmanr(gt_arr, opt_times)

    print(f"\n{'='*60}")
    print(f"Mode                 : {args.mode}")
    print(f"Spearman correlation : {corr:.4f}")
    print(f"Mean  |t error|      : {errs.mean():.4f}")
    print(f"Median |t error|     : {np.median(errs):.4f}")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────
    result = {
        'mode':         args.mode,
        'spearman':     float(corr),
        'mean_t_err':   float(errs.mean()),
        'median_t_err': float(np.median(errs)),
        'gt_times':     gt_times,
        'opt_times':    opt_times.tolist(),
        'config': {
            'n_rays':  args.n_rays,
            **({'n_steps': args.n_steps, 'lr': args.lr,
                'n_starts': args.n_starts, 'warmup_steps': args.warmup_steps}
               if args.mode == 'multistart' else
               {'n_steps': args.n_steps, 'lr': args.lr,
                'n_grid': args.n_grid, 'top_k': args.top_k}
               if args.mode == 'grid' else
               {'n_grid': args.n_grid}
               if args.mode == 'grid-only' else
               {'n_random': args.n_random}),
        }
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {os.path.join(out_dir, 'results.json')}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_t_vs_step(all_t, gt_times, args.n_steps,
                   os.path.join(out_dir, 't_vs_step.png'))
    plot_psnr_vs_step(all_psnr, args.n_steps,
                      os.path.join(out_dir, 'psnr_vs_step.png'))
    plot_spearman_vs_step(all_t, gt_times, args.n_steps,
                          os.path.join(out_dir, 'spearman_vs_step.png'))

    print(f"\nAll outputs in: {out_dir}")


if __name__ == '__main__':
    main()
