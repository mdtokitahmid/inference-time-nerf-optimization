"""
Test-time timestamp recovery for D-NeRF scenes.

Given a trained K-Planes model, recovers the timestamp of each test image
by gradient descent on t only (model weights frozen). No GT timestamps
used during optimization — only image pixels + known camera pose.

Produces 3 plots:
  t_vs_step.png       — recovered t per image converging to GT (dotted)
  psnr_vs_step.png    — PSNR per image vs optimization step
  spearman_vs_step.png — global Spearman correlation vs step

Scenes available: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex

Usage:
    cd /scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
    PYTHONPATH='.' /scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python \
        /scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/infer.py \
        --scene mutant --n-steps 50 --lr 0.05 --n-rays 2048
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
# OPTIMIZATION
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


def optimize_t(model, rays_o, rays_d, gt_rgb, bg_color, near_far, device, args):
    """Optimize a single scalar t to minimize pixel MSE. Returns per-step history."""
    t_raw     = torch.nn.Parameter(torch.empty(1, device=device).uniform_(-2.0, 2.0))
    optimizer = torch.optim.Adam([t_raw], lr=args.lr)

    t_history    = []
    psnr_history = []

    model.eval()
    for _ in range(args.n_steps):
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
# PLOTS
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
    gt   = np.array(gt_times)
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
    p.add_argument('--n-steps', type=int,   default=50,
                   help='Gradient steps per test image')
    p.add_argument('--lr',      type=float, default=0.05)
    p.add_argument('--n-rays',  type=int,   default=2048,
                   help='Random rays per image for optimization')
    p.add_argument('--ckpt',    default=None,
                   help='Path to model.pth  (default: logs/<scene>/<scene>/model.pth)')
    args = p.parse_args()

    device   = torch.device('cuda:0')
    log_dir  = os.path.join(LOGS_ROOT, args.scene)
    ckpt_path = args.ckpt or os.path.join(log_dir, args.scene, 'model.pth')
    out_dir  = os.path.join(log_dir, 'infer')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene     : {args.scene}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Steps/img : {args.n_steps}   lr={args.lr}   rays={args.n_rays}")
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
    dset     = kp_trainer.test_dataset
    N        = len(dset.poses)
    H, W     = dset.img_h, dset.img_w
    near_far = dset.per_cam_near_fars[0].to(device)
    nf       = near_far.unsqueeze(0) if near_far.dim() == 1 else near_far

    print(f"Optimizing t for {N} test images...\n")

    all_t    = []
    all_psnr = []
    gt_times = []

    for i in range(N):
        sample   = dset[i]
        rays_o   = sample['rays_o'].to(device)
        rays_d   = sample['rays_d'].to(device)
        gt_rgb   = sample['imgs'].to(device)
        bg_color = sample['bg_color'].to(device)
        gt_t     = sample['timestamps'].squeeze().item()

        perm   = torch.randperm(H * W, device=device)[:args.n_rays]
        t_hist, psnr_hist = optimize_t(
            model, rays_o[perm], rays_d[perm], gt_rgb[perm],
            bg_color, nf, device, args
        )
        all_t.append(t_hist)
        all_psnr.append(psnr_hist)
        gt_times.append(gt_t)

        err = abs(gt_t - t_hist[-1])
        print(f"  img {i:2d} | gt={gt_t:.4f} | recovered={t_hist[-1]:.4f} "
              f"| err={err:.4f} | psnr={psnr_hist[-1]:.2f} dB")

    # ── Summary ───────────────────────────────────────────────────────────
    opt_times = np.array([h[-1] for h in all_t])
    gt_arr    = np.array(gt_times)
    errs      = np.abs(gt_arr - opt_times)
    corr, _   = spearmanr(gt_arr, opt_times)

    print(f"\n{'='*60}")
    print(f"Spearman correlation : {corr:.4f}")
    print(f"Mean  |t error|      : {errs.mean():.4f}")
    print(f"Median |t error|     : {np.median(errs):.4f}")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({
            'spearman':     float(corr),
            'mean_t_err':   float(errs.mean()),
            'median_t_err': float(np.median(errs)),
            'gt_times':     gt_times,
            'opt_times':    opt_times.tolist(),
        }, f, indent=2)

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
