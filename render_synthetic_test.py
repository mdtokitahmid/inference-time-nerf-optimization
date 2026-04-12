"""
Render a synthetic test set at Beta-distributed random timestamps.

Timestamps are sampled from Beta(alpha, beta) in [0,1] then mapped to [-1,1].
This produces a non-uniform distribution where some timestamps are very close
together and others are farther apart — a harder test than the equally-spaced
D-NeRF test set.

Beta shape guide:
  alpha=beta=0.5  →  U-shaped (clusters near both ends, sparse in middle)
  alpha=beta=2    →  bell-shaped (concentrates near t=0)
  alpha=0.5, beta=2 →  skewed left (clusters near t=-1)
  alpha=2, beta=0.5 →  skewed right (clusters near t=+1)

Camera poses are cycled from the D-NeRF test set (20 poses).
Outputs: logs/<scene>/synthetic_test_a<alpha>_b<beta>/test_set.npz
         logs/<scene>/synthetic_test_a<alpha>_b<beta>/timestamp_dist.png

Usage:
    cd /scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
    export PYTHONPATH='.'
    python ../final_experiments/render_synthetic_test.py --scene mutant --n-images 20 --beta-alpha 0.5 --beta-beta 0.5 --seed 42
"""

import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# RENDER
# ─────────────────────────────────────────────

def render_full_image(model, rays_o, rays_d, bg_color, near_far, t_val, chunk=2048):
    """Render all rays for one image. Returns (H*W, 3) cpu tensor."""
    rgbs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            ro = rays_o[i:i+chunk]
            rd = rays_d[i:i+chunk]
            ts = t_val.expand(ro.shape[0])
            nf = near_far.expand(ro.shape[0], -1)
            out = model(ro, rd, bg_color, nf, timestamps=ts)
            rgbs.append(out['rgb'].cpu())
    return torch.cat(rgbs, dim=0)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene',      required=True,
                   choices=['bouncingballs','hellwarrior','hook','jumpingjacks',
                            'lego','mutant','standup','trex'])
    p.add_argument('--n-images',   type=int,   default=20,
                   help='Number of synthetic test images to render')
    p.add_argument('--beta-alpha', type=float, default=0.5,
                   help='Beta distribution alpha parameter')
    p.add_argument('--beta-beta',  type=float, default=0.5,
                   help='Beta distribution beta parameter')
    p.add_argument('--t-min',      type=float, default=None,
                   help='If set, sample t uniformly in [t-min, t-max] instead of Beta. '
                        'Both --t-min and --t-max must be provided together.')
    p.add_argument('--t-max',      type=float, default=None,
                   help='Upper bound for uniform narrow-window sampling.')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--ckpt',       default=None)
    args = p.parse_args()

    if (args.t_min is None) != (args.t_max is None):
        raise ValueError('--t-min and --t-max must be provided together')
    narrow_window = args.t_min is not None

    device    = torch.device('cuda:0')
    log_dir   = os.path.join(LOGS_ROOT, args.scene)
    ckpt_path = args.ckpt or os.path.join(log_dir, args.scene, 'model.pth')

    if narrow_window:
        tag = f"window_{args.t_min}_{args.t_max}"
    else:
        tag = f"a{args.beta_alpha}_b{args.beta_beta}"
    if args.n_images != 20:
        tag += f"_n{args.n_images}"
    out_dir   = os.path.join(log_dir, f'synthetic_test_{tag}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene      : {args.scene}")
    print(f"N images   : {args.n_images}")
    if narrow_window:
        print(f"Sampling   : Uniform[{args.t_min}, {args.t_max}]  (narrow window)")
    else:
        print(f"Beta(α,β)  : ({args.beta_alpha}, {args.beta_beta})")
    print(f"Seed       : {args.seed}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {out_dir}")
    print(f"{'='*60}\n")

    # ── Sample timestamps ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    if narrow_window:
        timestamps = rng.uniform(args.t_min, args.t_max, size=args.n_images).astype(np.float32)
    else:
        t_beta     = rng.beta(args.beta_alpha, args.beta_beta, size=args.n_images)
        timestamps = (2.0 * t_beta - 1.0).astype(np.float32)   # [0,1] → [-1,1]

    gaps = np.diff(np.sort(timestamps))
    print(f"Sampled timestamps (unsorted): {np.round(timestamps, 4).tolist()}")
    print(f"  min={timestamps.min():.4f}  max={timestamps.max():.4f}"
          f"  mean={timestamps.mean():.4f}  std={timestamps.std():.4f}")
    print(f"  min_gap={gaps.min():.4f}  mean_gap={gaps.mean():.4f}\n")

    # ── Plot timestamp distribution ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    title_str = (f'Uniform[{args.t_min},{args.t_max}]' if narrow_window
                 else f'Beta({args.beta_alpha},{args.beta_beta})')
    axes[0].hist(timestamps, bins=20, range=(-1, 1), color='steelblue', edgecolor='white')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('count')
    axes[0].set_title(f'Histogram  {title_str}')
    axes[1].scatter(range(args.n_images), np.sort(timestamps), s=30, color='crimson')
    axes[1].set_xlabel('rank'); axes[1].set_ylabel('t (sorted)')
    axes[1].set_title('Sorted timestamps')
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[2].hist(gaps, bins=30, color='darkorange', edgecolor='white')
    axes[2].set_xlabel('gap between consecutive t'); axes[2].set_ylabel('count')
    axes[2].set_title(f'Gap distribution  (min={gaps.min():.4f})')
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    dist_path = os.path.join(out_dir, 'timestamp_dist.png')
    fig.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved distribution plot: {dist_path}")

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

    # ── Dataset (for camera poses / rays) ─────────────────────────────────
    dset     = kp_trainer.test_dataset
    n_poses  = len(dset.poses)
    near_far = dset.per_cam_near_fars[0].to(device)
    nf       = near_far.unsqueeze(0) if near_far.dim() == 1 else near_far

    print(f"\nRendering {args.n_images} images (cycling over {n_poses} test poses)...\n")

    all_images  = []
    all_rays_o  = []
    all_rays_d  = []
    all_bg      = []

    for i in range(args.n_images):
        pose_idx = i % n_poses
        sample   = dset[pose_idx]
        rays_o   = sample['rays_o'].to(device)   # (H*W, 3)
        rays_d   = sample['rays_d'].to(device)
        bg_color = sample['bg_color'].to(device)
        t_val    = torch.tensor(float(timestamps[i]), device=device, dtype=torch.float32)

        rgb = render_full_image(model, rays_o, rays_d, bg_color, nf, t_val)

        all_images.append(rgb.numpy())
        all_rays_o.append(rays_o.cpu().numpy())
        all_rays_d.append(rays_d.cpu().numpy())
        all_bg.append(bg_color.cpu().numpy())

        print(f"  img {i:2d} | pose={pose_idx} | t={timestamps[i]:+.4f} | "
              f"pixels={rgb.shape[0]}")

    # ── Save .npz ─────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, 'test_set.npz')
    np.savez_compressed(
        npz_path,
        images     = np.stack(all_images),    # (N, H*W, 3)  float32
        timestamps = timestamps,              # (N,)         float32
        rays_o     = np.stack(all_rays_o),    # (N, H*W, 3)  float32
        rays_d     = np.stack(all_rays_d),    # (N, H*W, 3)  float32
        bg_color   = np.stack(all_bg),        # (N, 3)       float32
        near_far   = near_far.cpu().numpy(),  # (2,) or (1,2)
    )
    print(f"\nSaved: {npz_path}")
    print(f"  images shape : {np.stack(all_images).shape}")
    print(f"  timestamps   : {np.round(timestamps, 4).tolist()}")
    print(f"\nDone. Pass to infer_ablation.py with:")
    print(f"  --synthetic-test {npz_path}")


if __name__ == '__main__':
    main()
