"""
Train K-Planes on a D-NeRF scene with GT poses and GT timestamps.
After training, evaluates PSNR on the full test set.

Scenes available: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex

Usage:
    cd /scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
    PYTHONPATH='.' /scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python \
        /scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/train.py \
        --scene mutant --n-steps 30001
"""

import argparse, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

KPLANES_DIR = "/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes"
DATA_ROOT   = "/scratch/gpfs/MONA/Toki/Academic/COS526/dnerf/data"
LOGS_ROOT   = "/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments/logs"

sys.path.insert(0, KPLANES_DIR)
from plenoxels.runners import video_trainer
from plenoxels.utils.parse_args import parse_optfloat


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def make_config(scene, n_steps, out_dir):
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
        # Optimization
        'num_steps':      n_steps,
        'batch_size':     1024,
        'scheduler_type': 'warmup_cosine',
        'optim_type':     'adam',
        'lr':             0.01,
        # Regularization
        'distortion_loss_weight':              0.00,
        'histogram_loss_weight':               1.0,
        'l1_time_planes':                      0.0001,
        'l1_time_planes_proposal_net':         0.0001,
        'plane_tv_weight':                     0.0001,
        'plane_tv_weight_proposal_net':        0.0001,
        'time_smoothness_weight':              0.01,
        'time_smoothness_weight_proposal_net': 0.001,
        # Training
        'save_every':    n_steps - 1,
        'valid_every':   n_steps - 1,
        'save_outputs':  True,
        'train_fp16':    True,
        # Raymarching
        'single_jitter':             False,
        'num_samples':               48,
        'num_proposal_iterations':   2,
        'num_proposal_samples':      [256, 128],
        'use_same_proposal_network': False,
        'use_proposal_weight_anneal': True,
        'proposal_net_args_list': [
            {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [64, 64, 64, 50]},
            {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 50]},
        ],
        # Model
        'concat_features_across_scales': True,
        'density_activation':            'trunc_exp',
        'linear_decoder':                False,
        'multiscale_res':                [1, 2, 4, 8],
        'grid_config': [{
            'grid_dimensions':      2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution':           [64, 64, 64, 75],
        }],
    }


# ─────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────

def render_batch(model, rays_o, rays_d, bg_color, near_far, t_val, device, chunk=4096):
    rgbs = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            ro = rays_o[i:i+chunk].to(device)
            rd = rays_d[i:i+chunk].to(device)
            ts = t_val.expand(ro.shape[0])
            nf = near_far.expand(ro.shape[0], -1)
            out = model(ro, rd, bg_color, nf, timestamps=ts)
            rgbs.append(out['rgb'])
    return torch.cat(rgbs, dim=0)


def evaluate_test_psnr(kp_trainer, device):
    model = kp_trainer.model.to(device)
    dset  = kp_trainer.test_dataset
    N     = len(dset.poses)
    near_far = dset.per_cam_near_fars[0].to(device)

    model.eval()
    psnrs = []
    for i in range(N):
        sample   = dset[i]
        rays_o   = sample['rays_o'].to(device)
        rays_d   = sample['rays_d'].to(device)
        gt       = sample['imgs'].to(device)
        bg_color = sample['bg_color'].to(device)
        t_val    = sample['timestamps'].squeeze().to(device)

        nf = near_far.unsqueeze(0) if near_far.dim() == 1 else near_far
        rgb_pred = render_batch(model, rays_o, rays_d, bg_color, nf, t_val, device)
        mse      = F.mse_loss(rgb_pred, gt).item()
        psnrs.append(-10.0 * np.log10(mse + 1e-8))

    return float(np.mean(psnrs)), psnrs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene',   required=True,
                   choices=['bouncingballs','hellwarrior','hook','jumpingjacks',
                            'lego','mutant','standup','trex'],
                   help='D-NeRF scene name')
    p.add_argument('--n-steps', type=int, default=30001)
    args = p.parse_args()

    device  = torch.device('cuda:0')
    out_dir = os.path.join(LOGS_ROOT, args.scene)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene   : {args.scene}")
    print(f"Steps   : {args.n_steps}")
    print(f"Out dir : {out_dir}")
    print(f"{'='*60}\n")

    # ── Build config + load data ──────────────────────────────────────────
    config = make_config(args.scene, args.n_steps, out_dir)
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

    # ── Train ─────────────────────────────────────────────────────────────
    kp_trainer.train()

    # ── Save model ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, args.scene, 'model.pth')
    print(f"\nModel checkpoint: {ckpt_path}")

    # ── Test PSNR ─────────────────────────────────────────────────────────
    print("\nEvaluating test PSNR with GT timestamps...")
    mean_psnr, per_img_psnrs = evaluate_test_psnr(kp_trainer, device)
    print(f"\n{'='*60}")
    print(f"Test PSNR (GT timestamps): {mean_psnr:.2f} dB")
    print(f"Per-image: {[f'{v:.2f}' for v in per_img_psnrs]}")
    print(f"{'='*60}")

    results = {'mean_psnr': mean_psnr, 'per_image_psnr': per_img_psnrs}
    with open(os.path.join(out_dir, 'train_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {out_dir}/train_results.json")


if __name__ == '__main__':
    main()
