# COS526 Final Project — Experimental Results

**Task**: Recover per-image timestamps from a dynamic NeRF scene at *test time*, given only pixels
and known camera poses (no GT timestamps used during inference).

**Method**: Train K-Planes with GT poses + GT timestamps. Freeze weights. For each test image,
optimize a single scalar `t = tanh(t_raw)` via gradient descent to minimize pixel reconstruction
MSE against the rendered output.

---

## Experiment 1 — Full Pipeline (train.py → infer.py)

All 8 D-NeRF scenes trained for **30,001 steps** with GT poses + GT timestamps.
Test-time timestamp recovery: **50 gradient steps**, lr=0.05, 2048 random rays per image.

### 1a. Training PSNR (GT timestamps, full test set)

> Source: `logs/<scene>/train_results.json`

| Scene          | Mean PSNR (dB) | Min  | Max  |
|----------------|---------------|------|------|
| bouncingballs  | 39.10         | 35.16| 42.94|
| hellwarrior    | 24.83         | 22.98| 27.30|
| hook           | 27.54         | 24.17| 31.42|
| jumpingjacks   | 31.35         | 25.79| 37.63|
| lego           | 25.37         | 23.73| 26.73|
| mutant         | 31.84         | 28.06| 35.27|
| standup        | 32.22         | 26.73| 36.34|
| trex           | 30.27         | 27.10| 34.10|

All 20 per-image PSNRs are stored in `per_image_psnr` in each `train_results.json`.

---

### 1b. Test-Time Timestamp Recovery

> Source: `logs/<scene>/infer/results.json`

Optimization recovers t ∈ (−1, 1) for each test image independently (no inter-image information used).
Spearman correlation measures how well the *ordering* of recovered timestamps matches GT ordering.

| Scene         | Spearman ρ | Mean \|err\| | Median \|err\| | Verdict       |
|---------------|-----------|-------------|---------------|---------------|
| mutant        | **0.788** | 0.192       | 0.024         | Works well    |
| standup       | **0.785** | 0.174       | 0.012         | Works well    |
| bouncingballs | 0.341     | 0.467       | 0.246         | Partial       |
| lego          | 0.253     | 0.512       | 0.127         | Partial       |
| hook          | 0.113     | 0.473       | 0.057         | Near-random   |
| hellwarrior   | −0.143    | 0.730       | 0.720         | Fails         |
| jumpingjacks  | **−0.220**| 0.871       | 0.955         | Fails (periodic) |
| trex          | **−0.287**| 0.844       | 0.969         | Fails (periodic) |

**Interpretation:**
- High Spearman (mutant, standup): monotonic, non-repetitive motion → single loss minimum near GT t
- Near-zero / negative Spearman (jumpingjacks, trex, hellwarrior): periodic or symmetric motion →
  multiple t values produce similar pixel loss, optimization lands at wrong local minimum
- Median error often much lower than mean → a few catastrophic failures inflate the mean

---

## File Structure

```
final_experiments/
├── train.py                         # Train K-Planes with GT poses + GT timestamps
├── infer.py                         # Freeze model, optimize t per test image
├── submit_all.sh                    # SLURM launcher (one job per scene)
├── make_gifs.py                     # Generate annotated GIFs from GT test images
├── RESULTS.md                       # This file
│
├── logs/
│   └── <scene>/
│       ├── train_results.json       # mean_psnr, per_image_psnr (20 values)
│       ├── <scene>/
│       │   └── model.pth            # Trained K-Planes checkpoint
│       ├── infer/
│       │   ├── results.json         # spearman, mean/median t_err, gt_times, opt_times
│       │   ├── t_vs_step.png        # Per-image: recovered t converging toward GT (dashed)
│       │   ├── psnr_vs_step.png     # Per-image: PSNR vs optimization step
│       │   └── spearman_vs_step.png # Global Spearman ρ vs step (all images)
│       └── slurm/
│           └── train_<jobid>.out    # SLURM stdout/stderr
│
└── gifs/
    └── <scene>.gif                  # GT test images annotated with GT timestamp, 0.5s/frame
```

### Key Fields in results.json

| Field           | Type         | Description                                                   |
|-----------------|--------------|---------------------------------------------------------------|
| `spearman`      | float        | Spearman ρ between gt_times and opt_times (higher = better)   |
| `mean_t_err`    | float        | Mean absolute error \|gt_t − recovered_t\|                    |
| `median_t_err`  | float        | Median absolute error (more robust to outliers)               |
| `gt_times`      | list[float]  | Ground-truth timestamps for 20 test images, range ≈ [−1, 1]  |
| `opt_times`     | list[float]  | Recovered timestamps after 50 optimization steps              |

---

## SLURM Job Info

- Cluster: Della (Princeton)
- Account: `mona`, QOS: `gpu-medium`
- Resources: 1 GPU, 4 CPUs, 16 GB RAM, 59 min wall time
- Each job: `train.py` (≈30k steps) → `infer.py` (50 steps × 20 images)

---

## Ablations (from earlier experiments)

> Conducted on `mutant` scene using `experiments/train_joint.py` (separate codebase).

| Condition                              | Spearman ρ | Notes                                          |
|----------------------------------------|-----------|------------------------------------------------|
| Reconstruction loss + smoothness reg   | ~0.93     | Cheating: smoothness uses consecutive indices  |
| Reconstruction loss only (random init) | ~0.08     | Near-random; loss alone cannot order images    |
| Reconstruction loss only (uniform init)| ~0.93     | High ρ comes from initialization, not learning |
| **Test-time t optimization (this work)**| **0.788** | No cheating; only pixels + pose used           |

**Key finding**: The reconstruction loss landscape is not reliably monotone in t — pixel MSE has
multiple local minima, especially for scenes with repeating or symmetric motion patterns.
Test-time optimization succeeds when the scene has a single dominant temporal motion signature.

---

## Plots Guide

| Plot                    | Location                          | What it shows                                              |
|-------------------------|-----------------------------------|------------------------------------------------------------|
| `t_vs_step.png`         | `logs/<scene>/infer/`             | One subplot per test image: recovered t (blue) vs GT t (red dashed) across 50 steps |
| `psnr_vs_step.png`      | `logs/<scene>/infer/`             | One subplot per test image: PSNR (dB) vs optimization step |
| `spearman_vs_step.png`  | `logs/<scene>/infer/`             | Single plot: global Spearman ρ across all images vs step   |
| `<scene>.gif`           | `gifs/`                           | GT test images sorted by time, with timestamp overlay      |

---

*Last updated: 2026-03-31*
