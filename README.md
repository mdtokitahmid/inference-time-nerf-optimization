# Inference-Time NeRF Optimization

**COS526 Final Project** — Recovering per-image timestamps from a dynamic NeRF scene at test time,
using only pixels and known camera poses (no ground-truth timestamps at inference).

## Method

1. Train [K-Planes](https://github.com/sarafridov/K-Planes) on a D-NeRF scene with GT poses + GT timestamps
2. Freeze the model weights
3. For each test image, optimize a single scalar `t = tanh(t_raw)` via Adam to minimize pixel reconstruction MSE
4. Evaluate recovered timestamps against GT using Spearman correlation

## Results Summary

| Scene         | Train PSNR (dB) | Spearman ρ | Mean \|t err\| |
|---------------|----------------|-----------|---------------|
| bouncingballs | 39.10          | 0.341     | 0.467         |
| hellwarrior   | 24.83          | −0.143    | 0.730         |
| hook          | 27.54          | 0.113     | 0.473         |
| jumpingjacks  | 31.35          | −0.220    | 0.871         |
| lego          | 25.37          | 0.253     | 0.512         |
| mutant        | 31.84          | **0.788** | 0.192         |
| standup       | 32.22          | **0.785** | 0.174         |
| trex          | 30.27          | −0.287    | 0.844         |

Works well for scenes with monotonic non-repetitive motion (mutant, standup).
Fails for periodic/symmetric motion (jumpingjacks, trex) due to multiple local minima.

## Setup

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/mdtokitahmid/inference-time-nerf-optimization.git
cd inference-time-nerf-optimization
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib scipy imageio Pillow
# K-Planes additional deps
pip install -r K-Planes/requirements.txt
```

### 3. Download the D-NeRF dataset

Get it from the [D-NeRF repo](https://github.com/albertpumarola/D-NeRF) and extract so the structure is:

```
/path/to/dnerf/data/
├── bouncingballs/
├── hellwarrior/
├── hook/
├── jumpingjacks/
├── lego/
├── mutant/
├── standup/
└── trex/
```

### 4. Update paths

Edit the top of `train.py` and `infer.py`:

```python
KPLANES_DIR = "/path/to/inference-time-nerf-optimization/K-Planes"
DATA_ROOT   = "/path/to/dnerf/data"
LOGS_ROOT   = "/path/to/your/logs"
```

## Usage

### Train a single scene

```bash
cd K-Planes
PYTHONPATH='.' python ../train.py --scene mutant --n-steps 30001
```

### Recover timestamps at test time

```bash
PYTHONPATH='.' python ../infer.py --scene mutant --n-steps 50 --lr 0.05 --n-rays 2048
```

Outputs saved to `logs/<scene>/infer/`:
- `results.json` — Spearman ρ, mean/median |t error|, gt and recovered timestamps
- `t_vs_step.png` — recovered t converging toward GT per image
- `psnr_vs_step.png` — PSNR vs optimization step per image
- `spearman_vs_step.png` — global Spearman ρ vs step

### Run all 8 scenes on SLURM (Princeton Della)

```bash
bash submit_all.sh               # all 8 scenes
bash submit_all.sh mutant lego   # specific scenes
```

Each job trains + runs inference sequentially (~50 min/scene on a single GPU).

### Generate GIFs from GT test images

```bash
python make_gifs.py
```

Produces annotated GIFs (timestamp overlaid) in `gifs/`.

## Repository Structure

```
├── train.py          # Train K-Planes with GT poses + timestamps
├── infer.py          # Test-time timestamp recovery
├── submit_all.sh     # SLURM launcher for Della cluster
├── make_gifs.py      # Generate annotated GIFs from test images
├── RESULTS.md        # Detailed results, tables, and file format docs
├── K-Planes/         # Submodule — forked from sarafridov/K-Planes
└── logs/             # (gitignored) training checkpoints and inference outputs
```

## Citation

```bibtex
@article{fridovich2023kplanes,
  title={K-Planes: Explicit Radiance Fields in Space, Time, and Appearance},
  author={Fridovich-Keil, Sara and Meanti, Giacomo and Warburg, Frederik and Recht, Benjamin and Kanazawa, Angjoo},
  journal={CVPR},
  year={2023}
}
```
