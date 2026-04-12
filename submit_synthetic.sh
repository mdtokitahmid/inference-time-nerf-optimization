#!/bin/bash
# Render Beta-distributed synthetic test sets and run all ablation modes on them.
# Each scene gets one job: render → multistart, grid, grid-only, random-only.
# Requires trained model.pth (from submit_all.sh).
#
# Usage:
#   bash submit_synthetic.sh                      # all 8 scenes
#   bash submit_synthetic.sh mutant lego trex     # specific scenes

SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)

if [ "$#" -gt 0 ]; then
    SCENES=("$@")
fi

PYTHON=/scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python
SCRIPT_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
KPLANES_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
LOG_DIR=${SCRIPT_DIR}/logs

# Synthetic test set config
N_IMAGES=20
BETA_ALPHA=0.5
BETA_BETA=0.5
SEED=42
TAG="_beta${BETA_ALPHA}_${BETA_BETA}"

# Inference config
OPT_STEPS=50
OPT_LR=0.05
OPT_RAYS=2048
N_STARTS=5
WARMUP_STEPS=5
N_GRID=10
TOP_K=3
N_RANDOM=75

for SCENE in "${SCENES[@]}"; do
    SLURM_LOG=${LOG_DIR}/${SCENE}/slurm
    mkdir -p "${SLURM_LOG}"

    NPZ_PATH=${LOG_DIR}/${SCENE}/synthetic_test_a${BETA_ALPHA}_b${BETA_BETA}/test_set.npz

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=synthetic_${SCENE}
#SBATCH --output=${SLURM_LOG}/synthetic_%j.out
#SBATCH --error=${SLURM_LOG}/synthetic_%j.err
#SBATCH --account=mona
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00

echo "=============================="
echo "Scene  : ${SCENE}"
echo "Job ID : \$SLURM_JOB_ID"
echo "Node   : \$SLURMD_NODENAME"
echo "Start  : \$(date)"
echo "=============================="

cd ${KPLANES_DIR}
export PYTHONPATH='.'

# ── Step 1: Render synthetic test set ────────────────────────────────────
echo ""
echo ">>> Rendering synthetic test set  Beta(${BETA_ALPHA}, ${BETA_BETA})..."
${PYTHON} ${SCRIPT_DIR}/render_synthetic_test.py --scene ${SCENE} --n-images ${N_IMAGES} --beta-alpha ${BETA_ALPHA} --beta-beta ${BETA_BETA} --seed ${SEED}

# ── Step 2: Run all ablation modes on synthetic test set ─────────────────
echo ""
echo ">>> multistart..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode multistart --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-starts ${N_STARTS} --warmup-steps ${WARMUP_STEPS} --synthetic-test ${NPZ_PATH} --tag ${TAG}

echo ""
echo ">>> grid..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --top-k ${TOP_K} --synthetic-test ${NPZ_PATH} --tag ${TAG}

echo ""
echo ">>> grid-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid-only --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --synthetic-test ${NPZ_PATH} --tag ${TAG}

echo ""
echo ">>> random-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode random-only --n-rays ${OPT_RAYS} --n-random ${N_RANDOM} --synthetic-test ${NPZ_PATH} --tag ${TAG}

echo ""
echo "=============================="
echo "Done: ${SCENE}  \$(date)"
echo "=============================="
EOF
    )

    echo "Submitted ${SCENE} → job ${JOB_ID}   logs: ${SLURM_LOG}/synthetic_${JOB_ID}.out"
done
