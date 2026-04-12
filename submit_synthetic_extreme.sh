#!/bin/bash
# Extreme narrow-window test: all images sampled from a tiny t range.
# This is designed so that:
#   grid-only  → all images fall in one bin → random ordering → ρ ≈ 0
#   random-only→ ~7-8 of 75 samples land in window → can't resolve → ρ ≈ 0
#   multistart → <10% chance any start lands in window → fails → ρ ≈ 0
#   grid+grad  → grid finds the nearest bin edge, gradient descent resolves within → ρ ≈ 1
#
# Two conditions, one job per scene per condition:
#
#   --condition window   : Uniform[-0.1, 0.1], 100 images  (THE extreme test)
#   --condition window-hard : Uniform[-0.05, 0.05], 50 images  (even tighter)
#
# Usage:
#   bash submit_synthetic_extreme.sh                       # all scenes, both conditions
#   bash submit_synthetic_extreme.sh --condition window mutant trex
#   bash submit_synthetic_extreme.sh --condition window-hard mutant

# ── Parse --condition flag ──────────────────────────────────────────────────
CONDITION="both"    # default: run both conditions
SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)
            CONDITION="$2"; shift 2 ;;
        *)
            POSITIONAL+=("$1"); shift ;;
    esac
done

if [ "${#POSITIONAL[@]}" -gt 0 ]; then
    SCENES=("${POSITIONAL[@]}")
fi

if [[ "$CONDITION" != "window" && "$CONDITION" != "window-hard" && "$CONDITION" != "both" ]]; then
    echo "Error: --condition must be one of: window, window-hard, both"
    exit 1
fi

PYTHON=/scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python
SCRIPT_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
KPLANES_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
LOG_DIR=${SCRIPT_DIR}/logs
SEED=42

# Inference hyperparams
OPT_STEPS=50
OPT_LR=0.05
OPT_RAYS=2048
N_STARTS=5
WARMUP_STEPS=5
N_GRID=10
TOP_K=3
N_RANDOM=75

run_condition() {
    local SCENE="$1"
    local COND_NAME="$2"
    local T_MIN="$3"
    local T_MAX="$4"
    local N_IMGS="$5"

    local SLURM_LOG=${LOG_DIR}/${SCENE}/slurm
    mkdir -p "${SLURM_LOG}"

    local TAG="_window_${T_MIN}_${T_MAX}_n${N_IMGS}"
    local NPZ=${LOG_DIR}/${SCENE}/synthetic_test_window_${T_MIN}_${T_MAX}_n${N_IMGS}/test_set.npz

    local JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=extreme_${COND_NAME}_${SCENE}
#SBATCH --output=${SLURM_LOG}/extreme_${COND_NAME}_%j.out
#SBATCH --error=${SLURM_LOG}/extreme_${COND_NAME}_%j.err
#SBATCH --account=mona
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00

echo "=============================="
echo "Scene     : ${SCENE}"
echo "Condition : ${COND_NAME}  t in [${T_MIN}, ${T_MAX}]  n=${N_IMGS}"
echo "Job ID    : \$SLURM_JOB_ID"
echo "Node      : \$SLURMD_NODENAME"
echo "Start     : \$(date)"
echo "=============================="

cd ${KPLANES_DIR}
export PYTHONPATH='.'

echo ">>> Rendering ${N_IMGS} images in window [${T_MIN}, ${T_MAX}]..."
${PYTHON} ${SCRIPT_DIR}/render_synthetic_test.py --scene ${SCENE} --n-images ${N_IMGS} --t-min ${T_MIN} --t-max ${T_MAX} --seed ${SEED}

echo ">>> multistart..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode multistart --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-starts ${N_STARTS} --warmup-steps ${WARMUP_STEPS} --synthetic-test ${NPZ} --tag ${TAG}

echo ">>> grid..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --top-k ${TOP_K} --synthetic-test ${NPZ} --tag ${TAG}

echo ">>> grid-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid-only --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --synthetic-test ${NPZ} --tag ${TAG}

echo ">>> random-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode random-only --n-rays ${OPT_RAYS} --n-random ${N_RANDOM} --synthetic-test ${NPZ} --tag ${TAG}

echo "=============================="
echo "Done: ${SCENE} ${COND_NAME}  \$(date)"
echo "=============================="
EOF
    )
    echo "Submitted ${SCENE}/${COND_NAME} → job ${JOB_ID}   logs: ${SLURM_LOG}/extreme_${COND_NAME}_${JOB_ID}.out"
}

# ── Submit jobs ─────────────────────────────────────────────────────────────
for SCENE in "${SCENES[@]}"; do
    if [[ "$CONDITION" == "window" || "$CONDITION" == "both" ]]; then
        run_condition "$SCENE" "window"      -0.1  0.1  100
    fi
    if [[ "$CONDITION" == "window-hard" || "$CONDITION" == "both" ]]; then
        run_condition "$SCENE" "window-hard" -0.05 0.05  50
    fi
done
