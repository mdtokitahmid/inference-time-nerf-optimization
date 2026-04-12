#!/bin/bash
# Stress-test: render hard Beta-distributed test sets and run all ablation modes.
#
# Two conditions per scene, submitted as parallel jobs:
#
#   Condition A — Beta(0.5, 0.5),  50 images
#     Same U-shape as existing test but 2.5× more images → denser packing.
#     98% of consecutive pairs fall within one grid bin (0.222).
#     47% of consecutive pairs smaller than avg random-only spacing (0.027).
#     0 ties → all pairs theoretically distinguishable if gradient signal exists.
#
#   Condition B — Beta(0.5, 0.5),  100 images
#     100% of pairs within one grid bin. 75% smaller than random-only spacing.
#     Grid-only completely blind within clusters; random-only misses 3/4 of pairs.
#     Grid+grad: systematic coverage + gradient refinement is the only viable path.
#
# Expected outcome: grid+grad wins; grid-only and random-only degrade.
#
# Usage:
#   bash submit_synthetic_hard.sh                   # all 8 scenes
#   bash submit_synthetic_hard.sh mutant trex       # specific scenes

SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)

if [ "$#" -gt 0 ]; then
    SCENES=("$@")
fi

PYTHON=/scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python
SCRIPT_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
KPLANES_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
LOG_DIR=${SCRIPT_DIR}/logs
SEED=42

# Inference hyperparams (same as before)
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

    # ── Condition A: Beta(0.5, 0.5), 50 images ───────────────────────────
    ALPHA_A=0.5; BETA_A=0.5; N_A=50
    TAG_A="_beta${ALPHA_A}_${BETA_A}_n${N_A}"
    NPZ_A=${LOG_DIR}/${SCENE}/synthetic_test_a${ALPHA_A}_b${BETA_A}_n${N_A}/test_set.npz

    JOB_A=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=hard_A_${SCENE}
#SBATCH --output=${SLURM_LOG}/hard_A_%j.out
#SBATCH --error=${SLURM_LOG}/hard_A_%j.err
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
echo "Condition : A  Beta(${ALPHA_A},${BETA_A})  n=${N_A}  [hard: 50 images]"
echo "Job ID    : \$SLURM_JOB_ID"
echo "Node      : \$SLURMD_NODENAME"
echo "Start     : \$(date)"
echo "=============================="

cd ${KPLANES_DIR}
export PYTHONPATH='.'

echo ">>> Rendering..."
${PYTHON} ${SCRIPT_DIR}/render_synthetic_test.py --scene ${SCENE} --n-images ${N_A} --beta-alpha ${ALPHA_A} --beta-beta ${BETA_A} --seed ${SEED}

echo ">>> multistart..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode multistart --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-starts ${N_STARTS} --warmup-steps ${WARMUP_STEPS} --synthetic-test ${NPZ_A} --tag ${TAG_A}

echo ">>> grid..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --top-k ${TOP_K} --synthetic-test ${NPZ_A} --tag ${TAG_A}

echo ">>> grid-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid-only --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --synthetic-test ${NPZ_A} --tag ${TAG_A}

echo ">>> random-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode random-only --n-rays ${OPT_RAYS} --n-random ${N_RANDOM} --synthetic-test ${NPZ_A} --tag ${TAG_A}

echo "=============================="
echo "Done: ${SCENE} Condition A  \$(date)"
echo "=============================="
EOF
    )
    echo "Submitted ${SCENE}/CondA → job ${JOB_A}"

    # ── Condition B: Beta(0.5, 0.5), 100 images ──────────────────────────
    ALPHA_B=0.5; BETA_B=0.5; N_B=100
    TAG_B="_beta${ALPHA_B}_${BETA_B}_n${N_B}"
    NPZ_B=${LOG_DIR}/${SCENE}/synthetic_test_a${ALPHA_B}_b${BETA_B}_n${N_B}/test_set.npz

    JOB_B=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=hard_B_${SCENE}
#SBATCH --output=${SLURM_LOG}/hard_B_%j.out
#SBATCH --error=${SLURM_LOG}/hard_B_%j.err
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
echo "Condition : B  Beta(${ALPHA_B},${BETA_B})  n=${N_B}  [very hard: 100 images]"
echo "Job ID    : \$SLURM_JOB_ID"
echo "Node      : \$SLURMD_NODENAME"
echo "Start     : \$(date)"
echo "=============================="

cd ${KPLANES_DIR}
export PYTHONPATH='.'

echo ">>> Rendering ${N_B} images (very hard test)..."
${PYTHON} ${SCRIPT_DIR}/render_synthetic_test.py --scene ${SCENE} --n-images ${N_B} --beta-alpha ${ALPHA_B} --beta-beta ${BETA_B} --seed ${SEED}

echo ">>> multistart..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode multistart --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-starts ${N_STARTS} --warmup-steps ${WARMUP_STEPS} --synthetic-test ${NPZ_B} --tag ${TAG_B}

echo ">>> grid..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --top-k ${TOP_K} --synthetic-test ${NPZ_B} --tag ${TAG_B}

echo ">>> grid-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid-only --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --synthetic-test ${NPZ_B} --tag ${TAG_B}

echo ">>> random-only..."
${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode random-only --n-rays ${OPT_RAYS} --n-random ${N_RANDOM} --synthetic-test ${NPZ_B} --tag ${TAG_B}

echo "=============================="
echo "Done: ${SCENE} Condition B  \$(date)"
echo "=============================="
EOF
    )
    echo "Submitted ${SCENE}/CondB → job ${JOB_B}"

done
