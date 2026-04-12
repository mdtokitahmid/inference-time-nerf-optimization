#!/bin/bash
# Submit ablation jobs (multistart + grid) for D-NeRF scenes.
# Requires trained model.pth in logs/<scene>/<scene>/model.pth (from submit_all.sh).
# Does NOT re-train; inference only.
#
# Usage:
#   cd /scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
#   bash submit_ablation.sh                     # all 8 scenes, both modes
#   bash submit_ablation.sh mutant lego trex    # specific scenes, both modes

SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)

if [ "$#" -gt 0 ]; then
    SCENES=("$@")
fi

PYTHON=/scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python
SCRIPT_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
KPLANES_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
LOG_DIR=${SCRIPT_DIR}/logs

OPT_STEPS=50
OPT_LR=0.05
OPT_RAYS=2048

# Multi-start hyperparams
N_STARTS=5
WARMUP_STEPS=5

# Grid search hyperparams
N_GRID=10
TOP_K=3

# Random-only hyperparams (75 = same forward-pass budget as multistart)
N_RANDOM=75

for SCENE in "${SCENES[@]}"; do
    SLURM_LOG=${LOG_DIR}/${SCENE}/slurm
    mkdir -p "${SLURM_LOG}"

    for MODE in multistart grid grid-only random-only; do

        JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ablation_${MODE}_${SCENE}
#SBATCH --output=${SLURM_LOG}/ablation_${MODE}_%j.out
#SBATCH --error=${SLURM_LOG}/ablation_${MODE}_%j.err
#SBATCH --account=mona
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

echo "=============================="
echo "Scene  : ${SCENE}"
echo "Mode   : ${MODE}"
echo "Job ID : \$SLURM_JOB_ID"
echo "Node   : \$SLURMD_NODENAME"
echo "Start  : \$(date)"
echo "=============================="

cd ${KPLANES_DIR}
export PYTHONPATH='.'

if [ "${MODE}" = "multistart" ]; then
    ${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode multistart --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-starts ${N_STARTS} --warmup-steps ${WARMUP_STEPS}
elif [ "${MODE}" = "grid" ]; then
    ${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid --n-steps ${OPT_STEPS} --lr ${OPT_LR} --n-rays ${OPT_RAYS} --n-grid ${N_GRID} --top-k ${TOP_K}
elif [ "${MODE}" = "grid-only" ]; then
    ${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode grid-only --n-rays ${OPT_RAYS} --n-grid ${N_GRID}
else
    ${PYTHON} ${SCRIPT_DIR}/infer_ablation.py --scene ${SCENE} --mode random-only --n-rays ${OPT_RAYS} --n-random ${N_RANDOM}
fi

echo ""
echo "=============================="
echo "Done: ${SCENE} ${MODE}  \$(date)"
echo "=============================="
EOF
        )

        echo "Submitted ${SCENE}/${MODE} → job ${JOB_ID}   logs: ${SLURM_LOG}/ablation_${MODE}_${JOB_ID}.out"
    done
done
