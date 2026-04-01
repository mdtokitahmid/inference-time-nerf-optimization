#!/bin/bash
# Submit one training + inference job per D-NeRF scene to Della GPU cluster.
# Each scene runs in its own SLURM job independently.
#
# Usage:
#   cd /scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
#   bash submit_all.sh
#
# Optional: train only some scenes
#   bash submit_all.sh mutant lego trex

SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)

# If specific scenes passed as args, use those instead
if [ "$#" -gt 0 ]; then
    SCENES=("$@")
fi

PYTHON=/scratch/gpfs/MONA/mt3204/envs/kplanes/bin/python
SCRIPT_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/final_experiments
KPLANES_DIR=/scratch/gpfs/MONA/Toki/Academic/COS526/K-Planes
LOG_DIR=${SCRIPT_DIR}/logs

N_STEPS=30001
OPT_STEPS=50
OPT_LR=0.05
OPT_RAYS=2048

for SCENE in "${SCENES[@]}"; do
    SLURM_LOG=${LOG_DIR}/${SCENE}/slurm
    mkdir -p "${SLURM_LOG}"

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=kplanes_${SCENE}
#SBATCH --output=${SLURM_LOG}/train_%j.out
#SBATCH --error=${SLURM_LOG}/train_%j.err
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

# ── Train ────────────────────────────────────────────────────────────────
echo ""
echo ">>> Training..."
${PYTHON} ${SCRIPT_DIR}/train.py --scene ${SCENE} --n-steps ${N_STEPS}

# ── Infer ────────────────────────────────────────────────────────────────
echo ""
echo ">>> Running test-time t optimization..."
${PYTHON} ${SCRIPT_DIR}/infer.py \
    --scene   ${SCENE} \
    --n-steps ${OPT_STEPS} \
    --lr      ${OPT_LR} \
    --n-rays  ${OPT_RAYS}

echo ""
echo "=============================="
echo "Done: ${SCENE}  \$(date)"
echo "=============================="
EOF
    )

    echo "Submitted ${SCENE} → job ${JOB_ID}   logs: ${SLURM_LOG}/train_${JOB_ID}.out"
done
