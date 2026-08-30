#!/bin/bash
#SBATCH --job-name="babel_heard_resume"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Resume BABEL heard attacker training from the latest checkpoint.
# Update CHECKPOINT below to the most recent .zip in the checkpoints/ dir.
#
# Usage:   sbatch job_scripts/train_learned_byzantine_heard_60M_resume_ditet.sh
# Output:  /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_heard_60M/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"
N_BYZ=2
N_ENVS=29
TIMESTEPS=60000000
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_${N_BYZ}byz_heard_60M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
HONEST_PATTERN="${MODEL_DIR}/**"

# Update this to the last checkpoint filename before submitting:
CHECKPOINT="${OUTPUT_DIR}/checkpoints/byz_agent_7999824_steps.zip"

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo ""
echo "=== Resuming BABEL heard attacker (target: 60M steps) ==="
echo "  Checkpoint   : $CHECKPOINT"
echo "  Honest model : $HONEST_PATTERN"
echo "  N_byz        : $N_BYZ"
echo "  N_envs       : $N_ENVS"
echo "  Total target : $TIMESTEPS"
echo "  obs_scope    : heard"
echo "  Output       : $OUTPUT_DIR"
echo ""

python src/Utils/train_learned_byzantine.py \
    --honest-model "$HONEST_PATTERN" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --n-envs       "$N_ENVS" \
    --max-offset   4.0 \
    --timesteps    "$TIMESTEPS" \
    --obs-scope    heard \
    --resume-from  "$CHECKPOINT" \
    --output       "$OUTPUT_DIR"

echo ""
echo "Finished at: $(date)"
echo "Byzantine model saved to: ${OUTPUT_DIR}/byz_agent.zip"
exit 0
