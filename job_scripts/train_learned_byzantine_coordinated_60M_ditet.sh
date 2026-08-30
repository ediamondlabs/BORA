#!/bin/bash
#SBATCH --job-name="babel_coord_60M"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train BABEL coordinated attacker for 60M timesteps on D-ITET CPU cluster.
# Partition auto-selected as cpu.normal (no GPU requested).
#
# Usage:   sbatch job_scripts/train_learned_byzantine_coordinated_60M_ditet.sh
# Output:  out/models/learned_byzantine/29_04_2byz_coordinated_60M/byz_agent.zip
#
# Follow-up: sbatch job_scripts/eval_full_heard_coord_4000ep_ditet.sh
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

MODEL_DIR="out/models/29_04"
N_BYZ=2
N_ENVS=29
TIMESTEPS=60000000
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_${N_BYZ}byz_coordinated_60M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
HONEST_PATTERN="${MODEL_DIR}/**"

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
echo "=== Training BABEL coordinated attacker (60M steps) ==="
echo "  Honest model : $HONEST_PATTERN"
echo "  N_byz        : $N_BYZ"
echo "  N_envs       : $N_ENVS"
echo "  Timesteps    : $TIMESTEPS"
echo "  obs_scope    : coordinated"
echo "  Output       : $OUTPUT_DIR"
echo ""

python src/Utils/train_learned_byzantine.py \
    --honest-model "$HONEST_PATTERN" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --n-envs       "$N_ENVS" \
    --max-offset   4.0 \
    --timesteps    "$TIMESTEPS" \
    --obs-scope    coordinated \
    --output       "$OUTPUT_DIR"

echo ""
echo "Finished at: $(date)"
echo "Byzantine model saved to: ${OUTPUT_DIR}/byz_agent.zip"
echo "Next step: sbatch job_scripts/eval_full_heard_coord_4000ep_ditet.sh"
exit 0
