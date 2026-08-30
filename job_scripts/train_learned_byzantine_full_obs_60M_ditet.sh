#!/bin/bash
#SBATCH --job-name="babel_full_obs_60M"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train BABEL full-obs attacker for 60M timesteps on D-ITET CPU cluster.
# "obs_full" mode: Byzantine agent falsifies the entire per-node obs vector
# (19-dim) rather than only the 2D user-direction offset.
#
# Usage:   sbatch job_scripts/train_learned_byzantine_full_obs_60M_ditet.sh
# Output:  /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_full_obs_60M/
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
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_${N_BYZ}byz_full_obs_60M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
HONEST_PATTERN="${MODEL_DIR}/**"

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo ""
echo "=== Training BABEL full-obs attacker (60M steps) ==="
echo "  Honest model : $HONEST_PATTERN"
echo "  N_byz        : $N_BYZ"
echo "  N_envs       : $N_ENVS"
echo "  Timesteps    : $TIMESTEPS"
echo "  attack_type  : obs_full"
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
    --attack-type  obs_full \
    --output       "$OUTPUT_DIR"

echo ""
echo "Finished at: $(date)"
echo "Byzantine model saved to: ${OUTPUT_DIR}/byz_agent.zip"
exit 0
