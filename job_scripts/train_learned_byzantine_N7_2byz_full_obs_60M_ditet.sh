#!/bin/bash
#SBATCH --job-name="N7_full_obs_60M"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train the BORA full-vector attacker against the N=7 honest victim, f=2, 60M
# steps (same budget as the N=5 full_obs run). Tests whether a larger honest
# majority (7 relay nodes against 2 Byzantine) reduces the full-vector attack.
# numbOfNodes=7 comes from the N=7 victim's own config; no override needed.
#
# Usage:   sbatch job_scripts/train_learned_byzantine_N7_2byz_full_obs_60M_ditet.sh
# Output:  /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/18_05_N7_2byz_full_obs_60M/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

HONEST_PATTERN="/itet-stor/arbaur/net_scratch/out/models/18_05_N7/**"   # N=7 victim policy
N_BYZ=2
N_ENVS=29
TIMESTEPS=60000000
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/18_05_N7_2byz_full_obs_60M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "=== Training BORA full-vector attacker, N=7, f=${N_BYZ}, ${TIMESTEPS} steps ==="
echo "  Honest model : $HONEST_PATTERN"
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
