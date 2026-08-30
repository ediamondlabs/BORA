#!/bin/bash
#SBATCH --job-name="babel_fullscope_30M"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train a full-vector BORA attacker for 30M timesteps with the FULL observation
# scope: the Byzantine policy sees the entire global state (every node's obs)
# when deciding its manipulation (--obs-scope full), and falsifies the entire
# per-node broadcast vector (--attack-type obs_full).
#
# This is the oracle-information version of the full-vector attack — contrast
# with the main BORA (29_04_2byz_full_obs_60M), which uses the realistic
# "heard" scope (own state + in-range neighbours only).
#
# Usage:   sbatch job_scripts/train_learned_byzantine_full_obs_fullscope_30M_ditet.sh
# Output:  /itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_full_obs_fullscope_30M/
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
TIMESTEPS=30000000
SCRATCH="/itet-stor/arbaur/net_scratch/out"
OUTPUT_DIR="${SCRATCH}/models/learned_byzantine/29_04_${N_BYZ}byz_full_obs_fullscope_30M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
HONEST_PATTERN="${MODEL_DIR}/**"
# --- evaluation (runs automatically after training) ---
EVAL_LOG_DIR="${SCRATCH}/evaluations/29_04_2byz_full_obs_fullscope_30M_4000ep"
EP_LEN=2000
EPS=500          # per shard
N_SHARDS=8       # 8 x 500 = 4000 episodes

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
echo "=== Training full-vector BORA, FULL observation scope (30M steps) ==="
echo "  Honest model : $HONEST_PATTERN"
echo "  N_byz        : $N_BYZ"
echo "  N_envs       : $N_ENVS"
echo "  Timesteps    : $TIMESTEPS"
echo "  attack_type  : obs_full"
echo "  obs_scope    : full   (attacker sees the entire global state)"
echo "  Output       : $OUTPUT_DIR"
echo ""

python src/Utils/train_learned_byzantine.py \
    --honest-model "$HONEST_PATTERN" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --n-envs       "$N_ENVS" \
    --max-offset   4.0 \
    --timesteps    "$TIMESTEPS" \
    --obs-scope    full \
    --attack-type  obs_full \
    --output       "$OUTPUT_DIR"

echo ""
echo "Training finished at: $(date)"
echo "Byzantine model saved to: ${OUTPUT_DIR}/byz_agent.zip"

# ===========================================================================
# Evaluation — runs automatically once training completes (8 x 500 = 4000 ep,
# distinct seeds per shard). Delta-tau is computed against the existing
# full-state no-Byzantine baseline (29_04_nobyz_4000ep, tau0 = 16.78).
# ===========================================================================
BYZ="${OUTPUT_DIR}/byz_agent.zip"
if [ ! -f "$BYZ" ]; then
    echo "ERROR: trained model not found at $BYZ — skipping eval."
    exit 1
fi
mkdir -p "$EVAL_LOG_DIR"

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo ""
echo "=== Evaluating full-scope BORA (${N_SHARDS}x${EPS}ep) -> ${EVAL_LOG_DIR} ==="
echo "  Honest agent: $AGENT_NAME"

PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((i * EPS))
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model   "$HONEST_PATTERN" \
        --byz-model      "$BYZ" \
        --eval-config    "$EVAL_CONFIG" \
        --n-byz          "$N_BYZ" \
        --max-offset     4.0 \
        --obs-scope      full \
        --attack-type    obs_full \
        --eps            "$EPS" \
        --ep-len         "$EP_LEN" \
        --log-dir        "$EVAL_LOG_DIR" \
        --agent-name     "$AGENT_NAME" \
        --episode-offset "$OFFSET" \
        >> "${EVAL_LOG_DIR}/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || { echo "  eval shard PID $p FAILED"; FAILED=1; }; done

echo ""
echo "Finished at: $(date)  (eval failures: ${FAILED})"
echo "Eval output: ${EVAL_LOG_DIR}/  (Delta-tau vs 29_04_nobyz_4000ep = 16.78)"
exit 0
