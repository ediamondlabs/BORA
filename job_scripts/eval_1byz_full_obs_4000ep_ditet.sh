#!/bin/bash
#SBATCH --job-name="babel_1byz_full_obs_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the f=1 full-vector BORA attacker, 4000 episodes split into 8
# parallel shards of 500 episodes (distinct seeds per shard via --episode-offset).
# Run after train_learned_byzantine_1byz_full_obs_30M_ditet.sh finishes.
#
# Usage:  sbatch job_scripts/eval_1byz_full_obs_4000ep_ditet.sh
# Output: /itet-stor/arbaur/net_scratch/out/evaluations/29_04_1byz_full_obs_30M_4000ep/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"
SCRATCH="/itet-stor/arbaur/net_scratch/out"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
N_BYZ=1
BYZ="${SCRATCH}/models/learned_byzantine/29_04_1byz_full_obs_30M/byz_agent.zip"
LOG_DIR="${SCRATCH}/evaluations/29_04_1byz_full_obs_30M_4000ep"
EP_LEN=2000
EPS=500           # episodes per shard
N_SHARDS=8        # 8 × 500 = 4000 episodes

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -f "$BYZ" ]; then
    echo "ERROR: Byzantine model not found at $BYZ"
    echo "Run sbatch job_scripts/train_learned_byzantine_1byz_full_obs_30M_ditet.sh first."
    exit 1
fi
mkdir -p "$LOG_DIR"

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Honest agent: $AGENT_NAME"
echo "=== Evaluating f=1 full-vector BORA  (${N_SHARDS}×${EPS}ep) -> ${LOG_DIR} ==="

PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((i * EPS))
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model   "${MODEL_DIR}/**" \
        --byz-model      "$BYZ" \
        --eval-config    "$EVAL_CONFIG" \
        --n-byz          "$N_BYZ" \
        --max-offset     4.0 \
        --obs-scope      heard \
        --attack-type    obs_full \
        --eps            "$EPS" \
        --ep-len         "$EP_LEN" \
        --log-dir        "$LOG_DIR" \
        --agent-name     "$AGENT_NAME" \
        --episode-offset "$OFFSET" \
        >> "${LOG_DIR}/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} shards. Waiting ..."
FAILED=0
for p in "${PIDS[@]}"; do
    wait "$p" || { echo "  shard PID $p FAILED"; FAILED=1; }
done

[ "$FAILED" -eq 0 ] && echo "All shards finished OK: $(date)" \
                    || { echo "Some shards failed — check shard_*.log"; exit 1; }
