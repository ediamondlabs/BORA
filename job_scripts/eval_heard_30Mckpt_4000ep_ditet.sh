#!/bin/bash
#SBATCH --job-name="heard_30M_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Convergence check for the heard (direction-only) BORA variant, mirroring the
# coord 30M-vs-60M comparison. heard already trained to 60M (eval = 34.5%); this
# evaluates its ~30M checkpoint at 4000 episodes. If the 30M eval matches the
# 60M, heard converged (its low score is real); if it is much weaker, heard was
# still under-trained at 60M.
#
# Usage:  sbatch job_scripts/eval_heard_30Mckpt_4000ep_ditet.sh
# Output: out/evaluations/29_04_2byz_heard_30M_4000ep/learned_obs_user_dir/
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
HEARD_DIR="${SCRATCH}/models/learned_byzantine/29_04_2byz_heard_60M"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
LOG_DIR="out/evaluations/29_04_2byz_heard_30M_4000ep"
N_BYZ=2
EP_LEN=2000
EPS=500
N_SHARDS=8
TARGET_STEPS=30000000

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs "$LOG_DIR"

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

# Pick the checkpoint whose step count is closest to 30M.
BYZ=$(for f in "${HEARD_DIR}"/checkpoints/byz_agent_*_steps.zip; do
        s=$(basename "$f" | sed 's/byz_agent_//; s/_steps.zip//')
        d=$(( s > TARGET_STEPS ? s - TARGET_STEPS : TARGET_STEPS - s ))
        echo "$d $f"
      done | sort -n | head -1 | cut -d' ' -f2-)

if [ -z "$BYZ" ] || [ ! -f "$BYZ" ]; then
    echo "ERROR: no heard checkpoint near 30M found under ${HEARD_DIR}/checkpoints"
    ls "${HEARD_DIR}/checkpoints" 2>/dev/null | head
    exit 1
fi
echo "Using checkpoint: $BYZ"

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')

echo "=== Evaluating heard 30M checkpoint (${N_SHARDS}x${EPS}ep) -> ${LOG_DIR} ==="
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
        --attack-type    obs_user_dir \
        --eps            "$EPS" \
        --ep-len         "$EP_LEN" \
        --log-dir        "$LOG_DIR" \
        --agent-name     "$AGENT_NAME" \
        --episode-offset "$OFFSET" \
        >> "${LOG_DIR}/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
[ "$FAILED" -eq 0 ] && echo "All shards done $(date)" || { echo "Some shards failed"; exit 1; }
