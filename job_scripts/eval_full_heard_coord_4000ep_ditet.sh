#!/bin/bash
#SBATCH --job-name="babel_fhc_eval_4000"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the full / heard / coordinated BABEL attackers, 4000 episodes each,
# split into 8 parallel shards of 500 episodes (via --episode-offset).
# All three models were trained with train_learned_byzantine.py, so the same
# eval entry point + sharding works for every scope.
#
#   full  : obs_full attack on the heard scope   (29_04_2byz_full_obs_60M)
#   heard : direction-lie on the heard scope      (29_04_2byz_heard_60M)
#   coord : direction-lie on the coordinated scope(29_04_2byz_coordinated_60M)
#
# Usage:  sbatch job_scripts/eval_full_heard_coord_4000ep_ditet.sh
# Output: /itet-stor/arbaur/net_scratch/out/evaluations/<name>_4000ep/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"                                   # honest victim policy
SCRATCH="/itet-stor/arbaur/net_scratch/out"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
N_BYZ=2
EP_LEN=2000
EPS=500           # episodes per shard
N_SHARDS=8        # 8 × 500 = 4000 episodes per model

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Honest agent: $AGENT_NAME"
echo ""

# name  obs_scope  attack_type
MODELS=(
    "29_04_2byz_full_obs_60M     heard        obs_full"
    "29_04_2byz_heard_60M        heard        obs_user_dir"
    "29_04_2byz_coordinated_60M  coordinated  obs_user_dir"
)

PIDS=()
for entry in "${MODELS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1; SCOPE=$2; ATK=$3
    BYZ="${SCRATCH}/models/learned_byzantine/${NAME}/byz_agent.zip"
    LOG_DIR="${SCRATCH}/evaluations/${NAME}_4000ep"
    if [ ! -f "$BYZ" ]; then
        echo "ERROR: Byzantine model not found at $BYZ"
        exit 1
    fi
    mkdir -p "$LOG_DIR"
    echo "=== ${NAME}  (scope=${SCOPE}, attack=${ATK})  ${N_SHARDS}×${EPS}ep -> ${LOG_DIR} ==="
    for i in $(seq 0 $((N_SHARDS - 1))); do
        OFFSET=$((i * EPS))
        python src/Utils/train_learned_byzantine.py --evaluate \
            --honest-model   "${MODEL_DIR}/**" \
            --byz-model      "$BYZ" \
            --eval-config    "$EVAL_CONFIG" \
            --n-byz          "$N_BYZ" \
            --max-offset     4.0 \
            --obs-scope      "$SCOPE" \
            --attack-type    "$ATK" \
            --eps            "$EPS" \
            --ep-len         "$EP_LEN" \
            --log-dir        "$LOG_DIR" \
            --agent-name     "$AGENT_NAME" \
            --episode-offset "$OFFSET" \
            >> "${LOG_DIR}/shard_${i}.log" 2>&1 &
        PIDS+=($!)
    done
done

echo ""
echo "Launched ${#PIDS[@]} shards (3 models × ${N_SHARDS}). Waiting ..."
FAILED=0
for p in "${PIDS[@]}"; do
    wait "$p" || { echo "  shard PID $p FAILED"; FAILED=1; }
done

[ "$FAILED" -eq 0 ] && echo "All shards finished OK: $(date)" \
                    || { echo "Some shards failed — check shard_*.log"; exit 1; }
