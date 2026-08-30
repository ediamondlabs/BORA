#!/bin/bash
#SBATCH --job-name="babel_1byz_eval_par"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the BABEL 1-byz learned attacker (local obs scope) for 4000 episodes
# using 4 parallel shards of 1000 episodes each (~4x speedup).
#
# Usage:   sbatch job_scripts/eval_learned_byzantine_1byz_local_4000ep_parallel_ditet.sh
# Output:  /itet-stor/arbaur/net_scratch/out/evaluations/29_04_learned_1byz_local_4000ep/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"
N_BYZ=1
BYZ_MODEL="out/models/learned_byzantine/29_04_${N_BYZ}byz_local/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
BASELINE_SOURCE="/itet-stor/arbaur/net_scratch/out/evaluations/29_04_jammer_2byz_v3_mag4.0"

N_EPS_PER_SHARD=1000
EP_LEN=2000
N_SHARDS=4
EXP_NAME="29_04_learned_1byz_local_4000ep"
LOG_DIR="/itet-stor/arbaur/net_scratch/out/evaluations/${EXP_NAME}"

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs
mkdir -p "$LOG_DIR"

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo ""

if [ ! -f "$BYZ_MODEL" ]; then
    echo "ERROR: Byzantine model not found at $BYZ_MODEL"
    exit 1
fi

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')

echo "=== Evaluating BABEL 1-byz learned attacker (${N_SHARDS} × ${N_EPS_PER_SHARD} episodes in parallel) ==="
echo "  Byzantine model : $BYZ_MODEL"
echo "  Episodes        : $((N_SHARDS * N_EPS_PER_SHARD)) total (${N_EPS_PER_SHARD} per shard)"
echo "  obs_scope       : local"
echo "  Output          : $LOG_DIR"
echo ""

PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((i * N_EPS_PER_SHARD))
    echo "  Starting shard $i (episodes ${OFFSET}–$((OFFSET + N_EPS_PER_SHARD - 1))) ..."
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model    "${MODEL_DIR}/**" \
        --byz-model       "$BYZ_MODEL" \
        --eval-config     "$EVAL_CONFIG" \
        --n-byz           "$N_BYZ" \
        --max-offset      4.0 \
        --obs-scope       local \
        --eps             "$N_EPS_PER_SHARD" \
        --ep-len          "$EP_LEN" \
        --log-dir         "$LOG_DIR" \
        --agent-name      "$AGENT_NAME" \
        --episode-offset  "$OFFSET" \
        >> "${LOG_DIR}/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${N_SHARDS} shards launched. Waiting for completion ..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  ERROR: shard $i (PID ${PIDS[$i]}) failed"
        FAILED=1
    else
        echo "  Shard $i done."
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more shards failed. Check shard_*.log in $LOG_DIR"
    exit 1
fi

echo ""
echo "=== Linking no_byzantine baseline ==="
LINK="${LOG_DIR}/no_byzantine"
if [ -L "$LINK" ] || [ -e "$LINK" ]; then
    echo "  Already present: no_byzantine (skipping)"
else
    ln -s "${BASELINE_SOURCE}/no_byzantine" "$LINK"
    echo "  Linked: no_byzantine → ${BASELINE_SOURCE}/no_byzantine"
fi

echo ""
echo "Finished at: $(date)"
echo "CSVs written to: $LOG_DIR"
exit 0
