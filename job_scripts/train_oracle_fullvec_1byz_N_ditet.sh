#!/bin/bash
#SBATCH --job-name="ofv_1byz_N"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train + auto-eval a 1-Byzantine FULL-VECTOR attacker at the ORACLE scope
# (--obs-scope full, --attack-type obs_full) against a frozen N-node honest
# victim, for TOTAL node counts N in {6, 8, 9}. Fills the 1-byz N-scaling gap
# between the existing oracle_fullvec_1byz_N5 and oracle_fullvec_1byz_N10 cells
# (same scope/attack as train_oracle_fullvec_30M_ditet.sh, fixed to n_byz=1).
#
# N is TOTAL nodes (so honest = N-1, Byzantine = 1). numbOfNodes is inherited
# from the victim's own config; no override needed.
#
# Requires (per N): the frozen honest victim + its matched no-byz baseline:
#   victim   : .../net_scratch/out/models/17_06_N<N>/                  (train_honest_no_obs_attack_ditet.sh <N>)
#   baseline : .../net_scratch/out/evaluations/17_06_N<N>_nobyz_4000ep/ (eval_nobyz_N_scaling_4000ep_ditet.sh)
#
# Usage (submit once per N — three separate jobs):
#   sbatch job_scripts/train_oracle_fullvec_1byz_N_ditet.sh 6
#   sbatch job_scripts/train_oracle_fullvec_1byz_N_ditet.sh 8
#   sbatch job_scripts/train_oracle_fullvec_1byz_N_ditet.sh 9
#
# Output (net_scratch):
#   models/learned_byzantine/oracle_fullvec_1byz_N<N>/byz_agent.zip
#   evaluations/oracle_fullvec_1byz_N<N>_2000ep/   (Delta-tau vs 17_06_N<N>_nobyz_4000ep)
#
# To use the Gossip (heard) scope instead of the oracle, swap both
# "--obs-scope full" occurrences below for "--obs-scope heard".
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N="$1"
if [ -z "$N" ]; then
    echo "Usage: sbatch $0 <N>   (total relay nodes; 6, 8, or 9)"
    exit 1
fi

# N -> victim model-dir prefix (N=6/8/9 trained under 17_06; N=10 under 16_06)
case "$N" in
    10) PREFIX="16_06_N10" ;;
    *)  PREFIX="17_06_N${N}" ;;
esac

SCRATCH="/itet-stor/arbaur/net_scratch/out"
HONEST_DIR="${SCRATCH}/models/${PREFIX}"
HONEST_PATTERN="${HONEST_DIR}/**"
BASELINE_DIR="${SCRATCH}/evaluations/${PREFIX}_nobyz_4000ep"
EXP_NAME="oracle_fullvec_1byz_N${N}"
OUTPUT_DIR="${SCRATCH}/models/learned_byzantine/${EXP_NAME}"
EVAL_LOG_DIR="${SCRATCH}/evaluations/${EXP_NAME}_2000ep"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"

N_BYZ=1
N_ENVS=29
TIMESTEPS=30000000
EP_LEN=2000
EPS=250          # per shard
N_SHARDS=8       # 8 x 250 = 2000 episodes

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -d "$HONEST_DIR" ]; then
    echo "ERROR: victim model dir not found for N=${N}: ${HONEST_DIR}"
    echo "       train it first: sbatch job_scripts/train_honest_no_obs_attack_ditet.sh ${N}"
    exit 1
fi
if [ ! -d "$BASELINE_DIR" ]; then
    echo "WARN: matched no-byz baseline not found: ${BASELINE_DIR}"
    echo "      (needed for Delta-tau) run: sbatch job_scripts/eval_nobyz_N_scaling_4000ep_ditet.sh"
fi

echo "=== Training ORACLE full-vector 1-byz attacker: ${EXP_NAME} ==="
echo "  Victim       : $HONEST_PATTERN"
echo "  N (total)    : $N   (honest=$((N - 1)), Byzantine=${N_BYZ})"
echo "  obs_scope    : full   (oracle: attacker sees the entire global state)"
echo "  attack_type  : obs_full   (full-vector falsification)"
echo "  Timesteps    : $TIMESTEPS"
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
# Evaluation — runs automatically once training completes.
# ===========================================================================
BYZ="${OUTPUT_DIR}/byz_agent.zip"
if [ ! -f "$BYZ" ]; then
    echo "ERROR: trained model not found at $BYZ — skipping eval."
    exit 1
fi
mkdir -p "$EVAL_LOG_DIR"

AGENT_NAME=$(find "$HONEST_DIR" -name "*.zip" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
[ -z "$AGENT_NAME" ] && AGENT_NAME="honest"

echo ""
echo "=== Evaluating ${EXP_NAME} (${N_SHARDS}x${EPS}ep) -> ${EVAL_LOG_DIR} ==="
echo "  Honest agent stem: $AGENT_NAME"

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
echo "Eval output: ${EVAL_LOG_DIR}/"
echo "Delta-tau baseline: ${PREFIX}_nobyz_4000ep"
exit 0
