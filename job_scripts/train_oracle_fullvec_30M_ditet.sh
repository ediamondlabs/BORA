#!/bin/bash
#SBATCH --job-name="oracle_fullvec"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train a FULL-VECTOR learned Byzantine attacker at the ORACLE observation
# scope (--obs-scope full: attacker sees the entire global state) and falsifies
# the whole per-node broadcast vector (--attack-type obs_full). Auto-evaluates
# afterwards (8 x 250 = 2000 episodes, distinct seeds per shard).
#
# Parametrized so one script covers all four requested cells (f in {1,2},
# N in {5,10}); N is inherited from the honest victim's config (no override).
#
# Usage (submit once per cell):
#   sbatch .../train_oracle_fullvec_30M_ditet.sh <exp_name> <honest_glob> <n_byz>
#
# The four cells:
#   # N=5  (victim out/models/29_04, tau0 = 16.78 = 29_04_nobyz_4000ep)
#   sbatch .../train_oracle_fullvec_30M_ditet.sh oracle_fullvec_1byz_N5  "out/models/29_04/**"        1
#   sbatch .../train_oracle_fullvec_30M_ditet.sh oracle_fullvec_2byz_N5  "out/models/29_04/**"        2
#   # N=10 (victim on net_scratch — MUST finish train_N10_no_obs_attack first;
#   #       baseline 16_06_N10_nobyz_4000ep produced by that same job)
#   sbatch .../train_oracle_fullvec_30M_ditet.sh oracle_fullvec_1byz_N10 "/itet-stor/arbaur/net_scratch/out/models/16_06_N10/**" 1
#   sbatch .../train_oracle_fullvec_30M_ditet.sh oracle_fullvec_2byz_N10 "/itet-stor/arbaur/net_scratch/out/models/16_06_N10/**" 2
#
# Output (net_scratch):
#   models/learned_byzantine/<exp_name>/byz_agent.zip
#   evaluations/<exp_name>_2000ep/   (Delta-tau vs the matched-N no-Byzantine baseline)
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP_NAME="$1"
HONEST_PATTERN="$2"
N_BYZ="$3"

if [ -z "$EXP_NAME" ] || [ -z "$HONEST_PATTERN" ] || [ -z "$N_BYZ" ]; then
    echo "Usage: sbatch $0 <exp_name> <honest_glob> <n_byz>"
    exit 1
fi

N_ENVS=29
TIMESTEPS=30000000
SCRATCH="/itet-stor/arbaur/net_scratch/out"
OUTPUT_DIR="${SCRATCH}/models/learned_byzantine/${EXP_NAME}"
EVAL_LOG_DIR="${SCRATCH}/evaluations/${EXP_NAME}_2000ep"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
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
echo "=== Training ORACLE-scope full-vector attacker: ${EXP_NAME} ==="
echo "  Honest model : $HONEST_PATTERN"
echo "  N_byz        : $N_BYZ"
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

HONEST_DIR="${HONEST_PATTERN%/\*\*}"
AGENT_NAME=$(find $HONEST_DIR -name "*.zip" -printf "%T@ %p\n" 2>/dev/null \
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
echo "Delta-tau baseline: N=5 -> 29_04_nobyz_4000ep (16.78); N=10 -> 16_06_N10_nobyz_4000ep"
exit 0
