#!/bin/bash
#SBATCH --job-name="babel_1byz_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the BABEL 1-byz learned attacker (local obs scope) for 4000 episodes
# on the D-ITET cluster. Output goes to /itet-stor/arbaur/net_scratch/.
#
# Usage:   sbatch job_scripts/eval_learned_byzantine_1byz_local_4000ep_ditet.sh
# Requires: ~/resilient-backbone/out/models/learned_byzantine/29_04_1byz_local/byz_agent.zip
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

MODEL_DIR="out/models/29_04"
N_BYZ=1
BYZ_MODEL="out/models/learned_byzantine/29_04_${N_BYZ}byz_local/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
BASELINE_SOURCE="/itet-stor/arbaur/net_scratch/out/evaluations/29_04_jammer_2byz_v3_mag4.0"

N_EPS=4000
EP_LEN=2000
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
    echo "Run train_learned_byzantine_1byz_local.sh first."
    exit 1
fi

if [ ! -d "${BASELINE_SOURCE}/no_byzantine" ]; then
    echo "ERROR: Baseline not found at ${BASELINE_SOURCE}/no_byzantine"
    exit 1
fi

HONEST_MODEL=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}')
if [ -z "$HONEST_MODEL" ]; then
    echo "ERROR: No model checkpoint found under $MODEL_DIR"
    exit 1
fi
AGENT_NAME=$(basename "$HONEST_MODEL" .zip)

echo "=== Evaluating BABEL 1-byz learned attacker (4000 episodes) ==="
echo "  Byzantine model : $BYZ_MODEL"
echo "  Honest model    : $HONEST_MODEL"
echo "  Episodes        : $N_EPS x $EP_LEN steps"
echo "  obs_scope       : local"
echo "  Output          : $LOG_DIR"
echo ""

python src/Utils/train_learned_byzantine.py --evaluate \
    --honest-model  "${MODEL_DIR}/**" \
    --byz-model     "$BYZ_MODEL" \
    --eval-config   "$EVAL_CONFIG" \
    --n-byz         "$N_BYZ" \
    --max-offset    4.0 \
    --obs-scope     local \
    --eps           "$N_EPS" \
    --ep-len        "$EP_LEN" \
    --log-dir       "$LOG_DIR" \
    --agent-name    "$AGENT_NAME"

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
