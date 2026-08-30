#!/bin/bash
#SBATCH --job-name="adv_honest_fo_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the adversarially-trained (vs full_obs) honest policy under three
# conditions, all written into one experiment dir for the bar_adv_defense plot:
#
#   no_byzantine     : f=0 baseline throughput of the defended policy
#   obs_user_dir     : heuristic direction lie (f=2, m=4)
#   learned_obs_full : the learned full_obs BORA attacker (f=2)
#
# Run AFTER train_adv_honest_vs_full_obs_30M_ditet.sh finishes.
#
# Usage:  sbatch job_scripts/eval_adv_honest_full_obs_3cond_ditet.sh
# Output: out/evaluations/adv_honest_full_obs_eval/{no_byzantine,obs_user_dir,learned_obs_full}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="${SCRATCH}/models/adv_honest_2byz_full_obs"          # defended policy (+ copied config)
BYZ_FULL="${SCRATCH}/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
EXP_NAME="adv_honest_full_obs_eval"
LOG_DIR="out/evaluations/${EXP_NAME}"
N_BYZ=2
MAGNITUDE=4.0
N_EPS=1000
EP_LEN=2000

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: defended model not found at $MODEL_DIR"
    echo "Run train_adv_honest_vs_full_obs_30M_ditet.sh first."
    exit 1
fi
if [ ! -f "$BYZ_FULL" ]; then
    echo "ERROR: full_obs Byzantine model not found at $BYZ_FULL"
    exit 1
fi

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Defended policy : $MODEL_DIR (agent $AGENT_NAME)"
echo ""

# --- 1) no_byzantine + heuristic direction lie (config-driven sweep) ---
echo "=== [1/2] no_byzantine + heuristic direction lie (m=${MAGNITUDE}) ==="
python src/Utils/run_byzantine_sweep.py \
    --model      "${MODEL_DIR}/**" \
    --name       "$EXP_NAME" \
    --n-byz      "$N_BYZ" \
    --magnitude  "$MAGNITUDE" \
    --conditions no_byzantine obs_user_dir \
    --eps        "$N_EPS" \
    --ep-len     "$EP_LEN" \
    --n-envs     8 \
    --parallel \
    --no-plot

# --- 2) learned full_obs BORA attacker (into the same experiment dir) ---
echo "=== [2/2] learned full_obs BORA attacker ==="
python src/Utils/train_learned_byzantine.py --evaluate \
    --honest-model "${MODEL_DIR}/**" \
    --byz-model    "$BYZ_FULL" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --max-offset   4.0 \
    --obs-scope    heard \
    --attack-type  obs_full \
    --eps          "$N_EPS" \
    --ep-len       "$EP_LEN" \
    --log-dir      "$LOG_DIR" \
    --agent-name   "$AGENT_NAME"

echo ""
echo "Finished at: $(date)"
echo "Conditions in ${LOG_DIR}/: no_byzantine, obs_user_dir, learned_obs_full"
exit 0
