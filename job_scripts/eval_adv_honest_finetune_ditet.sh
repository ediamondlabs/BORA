#!/bin/bash
#SBATCH --job-name="adv_finetune_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate a defended honest policy on its no-Byzantine baseline and under the
# full-vector BORA, BOTH through train_learned_byzantine. This is the reliable
# route: the run_byzantine_sweep no_byzantine condition kept coming back EMPTY
# with obs_type=full, whereas train_learned_byzantine --max-offset 0 works
# (it gave the 16.78 main baseline and the defended 3.3 baseline).
#
# Defaults to the FINE-TUNED model; repoint MODEL_DIR to re-eval the 60M-resumed
# or from-scratch model.
#
# Usage:  sbatch job_scripts/eval_adv_honest_finetune_ditet.sh
# Output: /itet-stor/arbaur/net_scratch/out/evaluations/adv_honest_finetune_nobyz/  (tau0_def, max_offset 0)
#         /itet-stor/arbaur/net_scratch/out/evaluations/adv_honest_finetune_bora/   (defended under BORA)
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="${SCRATCH}/models/adv_honest_2byz_full_obs_finetune"   # defended (fine-tuned) policy
MODEL_PATTERN="${MODEL_DIR}/**"
BYZ_FULL="${SCRATCH}/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
N_BYZ=2
N_EPS=1000
EP_LEN=2000

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
if [ ! -d "$MODEL_DIR" ]; then echo "ERROR: model not found at $MODEL_DIR"; exit 1; fi
if [ -z "$(find "$MODEL_DIR" -maxdepth 2 -name '*.yaml' -print -quit)" ]; then
    echo "ERROR: no .yaml config in $MODEL_DIR (the trainer should have copied it)."; exit 1
fi

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Defended policy: $MODEL_DIR (agent $AGENT_NAME)"

run () {   # $1 = experiment name, $2 = max offset
    echo "=== ${1} (max_offset=$2) ${N_EPS}ep ==="
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model "$MODEL_PATTERN" \
        --byz-model    "$BYZ_FULL" \
        --eval-config  "$EVAL_CONFIG" \
        --n-byz        "$N_BYZ" \
        --max-offset   "$2" \
        --obs-scope    heard \
        --attack-type  obs_full \
        --eps          "$N_EPS" \
        --ep-len       "$EP_LEN" \
        --log-dir      "${SCRATCH}/evaluations/$1" \
        --agent-name   "$AGENT_NAME"
}

run adv_honest_finetune_nobyz  0.0
run adv_honest_finetune_bora   4.0

echo ""
echo "Finished at: $(date)"
echo "Outputs in ${SCRATCH}/evaluations/{adv_honest_finetune_nobyz, adv_honest_finetune_bora}/"
exit 0
