#!/bin/bash
#SBATCH --job-name="adv_base_dirlie"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Produces the two adversarial-defense conditions that came back EMPTY from
# eval_adv_honest_full_obs_3cond (its run_byzantine_sweep part failed):
#
#   1) Defended no-Byzantine baseline  -> via train_learned_byzantine --max-offset 0
#      (the reliable path; the full_obs Byzantine nodes are present but truthful,
#       so this is the defended policy's matched tau0). Avoids the run_byzantine_sweep
#       obs_type=full failure entirely.
#   2) Defended under the heuristic direction lie -> via run_byzantine_sweep with
#      --obs-type full --n-envs 8 (vectorised path that handles obs_type=full).
#
# Together with the already-evaluated defended-under-BORA result
# (adv_honest_full_obs_eval/learned_obs_full, tau = 7.20), these complete the
# full-obs adversarial-defense comparison so Delta-tau can be computed on the
# defended policy's own baseline.
#
# Usage:  sbatch job_scripts/eval_adv_honest_baseline_dirlie_ditet.sh
# Output: out/evaluations/adv_honest_full_obs_nobyz/learned_obs_full/   (tau0_def)
#         out/evaluations/adv_honest_full_obs_dirlie/obs_user_dir/      (dir-lie)
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="${SCRATCH}/models/adv_honest_2byz_full_obs"           # defended policy (+ config)
MODEL_PATTERN="${MODEL_DIR}/**"
BYZ_FULL="${SCRATCH}/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
N_BYZ=2
MAGNITUDE=4.0
N_EPS=1000
EP_LEN=2000
N_ENVS=8

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -d "$MODEL_DIR" ]; then echo "ERROR: defended model not found at $MODEL_DIR"; exit 1; fi
if [ -z "$(find "$MODEL_DIR" -maxdepth 2 -name '*.yaml' -print -quit)" ]; then
    echo "ERROR: no .yaml config in $MODEL_DIR — push the victim config first (same as the 3cond job)."
    exit 1
fi

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Defended policy: $MODEL_DIR (agent $AGENT_NAME)"

# --- 1) defended no-Byzantine baseline (max_offset 0 -> truthful Byzantine) ---
echo "=== [1/2] defended no-Byzantine baseline (max_offset 0) ==="
python src/Utils/train_learned_byzantine.py --evaluate \
    --honest-model "$MODEL_PATTERN" \
    --byz-model    "$BYZ_FULL" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --max-offset   0.0 \
    --obs-scope    heard \
    --attack-type  obs_full \
    --eps          "$N_EPS" \
    --ep-len       "$EP_LEN" \
    --log-dir      "out/evaluations/adv_honest_full_obs_nobyz" \
    --agent-name   "$AGENT_NAME"

# --- 2) defended under heuristic direction lie (obs_type=full, vectorised) ---
echo "=== [2/2] defended under heuristic direction lie (mag=${MAGNITUDE}) ==="
python src/Utils/run_byzantine_sweep.py \
    --model      "$MODEL_PATTERN" \
    --name       "adv_honest_full_obs_dirlie" \
    --obs-type   full \
    --magnitude  "$MAGNITUDE" \
    --eps        "$N_EPS" \
    --ep-len     "$EP_LEN" \
    --n-byz      "$N_BYZ" \
    --n-envs     "$N_ENVS" \
    --conditions obs_user_dir \
    --no-plot

echo ""
echo "Finished at: $(date)"
echo "Defended: tau0 in adv_honest_full_obs_nobyz/, dir-lie in adv_honest_full_obs_dirlie/,"
echo "          BORA already in adv_honest_full_obs_eval/learned_obs_full/ (tau=7.20)."
