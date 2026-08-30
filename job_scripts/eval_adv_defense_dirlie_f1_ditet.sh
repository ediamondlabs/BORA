#!/bin/bash
#SBATCH --job-name="adv_def_extra"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Fills the two MISSING conditions for the adversarial-defense bar chart
# (fig:bar_adv_defense), so it is no longer "a bit empty" (supervisor L781):
#
#   #1  Heuristic DIRECTION LIE (obs_user_dir, f=2, mag=4) on BOTH policies
#       -> shows the defense generalises to an attack it never saw in training
#       (it was fine-tuned only against BORA).
#   #2  Single-Byzantine BORA (f=1) on BOTH policies
#       -> shows the defense across threat intensity (f=1 vs the existing f=2).
#
# Both are EVAL-ONLY (no training). Run on arton12, then rsync the CSVs back to
# out/evaluations/adv_defense_extra/ and the figure/prose get wired locally.
#
# Already known (do NOT need re-running, here for reference / Delta-tau baselines):
#   undefended NB  = 16.78   defended NB  = 16.35
#   undefended BORA f=1 = 0.43 (line_scaling N=5)   undefended BORA f=2 = 0.14
#   defended  BORA f=2 = 8.18
#
# Usage:  sbatch job_scripts/eval_adv_defense_dirlie_f1_ditet.sh
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

SCRATCH="/itet-stor/arbaur/net_scratch/out"
UNDEF_DIR="${SCRATCH}/models/29_04"                              # undefended (standard) N=5 honest policy
DEF_DIR="${SCRATCH}/models/adv_honest_2byz_full_obs_finetune"   # adversarially fine-tuned policy
BYZ_F1="${SCRATCH}/models/learned_byzantine/29_04_1byz_full_obs_30M/byz_agent.zip"  # f=1 full-obs BORA attacker
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
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
for d in "$UNDEF_DIR" "$DEF_DIR"; do
    if [ ! -d "$d" ]; then echo "ERROR: model not found at $d"; exit 1; fi
    if [ -z "$(find "$d" -maxdepth 2 -name '*.yaml' -print -quit)" ]; then
        echo "ERROR: no .yaml config in $d — push the victim config first."; exit 1
    fi
done
if [ ! -f "$BYZ_F1" ]; then echo "ERROR: f=1 BORA attacker not found at $BYZ_F1"; exit 1; fi

agent_of () { find "$1" -name "*.zip" -printf "%T@ %p\n" | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//'; }

# --- #1 heuristic direction lie (f=2), vectorised obs_type=full path ---
# Adv-trained policies are saved as a step-less 'honest_adv.zip' (named-agent
# layout); run_byzantine_sweep's agent discovery needs a step number in the
# filename and globs '**' non-recursively, so point --model at the explicit
# .zip. Standard PPO_* victims (29_04) keep the '/**' dir glob.
dirlie () {   # $1 = model dir, $2 = output name
    echo "=== [#1 dir-lie] $2 (f=2, mag=${MAGNITUDE}) ==="
    local model_pattern="${1}/**"
    [ -f "${1}/honest_adv.zip" ] && model_pattern="${1}/honest_adv.zip"
    python src/Utils/run_byzantine_sweep.py \
        --model      "$model_pattern" \
        --name       "$2" \
        --obs-type   full \
        --magnitude  "$MAGNITUDE" \
        --eps        "$N_EPS" \
        --ep-len     "$EP_LEN" \
        --n-byz      2 \
        --n-envs     "$N_ENVS" \
        --conditions obs_user_dir \
        --no-plot
}

# --- #2 single-Byzantine BORA (f=1), reliable train_learned_byzantine path ---
bora_f1 () {   # $1 = model dir, $2 = output name
    echo "=== [#2 BORA f=1] $2 ==="
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model "${1}/**" \
        --byz-model    "$BYZ_F1" \
        --eval-config  "$EVAL_CONFIG" \
        --n-byz        1 \
        --max-offset   "$MAGNITUDE" \
        --obs-scope    heard \
        --attack-type  obs_full \
        --eps          "$N_EPS" \
        --ep-len       "$EP_LEN" \
        --log-dir      "out/evaluations/adv_defense_extra/$2" \
        --agent-name   "$(agent_of "$1")"
}

dirlie  "$UNDEF_DIR" "adv_defense_extra/undef_dirlie"
dirlie  "$DEF_DIR"   "adv_defense_extra/def_dirlie"
bora_f1 "$UNDEF_DIR" "undef_bora_f1"
bora_f1 "$DEF_DIR"   "def_bora_f1"

echo ""
echo "Finished at: $(date)"
echo "Outputs under out/evaluations/adv_defense_extra/{undef_dirlie,def_dirlie,undef_bora_f1,def_bora_f1}/"
exit 0
