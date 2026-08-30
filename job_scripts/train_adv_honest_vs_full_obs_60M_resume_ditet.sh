#!/bin/bash
#SBATCH --job-name="adv_honest_60M_resume"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Resume the adversarial honest training (vs frozen obs_full BORA) from the
# latest 30M checkpoint and continue to 60M TOTAL steps. The 30M-from-scratch
# run produced a weak policy (clean tau ~3.3); the extra budget tests whether
# more training lets it actually learn to relay under the full-vector attacker.
#
# Resumes from the newest checkpoints/honest_adv_<steps>_steps.zip and keeps
# writing into the SAME model dir, so honest_adv.zip is overwritten at the end.
#
# Usage:  sbatch job_scripts/train_adv_honest_vs_full_obs_60M_resume_ditet.sh
# Output: /itet-stor/arbaur/net_scratch/out/models/adv_honest_2byz_full_obs/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

HONEST_PATTERN="out/models/29_04/**"
BYZ_MODEL="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/adv_honest_2byz_full_obs"
N_BYZ=2
N_ENVS=29
TIMESTEPS=60000000      # TOTAL target (script computes remaining = TIMESTEPS - checkpoint steps)

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -f "$BYZ_MODEL" ]; then
    echo "ERROR: frozen Byzantine model not found at $BYZ_MODEL"; exit 1
fi

# Pick the checkpoint with the highest step count (honest_adv_<steps>_steps.zip).
RESUME=$(ls "${OUTPUT_DIR}"/checkpoints/honest_adv_*_steps.zip 2>/dev/null \
         | sort -t_ -k3 -n | tail -1)
if [ -z "$RESUME" ]; then
    echo "ERROR: no checkpoint found in ${OUTPUT_DIR}/checkpoints/ to resume from"; exit 1
fi
echo "=== Resuming adversarial honest training to ${TIMESTEPS} total steps ==="
echo "  Resume checkpoint : $RESUME"
echo "  Frozen attacker   : $BYZ_MODEL"
echo "  Output            : $OUTPUT_DIR"
echo ""

python src/Utils/train_adv_honest_vs_full_obs.py \
    --honest-model "$HONEST_PATTERN" \
    --byz-model    "$BYZ_MODEL" \
    --eval-config  "$EVAL_CONFIG" \
    --n-byz        "$N_BYZ" \
    --max-offset   4.0 \
    --obs-type     full \
    --obs-scope    heard \
    --attack-type  obs_full \
    --ep-len       2000 \
    --timesteps    "$TIMESTEPS" \
    --n-envs       "$N_ENVS" \
    --seed         9 \
    --output       "$OUTPUT_DIR" \
    --resume-from  "$RESUME"

echo ""
echo "Finished at: $(date)"
echo "Resumed policy saved to: ${OUTPUT_DIR}/honest_adv.zip"
exit 0
