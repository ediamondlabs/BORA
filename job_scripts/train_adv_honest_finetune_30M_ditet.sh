#!/bin/bash
#SBATCH --job-name="adv_honest_finetune"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Adversarial-training defense, FINE-TUNING variant: warm-start the trainable
# honest policy from the undefended 16.78 policy (out/models/29_04) instead of
# from scratch, then adapt it for 30M steps with f=2 Byzantine nodes driven by
# the frozen full_obs BORA attacker. Starting from a competent relaying policy
# gives a usable reward signal, which from-scratch training lacked (clean tau
# collapsed to ~3.3).
#
# Output goes to a SEPARATE dir so the from-scratch model is preserved.
#
# Usage:  sbatch job_scripts/train_adv_honest_finetune_30M_ditet.sh
# Output: /itet-stor/arbaur/net_scratch/out/models/adv_honest_2byz_full_obs_finetune/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

HONEST_PATTERN="out/models/29_04/**"
INIT_FROM="out/models/29_04/**"      # warm-start the honest policy from the undefended model
BYZ_MODEL="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine/29_04_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
OUTPUT_DIR="/itet-stor/arbaur/net_scratch/out/models/adv_honest_2byz_full_obs_finetune"
N_BYZ=2
N_ENVS=29
TIMESTEPS=30000000

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
if [ ! -f "$BYZ_MODEL" ]; then echo "ERROR: frozen Byzantine model not found at $BYZ_MODEL"; exit 1; fi

echo "=== Adversarial honest FINE-TUNING vs frozen obs_full BORA (f=${N_BYZ}, ${TIMESTEPS} steps) ==="
echo "  Warm-start from : $INIT_FROM"
echo "  Frozen attacker : $BYZ_MODEL"
echo "  Output          : $OUTPUT_DIR"
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
    --init-from    "$INIT_FROM"

echo ""
echo "Finished at: $(date)"
echo "Fine-tuned honest policy saved to: ${OUTPUT_DIR}/honest_adv.zip"
exit 0
