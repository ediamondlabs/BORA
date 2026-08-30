#!/bin/bash
#SBATCH --job-name="train_N10"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train an undefended PADRE honest victim policy with N=10 relay nodes
# (config: src/ubelix_train/16_06_N10_no_obs_attack.yaml; identical to the
# N=5 / N=7 victims except numbOfNodes=10). This victim is the FROZEN target
# the N=10 oracle attacker jobs train against, so it must finish first.
#
# After training it also evaluates the clean (no-Byzantine) baseline at
# obs_type=full so the N=10 oracle results have a matched tau0(N=10) to
# compute Delta-tau against (the N=5 jobs reuse 29_04_nobyz = 16.78).
#
# Usage:   sbatch job_scripts/train_N10_no_obs_attack_ditet.sh
# Outputs (net_scratch):
#          .../net_scratch/out/models/16_06_N10/**                 (victim policy)
#          .../net_scratch/out/evaluations/16_06_N10_nobyz_4000ep/ (baseline)
# Next:    the two N=10 oracle jobs (see train_oracle_fullvec_30M_ditet.sh),
#          pointing --honest-model at .../net_scratch/out/models/16_06_N10/**
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG="src/ubelix_train/16_06_N10_no_obs_attack.yaml"
N_ENVS=29
SCRATCH="/itet-stor/arbaur/net_scratch/out"
BASELINE_DIR="${SCRATCH}/evaluations/16_06_N10_nobyz_4000ep"
EP_LEN=2000
N_EPS=2000
N_ENVS_EVAL=4

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "=== Training N=10 honest victim (30M steps) ==="
echo "  Config : $CONFIG"
echo "  n_envs : $N_ENVS"
echo ""

python src/train.py "$CONFIG" --n_envs "$N_ENVS"

echo ""
echo "Victim training finished at: $(date)"
echo "Model saved under: ${SCRATCH}/models/16_06_N10/"

# --- clean (no-Byzantine) baseline at obs_type=full, for tau0(N=10) ---
echo ""
echo "=== Evaluating N=10 no-Byzantine baseline (obs_type=full, ${N_EPS}ep) ==="
mkdir -p "$BASELINE_DIR"
python src/Utils/run_byzantine_sweep.py \
    --model        "${SCRATCH}/models/16_06_N10/**" \
    --name         "16_06_N10_nobyz_4000ep" \
    --results-base "${SCRATCH}/evaluations" \
    --obs-type     full \
    --n-byz        0 \
    --eps          "$N_EPS" \
    --ep-len       "$EP_LEN" \
    --n-envs       "$N_ENVS_EVAL" \
    --conditions   no_byzantine \
    --parallel \
    --no-plot

echo ""
echo "Finished at: $(date)"
echo "Victim:   ${SCRATCH}/models/16_06_N10/   |   Baseline: ${BASELINE_DIR}/"
exit 0
