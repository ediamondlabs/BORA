#!/bin/bash
#SBATCH --job-name="train_honest"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=30
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Train an undefended PADRE honest victim policy with a parametrized node count
# N (config: src/ubelix_train/17_06_N${N}_no_obs_attack.yaml; identical to the
# N=5 / N=7 / N=10 victims except numbOfNodes). These victims are the FROZEN
# targets the BORA-float (oracle-scope full-vector) attacker trains against, so
# they must finish first. Intended for the N-scaling sweep (N=6, 8, 9), filling
# the gaps between the existing N=5 (29_04), N=7 (18_05_N7), N=10 (16_06_N10).
#
# After training it also evaluates the clean (no-Byzantine) baseline at
# obs_type=full so the matched-N BORA results have a tau0(N) to compute
# Delta-tau against.
#
# Usage:   sbatch job_scripts/train_honest_no_obs_attack_ditet.sh <N>
#   e.g.   sbatch job_scripts/train_honest_no_obs_attack_ditet.sh 6
#          sbatch job_scripts/train_honest_no_obs_attack_ditet.sh 8
#          sbatch job_scripts/train_honest_no_obs_attack_ditet.sh 9
# Outputs (net_scratch):
#          .../net_scratch/out/models/17_06_N<N>/**                 (victim policy)
#          .../net_scratch/out/evaluations/17_06_N<N>_nobyz_4000ep/ (baseline)
# Next:    BORA float (train_oracle_fullvec_30M_ditet.sh) pointing
#          --honest-model at .../net_scratch/out/models/17_06_N<N>/**
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
    echo "Usage: sbatch $0 <N>   (number of relay nodes, e.g. 6, 8, 9)"
    exit 1
fi

CONFIG="src/ubelix_train/17_06_N${N}_no_obs_attack.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

N_ENVS=29
SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="${SCRATCH}/models/17_06_N${N}"
BASELINE_DIR="${SCRATCH}/evaluations/17_06_N${N}_nobyz_4000ep"
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
echo "=== Training N=${N} honest victim (30M steps) ==="
echo "  Config : $CONFIG"
echo "  n_envs : $N_ENVS"
echo ""

python src/train.py "$CONFIG" --n_envs "$N_ENVS"

echo ""
echo "Victim training finished at: $(date)"
echo "Model saved under: ${MODEL_DIR}/"

# --- clean (no-Byzantine) baseline at obs_type=full, for tau0(N) ---
echo ""
echo "=== Evaluating N=${N} no-Byzantine baseline (obs_type=full, ${N_EPS}ep) ==="
mkdir -p "$BASELINE_DIR"
python src/Utils/run_byzantine_sweep.py \
    --model        "${MODEL_DIR}/**" \
    --name         "17_06_N${N}_nobyz_4000ep" \
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
echo "Victim:   ${MODEL_DIR}/   |   Baseline: ${BASELINE_DIR}/"
exit 0
