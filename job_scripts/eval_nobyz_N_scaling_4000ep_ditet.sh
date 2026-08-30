#!/bin/bash
#SBATCH --job-name="nobyz_Nscaling_4000"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Produce the clean (no-Byzantine) baselines for the N-scaling sweep, so every
# matched-N BORA result has a tau0(N) to compute Delta-tau against. Eval only:
# it assumes the FROZEN honest victims are already trained on net_scratch
#   N=6,8,9 -> .../models/17_06_N<N>   (train_honest_no_obs_attack_ditet.sh)
#   N=10    -> .../models/16_06_N10    (train_N10_no_obs_attack_ditet.sh)
#
# Each baseline is run at obs_type=full, n_byz=0, 2000ep, matching the existing
# per-N baselines so the numbers are directly comparable to the N=5 (29_04) and
# N=7 (18_05_N7) baselines already on disk.
#
# Usage:  sbatch job_scripts/eval_nobyz_N_scaling_4000ep_ditet.sh
# Output (net_scratch): out/evaluations/{16_06_N10,17_06_N6,17_06_N8,17_06_N9}_nobyz_4000ep/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH="/itet-stor/arbaur/net_scratch/out"
EP_LEN=2000
N_EPS=2000
N_ENVS_EVAL=4

# N -> model-dir prefix (N=10 trained under 16_06, N=6/8/9 under 17_06)
N_LIST=(10 6 8 9)
prefix_for_N () {
    case "$1" in
        10) echo "16_06_N10" ;;
        *)  echo "17_06_N$1" ;;
    esac
}

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

for N in "${N_LIST[@]}"; do
    PREFIX=$(prefix_for_N "$N")
    MODEL_DIR="${SCRATCH}/models/${PREFIX}"
    BASELINE_DIR="${SCRATCH}/evaluations/${PREFIX}_nobyz_4000ep"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: victim model dir not found for N=${N}: ${MODEL_DIR}"
        echo "       train it first (train_N10_no_obs_attack_ditet.sh / train_honest_no_obs_attack_ditet.sh ${N})"
        exit 1
    fi

    echo ""
    echo "=== N=${N} no-Byzantine baseline (obs_type=full, ${N_EPS}ep) -> ${BASELINE_DIR} ==="
    mkdir -p "$BASELINE_DIR"
    # NOTE: the victim checkpoints are nested two levels deep
    #   ${MODEL_DIR}/<date>/PPO_..._no_obs_attack/rl_model_*.zip
    # and run_byzantine_sweep globs WITHOUT recursive=True (so '**' == '*').
    # A plain "${MODEL_DIR}/**" only reaches the <date> dir and finds no agent
    # file. Match the PPO run dir explicitly instead.
    python src/Utils/run_byzantine_sweep.py \
        --model        "${MODEL_DIR}/*/PPO_*" \
        --name         "${PREFIX}_nobyz_4000ep" \
        --results-base "${SCRATCH}/evaluations" \
        --obs-type     full \
        --n-byz        0 \
        --eps          "$N_EPS" \
        --ep-len       "$EP_LEN" \
        --n-envs       "$N_ENVS_EVAL" \
        --conditions   no_byzantine \
        --parallel \
        --no-plot
    echo "N=${N} baseline done at: $(date)"
done

echo ""
echo "Finished at: $(date)"
echo "Baselines in ${SCRATCH}/evaluations/: 16_06_N10_nobyz_4000ep, 17_06_N6_nobyz_4000ep, 17_06_N8_nobyz_4000ep, 17_06_N9_nobyz_4000ep"
exit 0
