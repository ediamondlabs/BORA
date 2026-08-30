#!/bin/bash
#SBATCH --job-name="harm_nojam_2byz"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Full heuristic harm-curve sweep without jammer (--no-jammers).
# Isolates Byzantine behaviour from physical jamming confounders.
# Mirrors harm_jammer_2byz_sweep.sh but with numbOfJammers=0.
#
# Output:  out/visualizations/magnitude_sweep/29_04_nojammer_2byz/
#          → copy harm_curve_*.tex/.csv to 0_Thesis/figures/tikz/ as needed
#
# Usage:   sbatch job_scripts/harm_nojammer_2byz_sweep_ditet.sh
# ---------------------------------------------------------------------------

MODEL_DIR="out/models/29_04"
MAGNITUDES=(0.0 0.5 1.0 2.0 4.0)
N_EPS=1000
EP_LEN=2000
N_BYZ=2
BASE_NAME="29_04_nojammer_${N_BYZ}byz"

MODEL_PATTERN="${MODEL_DIR}/**"

# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Running on node: $(hostname)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo ""
echo "=== Heuristic harm-curve sweep — no jammer, ${N_BYZ} Byzantine nodes ==="
echo "  Model    : $MODEL_PATTERN"
echo "  Episodes : $N_EPS  |  ep_len: $EP_LEN"
echo "  Mags     : ${MAGNITUDES[*]}"
echo ""

EXP_NAMES=()
PIDS=()

for MAG in "${MAGNITUDES[@]}"; do
    EXP_NAME="${BASE_NAME}_mag${MAG}"
    EXP_NAMES+=("$EXP_NAME")

    echo "=== Starting magnitude ${MAG} ==="
    python src/Utils/run_byzantine_sweep.py \
        --model      "$MODEL_PATTERN" \
        --name       "$EXP_NAME" \
        --magnitude  "$MAG" \
        --eps        "$N_EPS" \
        --ep-len     "$EP_LEN" \
        --n-byz      "$N_BYZ" \
        --conditions no_byzantine obs_noise obs_user_dir obs_capacity obs_demand obs_interference_lie obs_replay \
        --no-jammers \
        --parallel \
        --no-plot &
    PIDS+=($!)
done

echo "Waiting for all magnitude runs to finish..."
for PID in "${PIDS[@]}"; do
    wait "$PID" || echo "WARNING: process $PID failed"
done

echo ""
echo "=== All runs complete. Generating plots ==="

python src/Utils/visualize/byzantine/compare_magnitude_sweep.py \
    --base       out/evaluations \
    --names      "${EXP_NAMES[@]}" \
    --magnitudes "${MAGNITUDES[@]}" \
    --output     "out/visualizations/magnitude_sweep/${BASE_NAME}"

echo ""
echo "Finished at: $(date)"
echo "Plots in: out/visualizations/magnitude_sweep/${BASE_NAME}/"
