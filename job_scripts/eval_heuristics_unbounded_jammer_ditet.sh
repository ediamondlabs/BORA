#!/bin/bash
#SBATCH --job-name="heur_unbounded"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# UNBOUNDED-magnitude heuristic sweep — answers the supervisor question
# "what happens if the attack magnitude is unbounded?" (Fig. harm_jammer_2byz).
# Extends the existing harm curve (29_04_jammer_2byz_v3, mags 0-4) to large
# magnitudes (8, 16, 32, 64) to show whether degradation saturates or keeps
# growing. Same config as the v3 curve: victim out/models/29_04, f=2, jammers
# on, obs_type=real (the realistic baseline, tau0=12.79), 2-Byzantine.
#
# IMPORTANT: --n-envs 1 (per-node eval path). obs_type=real with --n-envs > 1
# silently falls back to obs_type=full in the vectorised factory, which would
# change the baseline; n_envs=1 keeps it on the real-observation curve.
#
# Uses BASE_NAME=29_04_jammer_2byz_v3 so the new points slot straight into the
# existing curve; the final compare_magnitude_sweep spans mags 0 -> 64.
#
# Usage:  sbatch job_scripts/eval_heuristics_unbounded_jammer_ditet.sh
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"
MODEL_PATTERN="${MODEL_DIR}/**"
RESULTS_BASE="/itet-stor/arbaur/net_scratch/out/evaluations"   # write directly to net_scratch
BASE_NAME="29_04_jammer_2byz_v3"
NEW_MAGS=(8.0 16.0 32.0 64.0)                          # the unbounded extension
ALL_MAGS=(0.0 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0)      # full curve for the plot
N_EPS=1000
EP_LEN=2000
N_BYZ=2
N_ENVS=1

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "Unbounded heuristic sweep (obs_type=real, jammers on, f=${N_BYZ}) mags: ${NEW_MAGS[*]}"

# All unbounded magnitudes run CONCURRENTLY (each run_byzantine_sweep --parallel
# itself fans the 7 conditions out, so this is 4 mags x 7 conditions = 28 workers).
PIDS=()
for MAG in "${NEW_MAGS[@]}"; do
    EXP_NAME="${BASE_NAME}_mag${MAG}"
    echo "=== launching magnitude ${MAG} -> ${RESULTS_BASE}/${EXP_NAME} ==="
    python src/Utils/run_byzantine_sweep.py \
        --model        "$MODEL_PATTERN" \
        --name         "$EXP_NAME" \
        --results-base "$RESULTS_BASE" \
        --magnitude    "$MAG" \
        --eps          "$N_EPS" \
        --ep-len       "$EP_LEN" \
        --n-byz        "$N_BYZ" \
        --n-envs       "$N_ENVS" \
        --conditions   no_byzantine obs_noise obs_user_dir obs_capacity obs_demand obs_interference_lie obs_interference_amplify \
        --parallel \
        --no-plot \
        > "logs/unbounded_mag${MAG}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} magnitude runs; waiting …"
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || { echo "  magnitude PID $p FAILED"; FAILED=1; }; done

echo ""
echo "=== Building the full harm curve (mags 0 -> 64) ==="
EXP_NAMES=()
for MAG in "${ALL_MAGS[@]}"; do EXP_NAMES+=("${BASE_NAME}_mag${MAG}"); done
python src/Utils/visualize/byzantine/compare_magnitude_sweep.py \
    --base       "$RESULTS_BASE" \
    --names      "${EXP_NAMES[@]}" \
    --magnitudes "${ALL_MAGS[@]}" \
    --output     "out/visualizations/magnitude_sweep/${BASE_NAME}_unbounded" || \
    echo "NOTE: the full-curve plot needs the mag 0-4 dirs in ${RESULTS_BASE} too; copy them there and re-run compare_magnitude_sweep."

echo ""
echo "Finished at: $(date)  (failures: ${FAILED})"
echo "Unbounded points: ${RESULTS_BASE}/${BASE_NAME}_mag{8.0,16.0,32.0,64.0}/"
exit 0
