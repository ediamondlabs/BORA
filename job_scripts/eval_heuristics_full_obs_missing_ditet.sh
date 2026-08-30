#!/bin/bash
#SBATCH --job-name="heur_full_missing"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=48
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Full migration of the heuristic attacks to obs_type=full (16.78 baseline).
# Runs ONLY the missing conditions, all in PARALLEL, 2000 episodes each:
#   - harm-curve mids : mag 1.0 and mag 2.0  (full heuristic set, f=2, N=5)
#   - scaling points  : direction lie at f=1 (N=5) and at N=7
# Already-computed points are reused: mag0.5 and mag4 exist, so they are NOT
# re-run. obs_type=full requires the vectorised path, hence --n-envs > 1.
#
# Usage:  sbatch job_scripts/eval_heuristics_full_obs_missing_ditet.sh
# Output (net_scratch): 29_04_heuristics_full_obs_{mag1.0, mag2.0, f1_mag4, N7_mag4}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH="/itet-stor/arbaur/net_scratch/out"
RESULTS_BASE="${SCRATCH}/evaluations"
N5_MODEL="out/models/29_04/**"
N7_MODEL="${SCRATCH}/models/18_05_N7/**"
HEUR="no_byzantine obs_noise obs_user_dir obs_capacity obs_demand obs_interference_lie obs_replay"
N_EPS=2000
EP_LEN=2000
N_ENVS=3

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "Missing heuristic-at-full-obs conditions, parallel, ${N_EPS}ep"

sweep () {   # $1=exp_name  $2=model  $3=n_byz  $4=magnitude  $5=conditions
    python src/Utils/run_byzantine_sweep.py \
        --model        "$2" \
        --name         "$1" \
        --results-base "$RESULTS_BASE" \
        --obs-type     full \
        --magnitude    "$4" \
        --n-byz        "$3" \
        --eps          "$N_EPS" \
        --ep-len       "$EP_LEN" \
        --n-envs       "$N_ENVS" \
        --conditions   $5 \
        --parallel \
        --no-plot \
        > "logs/${1}_${SLURM_JOB_ID}.log" 2>&1
}

PIDS=()
# harm-curve mid-points (full heuristic set, f=2, N=5)
sweep 29_04_heuristics_full_obs_mag1.0  "$N5_MODEL" 2 1.0 "$HEUR" &  PIDS+=($!)
sweep 29_04_heuristics_full_obs_mag2.0  "$N5_MODEL" 2 2.0 "$HEUR" &  PIDS+=($!)
# scaling points (direction lie only) at f=1 (N=5) and N=7
sweep 29_04_heuristics_full_obs_f1_mag4 "$N5_MODEL" 1 4.0 "no_byzantine obs_user_dir" &  PIDS+=($!)
sweep 29_04_heuristics_full_obs_N7_mag4 "$N7_MODEL" 2 4.0 "no_byzantine obs_user_dir" &  PIDS+=($!)

echo "Launched ${#PIDS[@]} parallel sweeps; waiting …"
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || { echo "  sweep PID $p FAILED"; FAILED=1; }; done

echo ""
echo "Finished at: $(date)  (failures: ${FAILED})"
echo "Outputs: ${RESULTS_BASE}/29_04_heuristics_full_obs_{mag1.0, mag2.0, f1_mag4, N7_mag4}/"
exit 0
