#!/bin/bash
#SBATCH --job-name="heur_full_obs_sweep"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=48
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Magnitude sweep of the heuristic attacks at obs_type=full, for the LOWER
# magnitudes only (0.5, 1.0, 2.0) -- mag=4 is already produced by
# eval_heuristics_full_obs_4000ep_ditet.sh. Together they give the full
# heuristic harm curve under full-state observation, comparable to the learned
# family and computable against the shared baseline (29_04_nobyz_4000ep = 16.78).
#
# Same setup as the mag=4 job: victim out/models/29_04, jammers on, f=2,
# obs_type=full (so --n-envs 8 forces the vectorised path), no no_byzantine.
# Magnitudes run sequentially; within each, the six conditions run concurrently
# (6 x 8 = 48 workers).
#
# Usage:  sbatch job_scripts/eval_heuristics_full_obs_sweep_ditet.sh
# Output: out/evaluations/29_04_heuristics_full_obs_mag{0.5,1.0,2.0}/{attack}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

MODEL_DIR="out/models/29_04"
MODEL_PATTERN="${MODEL_DIR}/**"
MAGNITUDES=(0.5 1.0 2.0)        # mag 4.0 already done by the mag4 job
N_EPS=2000                       # lighter than the mag=4 comparison point (4000)
EP_LEN=2000
N_BYZ=2
N_ENVS=8

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "Model: $MODEL_PATTERN | obs_type=full | jammers on | f=${N_BYZ} | mags=${MAGNITUDES[*]} | ${N_EPS}ep"

for MAG in "${MAGNITUDES[@]}"; do
    EXP_NAME="29_04_heuristics_full_obs_mag${MAG}"
    echo ""
    echo "=== magnitude ${MAG} -> out/evaluations/${EXP_NAME} ==="
    python src/Utils/run_byzantine_sweep.py \
        --model        "$MODEL_PATTERN" \
        --name         "$EXP_NAME" \
        --results-base "/itet-stor/arbaur/net_scratch/out/evaluations" \
        --obs-type   full \
        --magnitude  "$MAG" \
        --eps        "$N_EPS" \
        --ep-len     "$EP_LEN" \
        --n-byz      "$N_BYZ" \
        --n-envs     "$N_ENVS" \
        --conditions obs_user_dir obs_capacity obs_demand obs_interference_lie obs_noise obs_replay \
        --parallel \
        --no-plot
done

echo ""
echo "Finished at: $(date)"
echo "Harm curve: combine 29_04_heuristics_full_obs_mag{0.5,1.0,2.0} with mag4 (4000ep)."
