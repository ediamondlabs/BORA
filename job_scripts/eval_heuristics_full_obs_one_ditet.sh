#!/bin/bash
#SBATCH --job-name="heur_full_one"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Run ONE heuristic-at-full-obs sweep (split out so each fits the per-job
# resource cap). 2000 episodes, obs_type=full, jammers on, output -> net_scratch.
#
# Usage (submit once per missing condition):
#   sbatch eval_heuristics_full_obs_one_ditet.sh <exp_name> <model_glob> <n_byz> <mag> "<conditions>"
#
# The four missing conditions:
#   sbatch .../eval_heuristics_full_obs_one_ditet.sh 29_04_heuristics_full_obs_mag1.0  "out/models/29_04/**" 2 1.0 "no_byzantine obs_noise obs_user_dir obs_capacity obs_demand obs_interference_lie obs_replay"
#   sbatch .../eval_heuristics_full_obs_one_ditet.sh 29_04_heuristics_full_obs_mag2.0  "out/models/29_04/**" 2 2.0 "no_byzantine obs_noise obs_user_dir obs_capacity obs_demand obs_interference_lie obs_replay"
#   sbatch .../eval_heuristics_full_obs_one_ditet.sh 29_04_heuristics_full_obs_f1_mag4 "out/models/29_04/**" 1 4.0 "no_byzantine obs_user_dir"
#   sbatch .../eval_heuristics_full_obs_one_ditet.sh 29_04_heuristics_full_obs_N7_mag4 "/itet-stor/arbaur/net_scratch/out/models/18_05_N7/**" 2 4.0 "no_byzantine obs_user_dir"
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP_NAME="$1"
MODEL="$2"
N_BYZ="$3"
MAG="$4"
CONDS="$5"
RESULTS_BASE="/itet-stor/arbaur/net_scratch/out/evaluations"
N_EPS=2000
EP_LEN=2000
N_ENVS=4

if [ -z "$EXP_NAME" ] || [ -z "$MODEL" ] || [ -z "$N_BYZ" ] || [ -z "$MAG" ] || [ -z "$CONDS" ]; then
    echo "Usage: sbatch $0 <exp_name> <model_glob> <n_byz> <mag> \"<conditions>\""
    exit 1
fi

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"
echo "Sweep: ${EXP_NAME} | model=${MODEL} | f=${N_BYZ} | mag=${MAG} | obs_type=full | ${N_EPS}ep"
echo "Conditions: ${CONDS}"

python src/Utils/run_byzantine_sweep.py \
    --model        "$MODEL" \
    --name         "$EXP_NAME" \
    --results-base "$RESULTS_BASE" \
    --obs-type     full \
    --magnitude    "$MAG" \
    --n-byz        "$N_BYZ" \
    --eps          "$N_EPS" \
    --ep-len       "$EP_LEN" \
    --n-envs       "$N_ENVS" \
    --conditions   $CONDS \
    --parallel \
    --no-plot

echo ""
echo "Finished at: $(date)"
echo "Output: ${RESULTS_BASE}/${EXP_NAME}/"
exit 0
