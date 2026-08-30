#!/bin/bash
#SBATCH --job-name="heur_full_obs_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=48
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Re-evaluate ALL heuristic observation attacks with obs_type=full, so they are
# directly comparable to the learned attacks (same victim out/models/29_04, same
# full-state observation, jammers on, f=2, mag=4, 4000 episodes). Their Delta-tau
# can then be computed against the SAME matched no-Byzantine baseline as the
# learned family (29_04_nobyz_4000ep, tau0 = 16.78) -- so no no_byzantine
# condition is run here.
#
# obs_type=full requires the VECTORISED eval path, hence --n-envs 8 (the
# sequential per-node path hits the node_idx/seed=103+node_idx bug). --parallel
# runs the six attack conditions concurrently (6 x 8 = 48 workers).
#
# Usage:  sbatch job_scripts/eval_heuristics_full_obs_4000ep_ditet.sh
# Output: out/evaluations/29_04_heuristics_full_obs_mag4/{obs_user_dir, obs_capacity,
#         obs_demand, obs_interference_lie, obs_noise, obs_replay}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

MODEL_DIR="out/models/29_04"
MODEL_PATTERN="${MODEL_DIR}/**"
EXP_NAME="29_04_heuristics_full_obs_mag4"
MAG=4.0
N_EPS=4000          # total episodes per condition (split across --n-envs workers)
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
echo "Model: $MODEL_PATTERN | obs_type=full | jammers on | f=${N_BYZ} | mag=${MAG} | ${N_EPS}ep"

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

echo ""
echo "Finished at: $(date)"
echo "Outputs in out/evaluations/${EXP_NAME}/ (Delta-tau vs 29_04_nobyz_4000ep = 16.78)"
