#!/bin/bash
#SBATCH --job-name="byz_convergence"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Report PPO training convergence (rollout/ep_rew_mean) for the BORA variants,
# to judge whether heard/coordinated/full_obs plateaued or were still improving.
# Reads the TB event files; the verdict prints to logs/<jobid>.out.
#
# Usage:  sbatch job_scripts/check_convergence_ditet.sh
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export MPLBACKEND=Agg

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone

MODELS="/itet-stor/arbaur/net_scratch/out/models/learned_byzantine"

echo "Node: $(hostname) | $(date)"
echo ""

python src/Utils/check_byz_convergence.py \
    "${MODELS}/29_04_2byz_heard_60M/tb_logs" \
    "${MODELS}/29_04_2byz_coordinated_60M/tb_logs" \
    "${MODELS}/29_04_2byz_full_obs_60M/tb_logs"

echo ""
echo "Done: $(date)"
