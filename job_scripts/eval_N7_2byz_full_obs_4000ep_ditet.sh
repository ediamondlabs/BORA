#!/bin/bash
#SBATCH --job-name="N7_full_obs_eval"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Evaluate the N=7 full_obs BORA attacker (f=2) plus its matched no-Byzantine
# baseline, 4000 episodes each in 8 distinct-seed shards. The no-byz baseline
# uses --max-offset 0 (Byzantine nodes present but broadcasting truthfully), so
# Delta-tau is computed against an N=7 baseline on identical seeds.
#
# Run AFTER train_learned_byzantine_N7_2byz_full_obs_60M_ditet.sh finishes.
#
# Usage:  sbatch job_scripts/eval_N7_2byz_full_obs_4000ep_ditet.sh
# Output: out/evaluations/{18_05_N7_nobyz_4000ep, 18_05_N7_2byz_full_obs_4000ep}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="${SCRATCH}/models/18_05_N7"                             # N=7 honest victim
BYZ="${SCRATCH}/models/learned_byzantine/18_05_N7_2byz_full_obs_60M/byz_agent.zip"
EVAL_CONFIG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
N_BYZ=2
EP_LEN=2000
EPS=500
N_SHARDS=8

set -o errexit

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs

echo "Node: $(hostname) | Start: $(date) | Job: ${SLURM_JOB_ID}"

if [ ! -f "$BYZ" ]; then
    echo "ERROR: N=7 full_obs model not found at $BYZ"
    echo "Run train_learned_byzantine_N7_2byz_full_obs_60M_ditet.sh first."
    exit 1
fi

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "N=7 honest agent: $AGENT_NAME"

run_scope () {   # $1 = experiment name, $2 = max offset
    local logdir="out/evaluations/$1_4000ep"
    mkdir -p "$logdir"
    echo "=== ${1} (max_offset=$2) ${N_SHARDS}x${EPS}ep -> ${logdir} ==="
    local pids=()
    for i in $(seq 0 $((N_SHARDS - 1))); do
        local off=$((i * EPS))
        python src/Utils/train_learned_byzantine.py --evaluate \
            --honest-model   "${MODEL_DIR}/**" \
            --byz-model      "$BYZ" \
            --eval-config    "$EVAL_CONFIG" \
            --n-byz          "$N_BYZ" \
            --max-offset     "$2" \
            --obs-scope      heard \
            --attack-type    obs_full \
            --eps            "$EPS" \
            --ep-len         "$EP_LEN" \
            --log-dir        "$logdir" \
            --agent-name     "$AGENT_NAME" \
            --episode-offset "$off" \
            >> "${logdir}/shard_${i}.log" 2>&1 &
        pids+=($!)
    done
    local failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=1; done
    [ "$failed" -eq 0 ] || { echo "Some ${1} shards failed"; exit 1; }
}

echo "=== [1/2] no_byzantine baseline (max_offset 0) ===" ; run_scope 18_05_N7_nobyz          0.0
echo "=== [2/2] N=7 full_obs attack ==="                  ; run_scope 18_05_N7_2byz_full_obs  4.0

echo ""
echo "Finished at: $(date)"
echo "Outputs: out/evaluations/{18_05_N7_nobyz_4000ep, 18_05_N7_2byz_full_obs_4000ep}"
exit 0
