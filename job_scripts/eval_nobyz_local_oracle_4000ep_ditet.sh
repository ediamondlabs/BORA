#!/bin/bash
#SBATCH --job-name="nobyz_loc_orac_4000"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Produce the missing 4000-episode runs so every BORA variant shares one
# episode count and one matched baseline:
#   no_byzantine : f=0 baseline of the victim policy (run_byzantine_sweep)
#   local        : direction-only BORA, local scope   (29_04_2byz_local)
#   oracle       : direction-only BORA, full scope     (29_04_2byz)
#
# local/oracle use 8 distinct-seed shards of 500 episodes (--episode-offset).
# After this lands, recompute Delta-tau for ALL variants against this baseline.
#
# Usage:  sbatch job_scripts/eval_nobyz_local_oracle_4000ep_ditet.sh
# Output: out/evaluations/{29_04_nobyz_4000ep, 29_04_2byz_local_4000ep, 29_04_2byz_oracle_4000ep}/
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="out/models/29_04"                                       # honest victim
SCRATCH="/itet-stor/arbaur/net_scratch/out"
BYZ_LOCAL="${SCRATCH}/models/learned_byzantine/29_04_2byz_local/byz_agent.zip"
BYZ_ORACLE="${SCRATCH}/models/learned_byzantine/29_04_2byz/byz_agent.zip"
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

AGENT_NAME=$(find "$MODEL_DIR" -name "*.zip" -printf "%T@ %p\n" \
    | sort -n | tail -1 | awk '{print $2}' | xargs basename | sed 's/\.zip$//')
echo "Honest agent: $AGENT_NAME"

# All three go through train_learned_byzantine (matched seeds, the seed fix).
# The no-Byzantine baseline uses --max-offset 0: the f=2 Byzantine nodes are
# present but their falsification is scaled to zero, so they broadcast truthfully
# and the throughput equals the no-attack baseline on identical seeds. This also
# avoids the obs_type=full single-agent eval path in run_byzantine_sweep
# (node_idx is None there for the joint honest policy).
run_scope () {   # $1 = experiment name, $2 = byz model, $3 = obs scope, $4 = max offset
    local logdir="out/evaluations/$1_4000ep"
    if [ ! -f "$2" ]; then
        echo "ERROR: Byzantine model not found at $2"
        exit 1
    fi
    mkdir -p "$logdir"
    echo "=== ${1} (scope=$3, max_offset=$4) ${N_SHARDS}x${EPS}ep -> ${logdir} ==="
    local pids=()
    for i in $(seq 0 $((N_SHARDS - 1))); do
        local off=$((i * EPS))
        python src/Utils/train_learned_byzantine.py --evaluate \
            --honest-model   "${MODEL_DIR}/**" \
            --byz-model      "$2" \
            --eval-config    "$EVAL_CONFIG" \
            --n-byz          "$N_BYZ" \
            --max-offset     "$4" \
            --obs-scope      "$3" \
            --attack-type    obs_user_dir \
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

echo "=== [1/3] no_byzantine baseline (max_offset 0) ===" ; run_scope 29_04_nobyz       "$BYZ_LOCAL"  local 0.0
echo "=== [2/3] local ==="                                ; run_scope 29_04_2byz_local  "$BYZ_LOCAL"  local 4.0
echo "=== [3/3] oracle ==="                               ; run_scope 29_04_2byz_oracle "$BYZ_ORACLE" full  4.0

echo ""
echo "Finished at: $(date)"
echo "Outputs in out/evaluations/: 29_04_nobyz_4000ep, 29_04_2byz_local_4000ep, 29_04_2byz_oracle_4000ep"
exit 0
