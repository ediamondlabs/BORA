#!/bin/bash
#SBATCH --job-name="shard_sanity"
#SBATCH --output=/home/arbaur/resilient-backbone/logs/%j.out
#SBATCH --error=/home/arbaur/resilient-backbone/logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=arton12

# ---------------------------------------------------------------------------
# Sanity check for the eval sharding seed fix.
#
# Two shards with different --episode-offset MUST evaluate different episodes
# (different throughput values). Before the fix every shard re-ran the same
# fixed-seed sequence, so the two CSVs were identical.
#
#   PASS = the two shards differ      (sharding fix is active)
#   FAIL = the two shards are identical(fix not applied / not synced to cluster)
#
# Usage:  sbatch job_scripts/eval_shard_sanity_check.sh
#   (or:  bash job_scripts/eval_shard_sanity_check.sh   on a compute node)
# ---------------------------------------------------------------------------

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

SCRATCH="/itet-stor/arbaur/net_scratch/out"
MODEL_DIR="out/models/29_04"
BYZ="${SCRATCH}/models/learned_byzantine/29_04_2byz_heard_60M/byz_agent.zip"
CFG="src/examples/evaluate/example_obs_manipulation_eval_config.yaml"
OUT="${SCRATCH}/evaluations/_shard_sanity"
EPS=3
EP_LEN=50

export MAMBA_ROOT_PREFIX="/scratch/arbaur/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate manet
cd ~/resilient-backbone
mkdir -p logs
rm -rf "$OUT"; mkdir -p "$OUT"

if [ ! -f "$BYZ" ]; then
    echo "ERROR: model not found at $BYZ"
    exit 1
fi

run_shard () {   # $1 = episode offset
    python src/Utils/train_learned_byzantine.py --evaluate \
        --honest-model "${MODEL_DIR}/**" \
        --byz-model    "$BYZ" \
        --eval-config  "$CFG" \
        --n-byz 2 --max-offset 4.0 \
        --obs-scope heard --attack-type obs_user_dir \
        --eps "$EPS" --ep-len "$EP_LEN" \
        --log-dir "$OUT" --agent-name sanity \
        --episode-offset "$1" > "$OUT/run_$1.log" 2>&1
}

echo "Running shard offset 0 ..."   ; run_shard 0
echo "Running shard offset 500 ..." ; run_shard 500

A="$OUT/learned_obs_user_dir/sanity.csv"
B="$OUT/learned_obs_user_dir/sanity_500.csv"
if [ ! -f "$A" ] || [ ! -f "$B" ]; then
    echo "FAIL: expected CSVs were not produced — check $OUT/run_*.log"
    ls -R "$OUT"
    exit 1
fi

# Compare the throughput value column (field 4). The episode label (field 2) is
# offset by design, so it is excluded from the comparison.
VA=$(cut -d, -f4 "$A" | tail -n +2)
VB=$(cut -d, -f4 "$B" | tail -n +2)

echo ""
if [ "$VA" == "$VB" ]; then
    echo "=== FAIL: the two shards produced IDENTICAL throughput values ==="
    echo "Sharding still duplicates episodes (fix not applied or not synced)."
    exit 1
fi

echo "=== PASS: the two shards produced DIFFERENT episodes ==="
echo "Sharding fix is active — the 4000ep run will give 4000 distinct episodes."
