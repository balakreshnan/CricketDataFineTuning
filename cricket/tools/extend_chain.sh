#!/usr/bin/env bash
# Append N resume chunks to a running train-full chain and re-point a downstream job (merge) at the new last chunk.
#   bash extend_chain.sh <any chunk job id of the chain> <last chunk job id> <N> [<merge job id>]
# Chunk arguments are taken from the "script:" line of the first job's log, so the new chunks resume the same run.
set -euo pipefail
FIRST=$1 LAST=$2 N=$3 MERGE=${4:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -a; source "$HERE/config.env"; set +a
LOG=$CROOT/logs/general_sa-cricket.sft-full-$FIRST.log
ARGS=$(grep -m1 "^script: train.py " "$LOG" | sed -E 's/^script: train.py //; s/ \(launch: .*//')
[[ -n "$ARGS" ]] || { echo "no script: line in $LOG"; exit 1; }
PART=$(squeue -j "$LAST" -h -o %P)
TL=$(squeue -j "$LAST" -h -o %l)
NODES=$(squeue -j "$LAST" -h -o %D)
DL=$(grep -m1 "stop new work after" "$LOG" | sed -E 's/.*after ([0-9]+) min.*/\1/')
prev=$LAST
for _ in $(seq 1 "$N"); do
  prev=$(SCRIPT=train.py JOB_ARGS="$ARGS" LAUNCH=torchrun DEADLINE_MINUTES=$DL \
    sbatch --parsable --account="$ACCOUNT" --partition="$PART" --chdir="$RUN_FROM" --export=ALL \
      --output="$CROOT/logs/%x-%j.log" --job-name=general_sa-cricket.sft-full --nodes="$NODES" --time="$TL" \
      --dependency=afterany:"$prev" "$HERE/job.sbatch")
  echo "chunk $prev (after previous, $NODES nodes, $TL, partition $PART, stop new steps after $DL min)"
done
if [[ -n "$MERGE" ]]; then
  scontrol update JobId="$MERGE" Dependency=afterok:"$prev"
  echo "merge $MERGE now waits for $prev: $(scontrol show job "$MERGE" | grep -oE 'Dependency=[^ ]*')"
fi
