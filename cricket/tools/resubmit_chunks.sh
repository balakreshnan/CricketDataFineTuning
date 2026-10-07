#!/usr/bin/env bash
# Resubmit the remaining training of a train-full run as N chunks of a new length (they resume from its checkpoints).
#   PARTITION=batch-xdr,batch-spx bash resubmit_chunks.sh <first chunk job id> <N> <time limit> <stop-after minutes> [<merge job id>]
# Arguments are copied from the first chunk's log, so the same run dir / checkpoints are used.
set -euo pipefail
FIRST=$1 N=$2 TL=$3 DL=$4 MERGE=${5:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -a; source "$HERE/config.env"; set +a
LOG=$CROOT/logs/general_sa-cricket.sft-full-$FIRST.log
ARGS=$(grep -m1 "^script: train.py " "$LOG" | sed -E 's/^script: train.py //; s/ \(launch: .*//')
[[ -n "$ARGS" ]] || { echo "no script: line in $LOG"; exit 1; }
NODES=$(sacct -j "$FIRST" -X -n -o NNodes | head -1 | tr -d ' ')
prev=""
for _ in $(seq 1 "$N"); do
  dep=(); [[ -n "$prev" ]] && dep=(--dependency=afterany:"$prev")
  prev=$(SCRIPT=train.py JOB_ARGS="$ARGS" LAUNCH=torchrun DEADLINE_MINUTES=$DL \
    sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --chdir="$RUN_FROM" --export=ALL \
      --output="$CROOT/logs/%x-%j.log" --job-name=general_sa-cricket.sft-full --nodes="$NODES" --time="$TL" \
      "${dep[@]}" "$HERE/job.sbatch")
  echo "chunk $prev ($NODES nodes, $TL, partition $PARTITION, stop new steps after $DL min)"
done
if [[ -n "$MERGE" ]]; then
  scontrol update JobId="$MERGE" Dependency=afterok:"$prev"
  echo "merge $MERGE now waits for $prev: $(scontrol show job "$MERGE" | grep -oE 'Dependency=[^ ]*')"
fi
