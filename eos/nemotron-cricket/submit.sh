#!/bin/bash
# Submit one stage from the login node. Logs: $LOG_DIR/nemotron-cricket/<job-name>-<jobid>.out
#   submit.sh import                                  HF -> Megatron base checkpoint (1 node)
#   submit.sh bench <nodes> <tag> [train.py args]     short throughput run (no ckpt/eval)
#   submit.sh train <nodes> <run> [train.py args]     full SFT, resumable; chain with DEPEND=<jobid>
#   submit.sh export <run> [iter_dir]                 Megatron -> HF (default: latest iter of <run>)
#   submit.sh eval <run|base>                        600-question benchmark with vLLM (base = original model)
#   submit.sh push <run>                              upload exported HF model to $HF_MODEL_REPO
# Env: PARTITION (default batch), TIME (default per stage), DEPEND=<jobid> (+ DEP_TYPE, default afterany).
set -euo pipefail
source "$HOME/clustercodes/nemotron-cricket/config.env"
cd "$RUN_FROM"
mkdir -p "$SLURM_LOGS" "$PROJ"/{ckpts,runs}
BASE_CKPT=$PROJ/ckpts/base-megatron
stage=$1; shift
dep=${DEPEND:+--dependency=${DEP_TYPE:-afterany}:$DEPEND}   # DEP_TYPE=afterok for export/push

common="--data $PROJ/data/sft --pretrained $BASE_CKPT"
case "$stage" in
  import) N=1; T=${TIME:-01:00:00}; NAME=import; ARGS="$BASE_CKPT" ;;
  bench)  N=$1; tag=$2; shift 2; T=${TIME:-01:00:00}; NAME="bench-$tag"
          ARGS="--run-name bench-$tag-N$N $common --save /dev/null --bench ${BENCH_STEPS:-25} $*" ;;
  train)  N=$1; run=$2; shift 2; T=${TIME:-04:00:00}; NAME="sft-$run"
          ARGS="--run-name $run $common --save $PROJ/ckpts/$run --exit-mins ${EXIT_MINS:-215} $*" ;;
  export) run=$1; it=${2:-$PROJ/ckpts/$run}; N=1; T=${TIME:-01:00:00}   # run dir -> latest iter at job start
          NAME="export-$run"; ARGS="$it $PROJ/runs/$run/hf" ;;
  eval)   run=$1; N=1; T=${TIME:-01:30:00}; NAME="eval-$run"   # run=base -> the original HF model
          m=$PROJ/runs/$run/hf; [[ $run == base ]] && m=$(ls -d $HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION)
          ARGS="--model $m --out $PROJ/runs/$run/eval/benchmark600.json" ;;
  push)   run=$1; N=1; T=${TIME:-02:00:00}; NAME="push-$run"; ARGS="$PROJ/runs/$run $HF_MODEL_REPO" ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac

STAGE=${stage/bench/train} ARGS="$ARGS" sbatch --parsable -p "$PARTITION" -N "$N" -t "$T" $dep \
  -J "general_sa-nemotron-cricket.$NAME" -o "$SLURM_LOGS/%x-%j.out" \
  --export=ALL "$CODE_DIR/job.sbatch"
