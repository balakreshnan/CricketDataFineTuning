#!/usr/bin/env bash
# Submit Rhea jobs (run on the login node via: rhea/bin/rx 'bash ~/clustercodes/submit.sh <stage> ...').
#   submit.sh setup                        # venv + model + dataset download to Lustre (1 node)
#   submit.sh smoke                        # 1 node, 20 steps on 2k rows: proves the pipeline before the real run
#   submit.sh train [-N nodes] [-t HH:MM:SS] [-n run-name] [-- extra train.py args]
#   DEPEND=afterany:<jobid> submit.sh train -n <same run-name> ...   # queue an automatic resume
#   submit.sh eval -N 4 -t 01:30:00 -n <run-name>   # base vs fine-tuned benchmark on the 600 eval questions
#   submit.sh push -n <run-name> [-- --public]   # upload the finished adapter to $HF_MODEL_REPO (private by default)
#   PUSH=1 submit.sh train ...             # also push the final adapter (private) to $HF_MODEL_REPO
# Env: PARTITION, ACCOUNT, GPUS_PER_NODE (defaults in config.env; set from discover.sh output).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export CODE_DIR=$here
source "$here/config.env"
mkdir -p "$SLURM_LOGS"
stage=${1:?stage: setup|smoke|train}; shift || true
nodes=1; tlim=04:00:00; name=cricket-lora-$(date +%m%d%H%M)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -N) nodes=$2; shift 2;; -t) tlim=$2; shift 2;; -n) name=$2; shift 2;; --) shift; break;; *) break;;
  esac
done
extra="$*"
common=(${DEPEND:+--dependency=$DEPEND} -A "$ACCOUNT" -p "$PARTITION" -o "$SLURM_LOGS/%x-%j.out")
case "$stage" in
  setup) sbatch "${common[@]}" -N 1 -t 02:00:00 -J general_sa-cricket-rhea.setup "$here/setup.sbatch";;
  smoke) sbatch "${common[@]}" -N 1 -t 00:40:00 -J general_sa-cricket-rhea.smoke \
           --export=ALL,CODE_DIR="$here",GPUS_PER_NODE="$GPUS_PER_NODE",RUN_NAME=smoke-$name,ARGS="--train_limit 2000 --max_steps 20 --eval_steps 10 --save_steps 20 --dev_rows 64 $extra" \
           "$here/train.sbatch";;
  train) sbatch "${common[@]}" -N "$nodes" -t "$tlim" -J "general_sa-cricket-rhea.$name" \
           --export=ALL,CODE_DIR="$here",GPUS_PER_NODE="$GPUS_PER_NODE",RUN_NAME="$name",ARGS="${PUSH:+--hub_model_id $HF_MODEL_REPO }--group_by_length $extra" \
           "$here/train.sbatch";;
  eval)  sbatch "${common[@]}" -N "$nodes" -t "$tlim" -J general_sa-cricket-rhea.eval --export=ALL,CODE_DIR="$here",GPUS_PER_NODE="$GPUS_PER_NODE",RUN_NAME="$name",ARGS="$extra" "$here/eval.sbatch";;
  push)  sbatch "${common[@]}" -N 1 -t 00:20:00 -J general_sa-cricket-rhea.push --export=ALL,CODE_DIR="$here",RUN_NAME="$name",ARGS="$extra" "$here/push.sbatch";;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
