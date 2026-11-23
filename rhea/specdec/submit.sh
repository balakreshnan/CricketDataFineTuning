#!/usr/bin/env bash
# Submit speculative-decoding jobs on Rhea (run on the login node: rhea/bin/rx 'bash ~/clustercodes/specdec/submit.sh <stage> ...').
#   submit.sh drafts                                   # download the draft models in $DRAFT_IDS to Lustre (1 node)
#   submit.sh smoke                                    # 1 node, 2 questions per GPU: proves target + draft work together
#   submit.sh bench [-N nodes] [-t HH:MM:SS] [-n run-name] [-- extra specdec_bench.py args]
#   PUSH=1 submit.sh bench ...                         # also upload results to the private HF dataset repo $SD_HF_REPO
#   submit.sh push -n <run-name> [-- --public]         # upload a finished run later
# Env: PROFILE=qwen36 (target model), DRAFTS="Qwen3.5-2B Qwen3.5-4B", TARGET_ADAPTER= (empty = base target), DEPEND=afterok:<id>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export CODE_DIR="$(dirname "$here")"
source "$here/specdec.env"
mkdir -p "$SLURM_LOGS"
stage=${1:?stage: drafts|smoke|bench|push}; shift || true
nodes=1; tlim=01:30:00; name=specdec-$(date +%m%d%H%M)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -N) nodes=$2; shift 2;; -t) tlim=$2; shift 2;; -n) name=$2; shift 2;; --) shift; break;; *) break;;
  esac
done
extra="$*"
common=(${DEPEND:+--dependency=$DEPEND} -A "$ACCOUNT" -p "$PARTITION" -o "$SLURM_LOGS/%x-%j.out")
exp="ALL,CODE_DIR=$CODE_DIR,GPUS_PER_NODE=$GPUS_PER_NODE,DRAFTS=$DRAFTS,TARGET_ADAPTER=$TARGET_ADAPTER,PUSH=${PUSH:-0}"
case "$stage" in
  drafts) sbatch "${common[@]}" -N 1 -t 01:00:00 -J general_sa-cricket-rhea.specdec-drafts --export="ALL,CODE_DIR=$CODE_DIR,DRAFT_IDS=$DRAFT_IDS" "$here/download_drafts.sbatch";;
  smoke)  sbatch "${common[@]}" -N 1 -t 00:40:00 -J general_sa-cricket-rhea.specdec-smoke \
            --export="$exp,RUN_NAME=smoke-$name,ARGS=--limit 8 --max_new_tokens 512 --gammas 8 --hf_reference $extra" "$here/specdec.sbatch";;
  bench)  sbatch "${common[@]}" -N "$nodes" -t "$tlim" -J general_sa-cricket-rhea.specdec \
            --export="$exp,RUN_NAME=$name,ARGS=$extra" "$here/specdec.sbatch";;
  push)   sbatch "${common[@]}" -N 1 -t 00:20:00 -J general_sa-cricket-rhea.specdec-push \
            --export="$exp,RUN_NAME=$name,ARGS=$extra,PUSH_ONLY=1" "$here/specdec.sbatch";;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
