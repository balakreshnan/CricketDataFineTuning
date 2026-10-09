#!/bin/bash
# Submit every remaining stage as one Slurm dependency chain (afterok), so it finishes without the laptop/bridge.
#   bash pipeline.sh [run-name]   env: PARTITION (tcpo), GEN_NODES (12), TRAIN_NODES (4), FROM (start|retry2),
#                                      STAGES (all|data|train), SFT_DATA (train input dir, default data/sft)
# Stages: data retry -> Dynamo resample of unsolved rows -> data fallback -> Dynamo on fallback questions ->
# data final (1:1 check, W&B, HF dataset push) -> pack (1-step bench on the real SFT data) -> SFT -> export ->
# Dynamo eval of the fine-tuned model -> report (W&B, model card facts) -> HF model push.
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
RUN=${1:-sft-rows-e1}
P=${PARTITION:-tcpo}; GEN_NODES=${GEN_NODES:-12}; TRAIN_NODES=${TRAIN_NODES:-4}
S=$CODE_DIR/submit.sh
log() { echo "$(date +%T) $*" | tee -a "$SLURM_LOGS/pipeline-$RUN.log"; }
sub() {  # sub <after-jobid|-> <name> <sbatch args...>  -> prints job id
  local dep=$1 name=$2; shift 2
  sbatch --parsable -p "$P" ${dep:+--dependency=afterok:$dep} -J "general_sa-olympics.$name" \
    -o "$SLURM_LOGS/%x-%j.out" --export=ALL "$@"
}

if [[ ${STAGES:-all} != train ]]; then
  if [[ ${FROM:-start} == start ]]; then
    j1=$(STAGE=retry sub "" data-retry "$CODE_DIR/data.sbatch")
    j2=$(K=6 TAG=rows SPLITS=rows_retry CONCURRENCY=1024 sub "$j1" datagen-retry -N "$GEN_NODES" -t 03:00:00 "$CODE_DIR/gen.sbatch")
    j3=$(STAGE=fallback sub "$j2" data-fallback "$CODE_DIR/data.sbatch")
    j4=$(K=8 TAG=rows SPLITS=rows_fallback CONCURRENCY=1024 sub "$j3" datagen-fallback -N 4 -t 02:00:00 "$CODE_DIR/gen.sbatch")
    log "data: retry $j1 -> gen $j2 -> fallback $j3 -> gen $j4"
  else  # FROM=retry2: rows still unsolved after retry + fallback -> all their questions, K=16, 28k-token budget
    j3=$(STAGE=retry RETRY_SPLIT=rows_retry2 sub "" data-retry2 "$CODE_DIR/data.sbatch")
    j4=$(K=16 MAX_TOKENS=28000 TAG=rows SPLITS=rows_retry2 CONCURRENCY=1024 \
           sub "$j3" datagen-retry2 -N "$GEN_NODES" -t 03:00:00 "$CODE_DIR/gen.sbatch")
    log "data: retry2 $j3 -> gen $j4"
  fi
  j5=$(STAGE=final sub "$j4" data-final "$CODE_DIR/data.sbatch")
  log "data: final+push $j5"
fi
[[ ${STAGES:-all} == data ]] && { squeue --me; exit 0; }

# STAGES=train: start now from SFT_DATA (a frozen copy of data/sft) instead of waiting for the data chain
train_args="--tp 1 --ep 8 --recompute selective --dispatcher hybridep --gbs 64"
# afterany: if the 1:1 check refuses the HF push, the SFT files were already written -> still train
# packing ~230K rows with 1 tokenizer worker took > 1 h on Grace (timed out at the default 1 h bench limit)
j6=$(PARTITION=$P DEPEND=${j5:-} DEP_TYPE=afterany BENCH_STEPS=1 TIME=04:00:00 bash $S bench 2 pack-$RUN $train_args)
j7=$(PARTITION=$P DEPEND=$j6 DEP_TYPE=afterok bash $S train "$TRAIN_NODES" $RUN $train_args \
       --epochs 1 --save-interval 100 --eval-interval 100 --eval-iters 10)
j8=$(PARTITION=$P DEPEND=$j7 DEP_TYPE=afterok bash $S export $RUN)
log "train: pack $j6 -> sft $j7 ($TRAIN_NODES nodes) -> export $j8"

j9=$(MODEL_DIR=$PROJ/runs/$RUN/hf K=4 TAG=eval-$RUN SPLITS="test rows_test" CONCURRENCY=1024 \
       sub "$j8" eval-$RUN -N 4 -t 02:00:00 "$CODE_DIR/gen.sbatch")
j10=$(STAGE=report RUN=$RUN sub "$j9" report-$RUN "$CODE_DIR/data.sbatch")
j11=$(PARTITION=$P DEPEND=$j10 DEP_TYPE=afterok bash $S push $RUN)
log "eval: dynamo $j9 -> report $j10 -> push model $j11 ($HF_MODEL_REPO)"
squeue --me
