#!/usr/bin/env bash
# Cricket reasoning pipeline on Hecate. Run on the login node:
#   bash .../code/cricket/submit.sh import-vllm      # once: NGC vLLM image -> squashfs on Lustre
#   bash .../code/cricket/submit.sh prepare          # download CricketData, build train/test questions (1 node)
#   bash .../code/cricket/submit.sh pilot            # 96 questions x k=4 on 1 node (vLLM): speed + quality check
#   bash .../code/cricket/submit.sh generate         # full reasoning dataset (vLLM, 4 nodes; resubmit to continue)
#   bash .../code/cricket/submit.sh push-dataset     # ONLY after approval: dataset -> Hugging Face (private)
#   bash .../code/cricket/submit.sh train            # LoRA SFT -> merge -> vLLM evals (base + fine-tuned) -> push
# Extra script flags: EXTRA_ARGS="..."; extra sbatch flags after the stage name (e.g. --dependency=afterok:123).
# One-to-one full dataset (all 425,119 source rows, 500K questions): prepare-full, pilot-full, generate-full [, retry-full]
# Dataset variants (default: questions / train / 4 nodes), e.g. a sampled 500K set:
#   QSET=questions-500k EXTRA_ARGS="--n_train 166700" bash .../submit.sh prepare
#   QSET=questions-500k GEN_NAME=train-500k GEN_NODES=8 bash .../submit.sh generate   (resubmit to continue)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a
source "$HERE/config.env"
set +a
mkdir -p "$CROOT"/{data,runs,checkpoints,logs}
stage=${1:-}
shift || true
COMMON=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --chdir="$RUN_FROM" --export=ALL --output="$CROOT/logs/%x-%j.log")

submit() {  # name nodes time launch script args minutes [sbatch flags...]   (PyTorch container: job.sbatch)
  local name=$1 nodes=$2 time=$3 launch=$4 script=$5 args=$6 minutes=$7
  shift 7
  SCRIPT=$script JOB_ARGS="$args ${EXTRA_ARGS:-}" LAUNCH=$launch DEADLINE_MINUTES=$minutes \
    sbatch "${COMMON[@]}" --job-name="general_sa-cricket.$name" --nodes="$nodes" --time="$time" "$@" "$HERE/job.sbatch"
}

submit_vllm() {  # name nodes time vllm_args finish_args minutes [sbatch flags...]   (vLLM container: vllm.sbatch)
  local name=$1 nodes=$2 time=$3 vargs=$4 fargs=$5 minutes=$6
  shift 6
  VLLM_ARGS="$vargs ${EXTRA_ARGS:-}" FINISH_SCRIPT=${FINISH_SCRIPT-finish_gen.py} FINISH_ARGS="$fargs" DEADLINE_MINUTES=$minutes \
    sbatch "${COMMON[@]}" --job-name="general_sa-cricket.$name" --nodes="$nodes" --time="$time" "$@" "$HERE/vllm.sbatch"
}

M="--model_dir $MODEL_DIR"
G="--k 4 --max_tokens 6144"
QD=$DATA_DIR/${QSET:-questions}          # question set folder (e.g. QSET=questions-500k)
GD=$DATA_DIR/gen/${GEN_NAME:-train}      # generated dataset folder (e.g. GEN_NAME=train-500k)
case "$stage" in
  import-vllm)
    src="docker://$(echo "$VLLM_IMAGE_REMOTE" | sed 's#/#\##')"   # nvcr.io/nvidia/vllm:tag -> nvcr.io#nvidia/vllm:tag
    jid=$(sbatch "${COMMON[@]}" --job-name=general_sa-cricket.import-vllm --nodes=1 --time=00:45:00 "$@" \
      --wrap "srun --ntasks=1 enroot import --output $VLLM_SQSH $src");;
  prepare)
    jid=$(submit prepare 1 00:30:00 python prepare.py "--data_dir $DATA_DIR --out_name ${QSET:-questions}" 25 "$@");;
  pilot)
    out=$DATA_DIR/gen/pilot-$(date +%m%d%H%M)
    jid=$(submit_vllm pilot 1 00:45:00 "$M --questions $DATA_DIR/questions/train.jsonl --out_dir $out/samples $G --limit 96" \
      "--questions_dir $DATA_DIR/questions --out_dir $out --limit 96" 35 "$@");;
  generate)
    jid=$(submit_vllm datagen "${GEN_NODES:-4}" 02:00:00 "$M --questions $QD/train.jsonl --out_dir $GD/samples $G" \
      "--questions_dir $QD --out_dir $GD" 100 "$@");;
  # ---- one-to-one dataset: every Valarmathy/CricketData row (425,119) gets >= 1 question (500,000 total by default)
  prepare-full)
    jid=$(submit prepare-full 1 00:45:00 python prepare.py "--data_dir $DATA_DIR --out_name ${QSET:-questions-full} --mode full" 40 "$@");;
  pilot-full)   # 40 questions per type (240) on 1 node: check wording/accuracy of every type before the big run
    QD=$DATA_DIR/${QSET:-questions-full}; out=$DATA_DIR/gen/pilot-full-$(date +%m%d%H%M)
    jid=$(FINISH_SCRIPT=build_full_dataset.py submit_vllm pilot-full 1 00:45:00 \
      "$M --questions $QD/pilot.jsonl --out_dir $out/samples $G" \
      "--questions_dir $QD --questions_file pilot.jsonl --out_dir $out" 35 "$@");;
  generate-full)  # resumable: resubmit (any node count) until nothing is pending
    QD=$DATA_DIR/${QSET:-questions-full}; GD=$DATA_DIR/gen/${GEN_NAME:-train-full}
    jid=$(FINISH_SCRIPT=build_full_dataset.py submit_vllm datagen-full "${GEN_NODES:-8}" 02:00:00 \
      "$M --questions $QD/all.jsonl --out_dir $GD/samples $G" "--questions_dir $QD --out_dir $GD" 100 "$@");;
  retry-full)   # questions with no correct sample: 8 more samples each, then rebuild (keeps one-to-one coverage)
    QD=$DATA_DIR/${QSET:-questions-full}; GD=$DATA_DIR/gen/${GEN_NAME:-train-full}
    jid=$(FINISH_SCRIPT=build_full_dataset.py submit_vllm retry-full "${GEN_NODES:-1}" 01:00:00 \
      "$M --questions $GD/retry.jsonl --out_dir $GD/samples --tag retry --k 8 --max_tokens 8000 --seed 1" \
      "--questions_dir $QD --out_dir $GD" 50 "$@");;
  fix-ambiguous)  # clarify runs<=wickets scores ("1/3") in unsolved questions before a retry (tag retry2)
    QD=$DATA_DIR/${QSET:-questions-full}; GD=$DATA_DIR/gen/${GEN_NAME:-train-full}
    jid=$(submit fix-ambiguous 1 00:20:00 python tools/fix_ambiguous_scores.py "$QD $GD" 15 "$@");;
  bench)        # SFT throughput benchmark on 1 node: EXTRA_ARGS = the variant's train.py flags
    jid=$(submit sft-bench 1 00:40:00 torchrun train.py "$M --data_dir $GD/final --croot $CROOT --no_eval \
--run_dir $CROOT/runs/bench-$(date +%m%d%H%M%S)-$RANDOM --dev_rows 64 --save_steps 100000 --log_steps 5" 35 "$@");;
  push-full)    # upload the one-to-one dataset (refuses if any source row is uncovered)
    GD=$DATA_DIR/gen/${GEN_NAME:-train-full}
    jid=$(submit push-full 1 01:00:00 python push_full_dataset.py \
      "--final_dir $GD/final --repo_id ${HF_FULL_DATASET_REPO:-${HF_DATASET_REPO}-Full}" 55 "$@");;
  push-dataset)
    jid=$(submit push-dataset 1 00:20:00 python push_dataset.py \
      "--final_dir $GD/final --repo_id $HF_DATASET_REPO --confirm" 15 "$@");;
  train)
    jid=$(submit sft 4 02:00:00 torchrun train.py "$M --data_dir $GD/final --croot $CROOT --no_eval" 90 "$@")
    run=$CROOT/runs/general_sa-cricket.sft-$jid; ck=$CROOT/checkpoints/general_sa-cricket.sft-$jid
    T="--questions $GD/final/test.jsonl --out_dir $run/eval $G"
    mj=$(EXTRA_ARGS= submit merge 1 00:40:00 python merge.py "$M --adapter $ck --out $ck/merged" 35 --dependency=afterok:$jid)
    bj=$(EXTRA_ARGS= FINISH_SCRIPT=none submit_vllm eval-base 1 01:00:00 "$M $T --tag base" "" 50)
    fj=$(EXTRA_ARGS= FINISH_SCRIPT=finish_eval.py submit_vllm eval-ft 1 01:00:00 "--model_dir $ck/merged $T --tag ft" \
      "--run_dir $run --adapter_dir $ck --k 4 --hub_model_id $HF_MODEL_REPO" 50 --dependency=afterok:$mj:$bj)
    echo "sft $jid -> merge $mj -> eval-ft $fj (with eval-base $bj); run dir $run";;
  train-full)   # 1 epoch on the one-to-one dataset within the 5 h partition limit (benchmarked: batch 8/GPU, length-grouped)
    GD=$DATA_DIR/gen/${GEN_NAME:-train-full}
    RUN=general_sa-cricket.sft-full-$(date +%m%d%H%M); run=$CROOT/runs/$RUN; ck=$CROOT/checkpoints/$RUN
    A="$M --data_dir $GD/final --croot $CROOT --run_dir $run --no_eval --epochs 1 --per_device_batch 8 --grad_accum 1 \
--sampling group_by_length --lr 2e-4 --warmup_ratio 0.03 --dev_rows 1000 --save_steps 200 --save_total_limit 2 --log_steps 10"
    N=${TRAIN_NODES:-8}
    s1=$(submit sft-full "$N" 04:59:00 torchrun train.py "$A" 280 "$@")
    s2=$(EXTRA_ARGS= submit sft-full "$N" 04:59:00 torchrun train.py "$A" 280 --dependency=afterany:$s1)  # resumes; no-op if done
    T="--questions $GD/final/eval.jsonl --out_dir $run/eval $G"
    mj=$(EXTRA_ARGS= submit merge 1 00:40:00 python merge.py "$M --adapter $ck --out $ck/merged" 35 --dependency=afterok:$s2)
    bj=$(EXTRA_ARGS= FINISH_SCRIPT=none submit_vllm eval-base 1 01:00:00 "$M $T --tag base" "" 50)
    fj=$(EXTRA_ARGS= FINISH_SCRIPT=finish_eval.py submit_vllm eval-ft 1 01:00:00 "--model_dir $ck/merged $T --tag ft" \
      "--run_dir $run --adapter_dir $ck --k 4 --hub_model_id ${HF_FULL_MODEL_REPO:-${HF_MODEL_REPO/-LoRA/-Full-LoRA}}" 50 \
      --dependency=afterok:$mj:$bj)
    jid=$s1
    echo "sft-full $s1 -> (resume-safety) $s2 -> merge $mj -> eval-ft $fj (with eval-base $bj); run dir $run";;
  *)
    sed -n '2,10p' "$0"; exit 1;;
esac
echo "$stage job $jid (logs in $CROOT/logs/)"
