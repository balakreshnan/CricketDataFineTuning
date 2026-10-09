#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J general_sa-olympics.data
# CPU stages of the 1:1 dataset, in the NeMo image on a compute node (login node kills long CPU jobs).
#   STAGE=retry     : list rows with no correct trace yet -> questions_$RETRY_SPLIT.jsonl (default rows_retry)
#   STAGE=fallback  : rows still unsolved -> template FB_LEVEL (1) steps on -> questions_rows_fallback[N].jsonl
#   STAGE=final     : build the dataset + SFT files, log stats to W&B, push to $HF_DATASET_REPO (refuses unless 1:1)
#   STAGE=report    : base vs fine-tuned benchmark report for RUN (eval_report.py), logged to W&B
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export PROJ RAW_DIR MODEL_ID WANDB_PROJECT HF_DATASET_REPO STAGE RUN=${RUN:-} RETRY_SPLIT=${RETRY_SPLIT:-rows_retry} SFT_DATA=${SFT_DATA:-}
export FB_LEVEL=${FB_LEVEL:-1} FINAL_ARGS=${FINAL_ARGS:-}   # e.g. "--allow-missing 8" (user-approved shortfall)
srun --container-image="$NEMO_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" \
     --container-env=HF_TOKEN,WANDB_API_KEY,WANDB_ENTITY bash -c '
set -euo pipefail
cd /home/bbalakreshna/clustercodes/olympics
PY=/opt/venv/bin/python; D=$PROJ/data; G=$PROJ/gen/rows
case $STAGE in
  retry)    $PY build_dataset.py --gen $G --unsolved-out $D/questions_${RETRY_SPLIT:-rows_retry}.jsonl | tail -25 ;;
  fallback) $PY build_dataset.py --gen $G --unsolved-out $D/unsolved_rows.jsonl | tail -25
            $PY prep_rows.py --fallback $FB_LEVEL ;;
  final)    $PY build_dataset.py --gen $G --wandb --push $HF_DATASET_REPO $FINAL_ARGS ;;
  report)   $PY eval_report.py $RUN ;;
esac'
