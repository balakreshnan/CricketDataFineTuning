#!/bin/bash
#SBATCH -A general_sa
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# Speculative-decoding benchmark (specdec_bench.py): target alone vs target + each draft, on the cricket eval questions.
# submit.sh sets -N/-t and exports RUN_NAME, ARGS, DRAFTS, TARGET_ADAPTER, PUSH, PUSH_ONLY.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/specdec/specdec.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then export WANDB_MODE=disabled; fi
cd "$RUN_FROM"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export WANDB_PROJECT=$SD_WANDB_PROJECT OUT_DIR=$SD_DIR/$RUN_NAME RUN_NAME ARGS GPUS_PER_NODE DRAFTS TARGET_ADAPTER SD_HF_REPO
export PUSH=${PUSH:-0} PUSH_ONLY=${PUSH_ONLY:-0}
mkdir -p "$OUT_DIR/gpulog"
echo "[$(date +%T)] job $SLURM_JOB_ID specdec $RUN_NAME on $SLURM_JOB_NODELIST; target $MODEL_DIR adapter '${TARGET_ADAPTER}' drafts '$DRAFTS'"
srun --overlap --ntasks-per-node=1 bash -c 'exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 15 > "'"$OUT_DIR"'/gpulog/$(hostname -s)-'"$SLURM_JOB_ID"'.csv"' >/dev/null 2>&1 &
GPULOG_PID=$!
CENV=HF_TOKEN,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_MODE,WANDB_DIR,OUT_DIR,RUN_NAME,ARGS,GPUS_PER_NODE,DRAFTS,TARGET_ADAPTER,SD_HF_REPO,PUSH,PUSH_ONLY,HF_HUB_OFFLINE
if [[ $PUSH_ONLY != 1 ]]; then
  srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
       --no-container-remap-root --container-workdir="$RUN_FROM" --container-env=$CENV bash -c '
    set -euo pipefail
    keep_adapter=${TARGET_ADAPTER-}; keep_drafts=$DRAFTS
    source '"$CODE_DIR"'/specdec/specdec.env
    source "$VENV/bin/activate"
    export WANDB_PROJECT=$SD_WANDB_PROJECT TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
    drafts=""; for d in $keep_drafts; do drafts="$drafts${drafts:+,}$d=$MODELS_DIR/$d"; done
    common="--model_dir $MODEL_DIR --target_id $MODEL_ID --data_dir $DATA_LOCAL --out_dir $OUT_DIR --run_name $RUN_NAME"
    if [[ ${AGGREGATE:-0} == 1 ]]; then
      exec python '"$CODE_DIR"'/specdec/specdec_bench.py $common --aggregate $ARGS
    fi
    # one independent process per GPU; rank = node index x GPUs per node + GPU index
    export WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
    pids=""
    for i in $(seq 0 $(( GPUS_PER_NODE - 1 ))); do
      RANK=$(( SLURM_NODEID * GPUS_PER_NODE + i )) CUDA_VISIBLE_DEVICES=$i python '"$CODE_DIR"'/specdec/specdec_bench.py $common         --adapter "$keep_adapter" --drafts "$drafts" $ARGS &
      pids="$pids $!"
    done
    rc=0; for pid in $pids; do wait $pid || rc=1; done
    exit $rc
  '
  AGGREGATE=1 srun -N 1 -n 1 --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home        --no-container-remap-root --container-workdir="$RUN_FROM" --container-env=$CENV,AGGREGATE bash -c '
    set -euo pipefail
    source '"$CODE_DIR"'/specdec/specdec.env
    source "$VENV/bin/activate"
    export WANDB_PROJECT=$SD_WANDB_PROJECT WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
    exec python '"$CODE_DIR"'/specdec/specdec_bench.py --model_dir "$MODEL_DIR" --target_id "$MODEL_ID" --data_dir "$DATA_LOCAL"       --out_dir "$OUT_DIR" --run_name "$RUN_NAME" --aggregate $ARGS
  '
fi
kill "$GPULOG_PID" 2>/dev/null || true
if [[ $PUSH == 1 || $PUSH_ONLY == 1 ]]; then
  export HF_HUB_OFFLINE=0
  srun -N 1 -n 1 --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
       --no-container-remap-root --container-env=HF_TOKEN,OUT_DIR,RUN_NAME,ARGS,SD_HF_REPO,PUSH_ONLY bash -c '
    set -euo pipefail
    source '"$CODE_DIR"'/specdec/specdec.env
    source "$VENV/bin/activate"
    export HF_HUB_OFFLINE=0
    exec python '"$CODE_DIR"'/specdec/specdec_push.py --out_dir "$OUT_DIR" --run_name "$RUN_NAME" --repo_id "$SD_HF_REPO" \
      --dataset_id "$DATASET_ID" $( [[ $PUSH_ONLY == 1 ]] && echo "$ARGS" )
  '
fi
echo "[$(date +%T)] specdec done"
