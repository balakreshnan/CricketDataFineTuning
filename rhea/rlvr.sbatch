#!/bin/bash
#SBATCH -A general_sa
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# RLVR job: torchrun rlvr.py on every allocated node (one process per GPU) inside the Rubin PyTorch container.
# submit.sh sets -J/-N/-t/-p and exports RUN_NAME and ARGS. Re-submitting the same RUN_NAME resumes from the last checkpoint.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
cd "$RUN_FROM"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT=$RLVR_WANDB_PROJECT
if [[ -z "${WANDB_API_KEY:-}" ]]; then export WANDB_MODE=disabled; fi
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29500
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
end=$(scontrol show job "$SLURM_JOB_ID" | grep -o "EndTime=[^ ]*" | cut -d= -f2)
export TRAIN_DEADLINE=$(( $(date -d "$end" +%s) - 720 ))     # a step can take minutes: stop 12 min before the limit
export RUN_DIR=$PROJ/runs/$RUN_NAME RUN_NAME ARGS GPUS_PER_NODE
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/train_result.json" ]]; then echo "run $RUN_NAME already finished: nothing to do"; exit 0; fi
echo "[$(date +%T)] job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_JOB_NODELIST ($SLURM_NNODES nodes x $GPUS_PER_NODE GPUs), ends $end"
echo "run $RUN_NAME args: $ARGS"

# GPU logger on every node (utilisation, memory, power every 15 s) -> $RUN_DIR/gpulog/<node>-<jobid>.csv. Never fatal.
mkdir -p "$RUN_DIR/gpulog"
srun --overlap --ntasks-per-node=1 bash -c 'exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 15 > "'"$RUN_DIR"'/gpulog/$(hostname -s)-'"$SLURM_JOB_ID"'.csv"' >/dev/null 2>&1 &
GPULOG_PID=$!

srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-workdir="$RUN_FROM" \
     --container-env=HF_TOKEN,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_MODE,WANDB_DIR bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  export WANDB_PROJECT=$RLVR_WANDB_PROJECT    # config.env just reset it to the supervised project
  export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
  exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    '"$CODE_DIR"'/rlvr.py --model_dir "$MODEL_DIR" --data_dir "$DATA_LOCAL" --out_dir "$RUN_DIR" --run_name "$RUN_NAME" $ARGS
'
kill "$GPULOG_PID" 2>/dev/null || true
echo "[$(date +%T)] finished"
