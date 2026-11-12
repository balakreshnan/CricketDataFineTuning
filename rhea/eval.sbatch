#!/bin/bash
#SBATCH -A general_sa
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# Base vs fine-tuned benchmark for a finished run (eval.py). submit.sh sets -N/-t and exports RUN_NAME and ARGS.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
cd "$RUN_FROM"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29500
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export RUN_DIR=$PROJ/runs/$RUN_NAME ARGS GPUS_PER_NODE
echo "[$(date +%T)] job $SLURM_JOB_ID eval of $RUN_NAME on $SLURM_JOB_NODELIST"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
  exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    '"$CODE_DIR"'/eval.py --model_dir "$MODEL_DIR" --data_dir "$DATA_LOCAL" --run_dir "$RUN_DIR" $ARGS
'
echo "[$(date +%T)] eval done"
