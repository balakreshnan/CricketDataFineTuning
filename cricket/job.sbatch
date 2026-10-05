#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# Generic job for every cricket stage; submit.sh sets name, nodes, time, SCRIPT, JOB_ARGS and LAUNCH.
#   LAUNCH=torchrun : one process per GPU on every node (generation, training, evals)
#   LAUNCH=python   : a single process on the first node (prepare, push)
set -euo pipefail
if [[ -f "$IMAGE_SQSH" ]]; then IMAGE="$IMAGE_SQSH"; else IMAGE="$IMAGE_REMOTE"; fi
project="$WANDB_PROJECT"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT="$project"
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != disabled ]]; then export WANDB_MODE=offline; fi
export RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export TRAIN_DEADLINE=$(( $(date +%s) + ${DEADLINE_MINUTES:-100} * 60 ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29500
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
echo "[$(date +%T)] job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_JOB_NODELIST, image $IMAGE, cwd $(pwd)"
echo "script: $SCRIPT $JOB_ARGS (launch: $LAUNCH, stop new work after ${DEADLINE_MINUTES:-100} min)"

srun --mpi=pmix --container-image="$IMAGE" --container-mounts="$LUSTRE_REAL:$LUSTRE_REAL" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
  set -euo pipefail
  source "$VENV/bin/activate"
  # Triton bundles an older ptxas that rejects Rubin (sm_107); use the CUDA toolkit one in the image.
  if [[ -z "${TRITON_PTXAS_PATH:-}" && -x /usr/local/cuda/bin/ptxas ]]; then export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas; fi
  if [[ "$LAUNCH" == python ]]; then
    if [[ "$SLURM_NODEID" == 0 ]]; then exec python "$CODE_DIR/$SCRIPT" $JOB_ARGS; fi
    exit 0
  fi
  NGPU=$(nvidia-smi -L | wc -l)
  # python -m (not the torchrun script): the torchrun script in the image runs system python and would skip the venv
  exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$NGPU" \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$CODE_DIR/$SCRIPT" $JOB_ARGS
'
echo "[$(date +%T)] finished"
