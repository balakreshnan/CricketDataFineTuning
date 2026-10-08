#!/bin/bash
#SBATCH -A general_sa
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# One job script for every stage; submit.sh sets -J/-N/-t/-p and exports STAGE + ARGS.
#   STAGE=import : HF -> Megatron checkpoint (1 node, EP8)
#   STAGE=train  : torchrun train.py $ARGS on every node (bench runs too)
#   STAGE=export : Megatron -> HF checkpoint ($ARGS = <megatron iter dir> <hf out dir>)
#   STAGE=push   : upload_hf.py $ARGS (first node only)
set -euo pipefail
source "$HOME/clustercodes/nemotron-cricket/config.env"
cd "$RUN_FROM"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != disabled ]]; then export WANDB_MODE=offline; fi
export MODEL_DIR=$(ls -d "$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29500
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export STAGE ARGS
echo "[$(date +%T)] job $SLURM_JOB_ID ($SLURM_JOB_NAME) stage=$STAGE on $SLURM_JOB_NODELIST ($SLURM_NNODES nodes)"
echo "args: $ARGS"

srun --mpi=pmix --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" \
     --container-env=HF_TOKEN,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_MODE bash -c '
  set -euo pipefail
  # /opt/venv is the image env with Megatron-Bridge. convert.sh is not used: its uv run re-resolves deps and
  # pulls a huggingface-hub version that the image transformers rejects.
  PY=/opt/venv/bin/python
  # node-local compile caches: a Lustre cache shared by all ranks races (inductor: CUDA driver error: file not found)
  export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
  CODE_DIR=$HOME/clustercodes/nemotron-cricket   # ~/.bashrc (env.sh) can reset CODE_DIR inside the container
  CONV="$PY -m torch.distributed.run --nproc_per_node=8 /opt/Megatron-Bridge/scripts/conversion/run_conversion.py"
  case "$STAGE" in
    import)
      [[ "$SLURM_NODEID" == 0 ]] || exit 0
      exec $CONV import --device gpu --tp 1 --pp 1 --ep 8 --etp 1 \
        --hf-model "$MODEL_DIR" --megatron-path $ARGS ;;
    export)
      [[ "$SLURM_NODEID" == 0 ]] || exit 0
      set -- $ARGS
      # a run dir (not an iter dir) means: its latest checkpoint, resolved now rather than at submit time
      if [[ -f "$1/latest_checkpointed_iteration.txt" ]]; then
        set -- "$1/iter_$(printf %07d "$(cat "$1/latest_checkpointed_iteration.txt")")" "$2"
      fi
      echo "exporting $1 -> $2"
      exec $CONV export --device gpu --tp 1 --pp 1 --ep 8 --etp 1 \
        --hf-model "$MODEL_DIR" --megatron-path "$1" --hf-path "$2" \
        --torch-dtype bfloat16 --export-weight-dtype bfloat16 ;;
    eval)
      [[ "$SLURM_NODEID" == 0 ]] || exit 0
      exec $PY "$CODE_DIR/eval_vllm.py" $ARGS ;;
    push)
      [[ "$SLURM_NODEID" == 0 ]] || exit 0
      exec $PY "$CODE_DIR/upload_hf.py" $ARGS ;;
    train)
      exec $PY -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node=8 \
        --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        "$CODE_DIR/train.py" --hf-model "$MODEL_DIR" $ARGS ;;
  esac
'
echo "[$(date +%T)] finished stage=$STAGE"
