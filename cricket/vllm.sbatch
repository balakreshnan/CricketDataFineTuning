#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# vLLM stages (dataset generation, evals). submit.sh sets name, nodes, time and:
#   VLLM_ARGS      gen_vllm.py flags (everything except --shard/--num_shards)
#   FINISH_SCRIPT  optional; then FINISH_ARGS - runs once afterwards in the PyTorch container (W&B, dataset, report)
# One vLLM engine per GPU (data parallel): shard = node * GPUs + local GPU.
set -euo pipefail
if [[ -f "$IMAGE_SQSH" ]]; then IMAGE="$IMAGE_SQSH"; else IMAGE="$IMAGE_REMOTE"; fi
project="$WANDB_PROJECT"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT="$project"
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != disabled ]]; then export WANDB_MODE=offline; fi
export RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export TRAIN_DEADLINE=$(( $(date +%s) + ${DEADLINE_MINUTES:-100} * 60 ))
export SHARD_LOG_DIR="$CROOT/logs/$RUN_NAME"
mkdir -p "$SHARD_LOG_DIR"
echo "[$(date +%T)] job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_JOB_NODELIST; vLLM image $VLLM_SQSH"
echo "gen_vllm.py $VLLM_ARGS (per-GPU logs: $SHARD_LOG_DIR)"

srun --container-image="$VLLM_SQSH" --container-mounts="$LUSTRE_REAL:$LUSTRE_REAL" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
  NGPU=$(nvidia-smi -L | wc -l); N=$((SLURM_NNODES * NGPU)); pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    s=$((SLURM_NODEID * NGPU + i))
    CUDA_VISIBLE_DEVICES=$i python3 "$CODE_DIR/gen_vllm.py" $VLLM_ARGS --shard $s --num_shards $N \
      > "$SHARD_LOG_DIR/shard$s.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  if [[ $SLURM_NODEID == 0 ]]; then grep -h "^\[" "$SHARD_LOG_DIR"/shard0.log | tail -5; fi
  exit $rc
' || { echo "a vLLM worker failed - see $SHARD_LOG_DIR"; tail -n 30 "$SHARD_LOG_DIR"/shard*.log | tail -n 60; }

if [[ -n "${FINISH_SCRIPT:-}" && "$FINISH_SCRIPT" != none ]]; then
  srun --nodes=1 --ntasks=1 --container-image="$IMAGE" --container-mounts="$LUSTRE_REAL:$LUSTRE_REAL" \
       --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
    source "$VENV/bin/activate"; exec python "$CODE_DIR/$FINISH_SCRIPT" $FINISH_ARGS'
fi
echo "[$(date +%T)] finished"
