#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J general_sa-olympics.setup
# One-off on a GB200 (aarch64) node: import the Dynamo-SGLang and NeMo images to Lustre (the login node can't:
# enroot-aufs2ovlfs needs capabilities there), then report what the Dynamo image offers.
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
# layers cached on Lustre (shared with the failed login-node attempt); unpack on node-local /tmp
export ENROOT_TEMP_PATH=/tmp/enroot-tmp-$SLURM_JOB_ID ENROOT_CACHE_PATH=$LUSTRE_DIR/cache/enroot-cache
mkdir -p "$ENROOT_TEMP_PATH"; df -h /tmp | tail -1
for pair in "$DYNAMO_IMAGE_REMOTE $DYNAMO_SQSH" "$NEMO_IMAGE_REMOTE $NEMO_SQSH"; do
  set -- $pair
  if [[ ! -f $2 ]]; then
    echo "[$(date +%T)] importing $1"
    enroot import -o "$2.part" "docker://$1" 2>&1 | grep -v "failed to set capabilities"
    mv "$2.part" "$2"
  fi
  ls -lh "$2"
done
rm -rf "/tmp/enroot-tmp-$SLURM_JOB_ID"

echo "[$(date +%T)] probing $DYNAMO_SQSH"
srun --container-image="$DYNAMO_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root bash -c '
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; uname -m; nproc; free -g | head -2
python -c "import sglang; print(\"sglang\", sglang.__version__)"
pip list 2>/dev/null | grep -i -E "^(ai-dynamo|ai-dynamo-runtime|sglang|flashinfer-python|torch|aiohttp|openai|httpx|transformers|wandb) "
which etcd nats-server python || true
echo ===== frontend; python -m dynamo.frontend --help 2>&1 | head -100
echo ===== sglang; python -m dynamo.sglang --help 2>&1 | grep -v "^\s*$" | grep -i -E -A2 "store|discovery|etcd|nats|request-plane|event-plane|endpoint|model-path|served|reasoning|mamba|tp-size|cuda-graph|context-length|^usage" | head -150
env | grep -i -E "^DYN_" || true
'
echo "[$(date +%T)] setup done"
