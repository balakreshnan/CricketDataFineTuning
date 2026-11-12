#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J general_sa-cricket-rhea.setup
# One job on a compute node (aarch64 Vera + 4x Vera Rubin):
#   1) import the Rubin PyTorch container to a squashfs on Lustre (login node is x86_64, so do it here)
#   2) venv on Lustre on top of the container's torch (transformers, peft, fla, wandb)
#   3) download model + dataset to Lustre
# Nothing large is written to $HOME. Needs HF_TOKEN in $SECRETS_FILE (dataset is private).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
mkdir -p "$VENV_DIR" "$MODELS_DIR" "$DATA_DIR" "$CONTAINER_DIR" "$SLURM_LOGS" "$PIP_CACHE_DIR" "$LUSTRE_DIR/cache/enroot-cache"
echo "[$(date +%T)] node $(hostname) arch $(uname -m)"

if [[ ! -f "$IMAGE_SQSH" ]]; then
  echo "[$(date +%T)] importing $IMAGE_REMOTE -> $IMAGE_SQSH"
  export ENROOT_TEMP_PATH=/raid/enroot-tmp-$SLURM_JOB_ID ENROOT_CACHE_PATH=$LUSTRE_DIR/cache/enroot-cache
  mkdir -p "$ENROOT_TEMP_PATH" 2>/dev/null || { export ENROOT_TEMP_PATH=/tmp/enroot-tmp-$SLURM_JOB_ID; mkdir -p "$ENROOT_TEMP_PATH"; }
  enroot import -o "$IMAGE_SQSH.part" "docker://${IMAGE_REMOTE/\//#}" 2>&1 | grep -v "failed to set capabilities" | tail -20
  mv "$IMAGE_SQSH.part" "$IMAGE_SQSH"
  rm -rf "$ENROOT_TEMP_PATH"
fi
ls -lh "$IMAGE_SQSH"

srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN bash -c '
set -euo pipefail
source '"$CODE_DIR"'/config.env
python - <<PY
import platform, torch
print("arch", platform.machine(), "| python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
[[ -x "$VENV/bin/python" ]] || python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
# pin the container builds of torch/triton/numpy so pip cannot replace them
pip list --format=freeze 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy)==" > "$VENV/constraints.txt" || true
echo "pinned:"; cat "$VENV/constraints.txt"
PIP_CONSTRAINT="$VENV/constraints.txt" pip install -q "transformers>=5.8.0" "peft>=0.18.0" "accelerate>=1.10.0" \
    "huggingface_hub>=1.0.0" "wandb>=0.22.0" "flash-linear-attention>=0.4.0" matplotlib
python - <<PY
import torch, transformers, peft, wandb
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401  (fails if transformers is too old)
try:
    import fla; f = fla.__version__
except Exception as e:
    f = f"NOT AVAILABLE ({e}): linear attention falls back to the slow torch path"
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| wandb", wandb.__version__, "| fla", f)
PY
echo "[$(date +%T)] downloading model + dataset to Lustre"
python - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("$MODEL_ID", local_dir="$MODEL_DIR", token=tok, max_workers=16)
snapshot_download("$DATASET_ID", repo_type="dataset", local_dir="$DATA_LOCAL", token=tok, max_workers=8)
PY
du -sh "$MODEL_DIR" "$DATA_LOCAL"; ls "$DATA_LOCAL" "$DATA_LOCAL/data" | head -30
wc -l "$DATA_LOCAL"/data/*.jsonl | tail -1
'
echo "[$(date +%T)] setup done"
