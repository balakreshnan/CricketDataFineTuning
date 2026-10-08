#!/bin/bash
#SBATCH -A general_sa
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J general_sa-nemotron-cricket.setup
# One-off: import the NeMo image to Lustre, download model + dataset into the Lustre HF cache,
# and report which training stacks in the image know this model.
set -euo pipefail
source "$HOME/clustercodes/nemotron-cricket/config.env"
cd "$RUN_FROM"
mkdir -p "$PROJ"/{data,runs,ckpts} "$CONTAINER_DIR"

if [[ ! -f "$IMAGE_SQSH" ]]; then
  echo "[$(date +%T)] importing $IMAGE_REMOTE"
  export ENROOT_TEMP_PATH=/tmp ENROOT_CACHE_PATH=/tmp/enroot-cache-$USER
  enroot import -o "$IMAGE_SQSH.part" "docker://$IMAGE_REMOTE" && mv "$IMAGE_SQSH.part" "$IMAGE_SQSH"
fi
ls -lh "$IMAGE_SQSH"

srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
set -euo pipefail
source "$HOME/clustercodes/nemotron-cricket/config.env"
nvidia-smi -L
python -c "import torch, transformers; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"gpus\", torch.cuda.device_count())"
echo "== downloads (public repos, no token needed)"
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --max-workers 16 | tail -1
hf download "$DATASET_ID" --repo-type dataset --revision "$DATASET_REVISION" | tail -1
echo "== training stacks in the image"
pip list 2>/dev/null | grep -i -E "^(megatron|nemo|nemo-automodel|nemo_automodel|transformer.engine|mamba|causal|peft|trl|accelerate|wandb|vllm) " || true
python - <<EOF
import importlib, pkgutil, os
for mod in ["megatron.bridge", "nemo_automodel", "mamba_ssm", "causal_conv1d", "transformer_engine"]:
    try:
        m = importlib.import_module(mod); print("OK ", mod, getattr(m, "__version__", ""), os.path.dirname(m.__file__))
    except Exception as e:
        print("NO ", mod, type(e).__name__, e)
EOF
for d in /opt/Megatron-Bridge /opt/Automodel /opt/NeMo; do [[ -d $d ]] && echo "dir $d"; done
grep -rIl -i -E "nemotron.?3\.?5|lightning" /opt/Megatron-Bridge/src /opt/Automodel/nemo_automodel /opt/Automodel/examples 2>/dev/null | head -30 || true
echo "-- nemotron recipes/examples"
find /opt/Megatron-Bridge /opt/Automodel -ipath "*nemotron*" \( -name "*.py" -o -name "*.yaml" \) 2>/dev/null | grep -v test | head -60
'
echo "[$(date +%T)] setup done"
