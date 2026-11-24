#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J general_sa-nemotron-rhea.model
# Download $MODEL_ID to $MODEL_DIR on Lustre and print its config, tensor-name patterns and chat-template tail
# (../download_model.py), which decide the model class, the LoRA target layers and the thinking-mode prompt.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemotron/nemotron.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
echo "[$(date +%T)] $MODEL_ID -> $MODEL_DIR"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/nemotron/nemotron.env
  source "$VENV/bin/activate"
  export MODEL_ID MODEL_DIR
  python '"$CODE_DIR"'/download_model.py
  du -sh "$MODEL_DIR"
  ls "$MODEL_DIR" | head -40
'
echo "[$(date +%T)] model download done"
