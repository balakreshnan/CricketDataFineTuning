#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J general_sa-cricket-rhea.model
# Download $MODEL_ID to $MODEL_DIR on Lustre (PROFILE picks the model) and print its config and layer layout.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
echo "[$(date +%T)] profile '${PROFILE}' model $MODEL_ID -> $MODEL_DIR"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  export MODEL_ID MODEL_DIR
  python '"$CODE_DIR"'/download_model.py
  du -sh "$MODEL_DIR"
'
echo "[$(date +%T)] model download done"
