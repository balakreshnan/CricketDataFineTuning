#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
# Download the draft models in $DRAFT_IDS to $MODELS_DIR/<name> on Lustre and print the facts that decide
# whether each can serve as a draft: vocabulary size, layers, width, parameter count.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/specdec/specdec.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export DRAFT_IDS
echo "[$(date +%T)] drafts: $DRAFT_IDS -> $MODELS_DIR"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN,DRAFT_IDS bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/specdec/specdec.env
  source "$VENV/bin/activate"
  export MODELS_DIR TARGET_DIR=$MODEL_DIR
  python '"$CODE_DIR"'/specdec/download_drafts.py
  du -sh $(for m in $DRAFT_IDS; do echo "$MODELS_DIR/${m##*/}"; done)
'
echo "[$(date +%T)] drafts done"
