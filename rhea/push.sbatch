#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -J general_sa-cricket-rhea.push
# Upload a finished run's LoRA adapter to the Hub. submit.sh exports RUN_NAME and ARGS (extra push.py flags).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export RUN_DIR=$PROJ/runs/$RUN_NAME ARGS HF_MODEL_REPO
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  exec python '"$CODE_DIR"'/push.py --run_dir "$RUN_DIR" --repo_id "$HF_MODEL_REPO" --base_model_id "$MODEL_ID" \
    --dataset_id "$DATASET_ID" $ARGS
'
echo "[$(date +%T)] push done"
