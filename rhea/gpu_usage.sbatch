#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -J general_sa-cricket-rhea.gpu-usage
# Export the GPU system metrics W&B sampled for a run to <run_dir>/gpu_usage.json. submit.sh exports RUN_NAME.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT=$RLVR_WANDB_PROJECT RUN_DIR=$PROJ/runs/$RUN_NAME
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_DIR bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  export WANDB_PROJECT=$RLVR_WANDB_PROJECT
  exec python '"$CODE_DIR"'/export_sysmetrics.py --run_dir "$RUN_DIR"
'
