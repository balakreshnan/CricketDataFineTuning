#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -J general_sa-cricket-rhea.rlvr-finish
# Log the before/after benchmark to the RLVR run's W&B page and push the adapter to $RLVR_HF_REPO (private).
# submit.sh exports RUN_NAME and ARGS (extra rlvr_finish.py flags, e.g. --no_push).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export WANDB_PROJECT=$RLVR_WANDB_PROJECT RUN_DIR=$PROJ/runs/$RUN_NAME ARGS RLVR_HF_REPO
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN,WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_DIR bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/config.env
  source "$VENV/bin/activate"
  export WANDB_PROJECT=$RLVR_WANDB_PROJECT    # config.env just reset it to the supervised project
  exec python '"$CODE_DIR"'/rlvr_finish.py --run_dir "$RUN_DIR" --repo_id "$RLVR_HF_REPO" --base_model_id "$MODEL_ID" \
    --dataset_id "$DATASET_ID" $ARGS
'
echo "[$(date +%T)] rlvr-finish done"
