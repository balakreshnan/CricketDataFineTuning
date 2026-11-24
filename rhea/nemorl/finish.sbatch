#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH -J general_sa-nemo-rl.finish
# After training: (1) check that the adapter works with plain PEFT (apply and merge), (2) if PUSH=1, upload the
# adapter only (not merged) to the PUBLIC Hugging Face repo $REPO with the model card from push_adapter.py.
# Export: RUN_NAME (required), PUSH (0), REPO, RESULTS (json file for the model card)
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemorl/nemorl.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export RUN_NAME PUSH=${PUSH:-0} REPO=${REPO:-Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-NeMo-RL-Rhea} RESULTS=${RESULTS:-}
export ADAPTER=$NT_DIR/runs/$RUN_NAME/adapter NRL_DATA=$NRL_ROOT/data OUT=$NRL_ROOT/runs/$RUN_NAME
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-env=HF_TOKEN bash -c '
  set -uo pipefail
  source '"$CODE_DIR"'/nemotron/nemotron.env
  source "$VENV/bin/activate"
  export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
  HF_HUB_OFFLINE=1 python '"$CODE_DIR"'/nemorl/peft_check.py '"$CODE_DIR"' "$MODEL_DIR" "$ADAPTER" "$NRL_DATA/validation.jsonl" "$OUT/peft_check.json" 2>&1 | grep -v -E "falling back|Warning|warn"
  rc=${PIPESTATUS[0]}; echo "[$(date +%T)] peft check exit code $rc"
  if [[ $rc == 0 && $PUSH == 1 ]]; then
    python '"$CODE_DIR"'/nemorl/push_adapter.py "$ADAPTER" "$REPO" ${RESULTS:+--results "$RESULTS"}
    echo "[$(date +%T)] push exit code $?"
  fi
'
echo "[$(date +%T)] finish job done"
