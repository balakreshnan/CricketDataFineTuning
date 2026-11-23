#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -J general_sa-cricket-rhea.specdec-profile
# Diagnostic: per-token cost of a hand-written decode loop vs generate() for the target model (profile_loop.py).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/specdec/specdec.env"
cd "$RUN_FROM"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 EXTRA_ENV=${EXTRA_ENV:-} NPROC=${NPROC:-1}
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/specdec/specdec.env
  source "$VENV/bin/activate"
  export MODEL_DIR DATA_LOCAL TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID $EXTRA_ENV
  for i in $(seq 0 $(( ${NPROC:-1} - 1 ))); do
    CUDA_VISIBLE_DEVICES=$i python '"$CODE_DIR"'/specdec/profile_loop.py 2>&1 | grep --line-buffered -E "tok/s|pieces|threads|Error" | sed -u "s/^/[gpu$i] /" &
  done
  wait
'
