#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J general_sa-nemotron-rhea.genprobe
# Diagnostic: generation speed of Nemotron in the training stack (gen_probe.py).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemotron/nemotron.env"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root bash -c '
  set -euo pipefail
  source '"$CODE_DIR"'/nemotron/nemotron.env
  source "$VENV/bin/activate"
  export MODEL_DIR DATA_LOCAL TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=0
  python '"$CODE_DIR"'/nemotron/gen_probe.py 2>&1 | grep -v -E "falling back|Warning|warn"
'
