#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J general_sa-olympics.prep-rows
# Build the 1:1 row questions on a compute node (the login node kills long CPU-heavy processes).
#   sbatch -p tcpo rows.sbatch            ARGS="--fallback" for the fallback pass
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
export PROJ RAW_DIR ARGS=${ARGS:-}
srun --container-image="$NEMO_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
cd /home/bbalakreshna/clustercodes/olympics && time /opt/venv/bin/python prep_rows.py $ARGS'
