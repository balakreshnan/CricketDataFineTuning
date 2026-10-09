#!/bin/bash
# One-off, on the login node (run under nohup): tools venv, model download, arm64 container imports.
#   nohup bash ~/clustercodes/olympics/setup_login.sh > $LOG_DIR/olympics/setup_login.log 2>&1 &
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
mkdir -p "$PROJ" "$SLURM_LOGS" "$CONTAINER_DIR" "$LUSTRE_DIR/venvs"

if [[ ! -x $TOOLS_VENV/bin/python ]]; then
  python3 -m venv "$TOOLS_VENV"
  "$TOOLS_VENV/bin/pip" install -q --upgrade pip
  "$TOOLS_VENV/bin/pip" install -q pandas pyarrow huggingface_hub datasets requests
fi
echo "[$(date +%T)] venv ok"

"$TOOLS_VENV/bin/hf" download "$MODEL_ID" --max-workers 16 | tail -1
echo "[$(date +%T)] model ok"

# login /tmp is a 2 GB tmpfs: unpack on Lustre
export ENROOT_TEMP_PATH=$LUSTRE_DIR/cache/enroot-tmp ENROOT_CACHE_PATH=$LUSTRE_DIR/cache/enroot-cache
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH"
for pair in "$DYNAMO_IMAGE_REMOTE $DYNAMO_SQSH" "$NEMO_IMAGE_REMOTE $NEMO_SQSH"; do
  set -- $pair
  if [[ ! -f $2 ]]; then
    echo "[$(date +%T)] importing $1"
    enroot import --arch aarch64 -o "$2.part" "docker://$1" && mv "$2.part" "$2"
  fi
  ls -lh "$2"
done
echo "[$(date +%T)] setup done"
