# Rhea environment: sourced at the top of every job script (and from ~/.bashrc).
# Rule: code runs from $CODE_DIR; everything large (weights, HF cache, datasets, ckpts, logs, caches) lives on Lustre.
# EDIT LUSTRE_DIR/CODE_DIR if `discover.sh` shows different paths.
export LUSTRE_DIR=${LUSTRE_DIR:-/lustre/fsw/general_sa/bbalakreshna}
export CODE_DIR=${CODE_DIR:-/home/bbalakreshna/clustercodes}

export MODELS_DIR=$LUSTRE_DIR/models
export DATA_DIR=$LUSTRE_DIR/datasets
export CKPT_DIR=$LUSTRE_DIR/ckpts
export LOG_DIR=$LUSTRE_DIR/logs
export CONTAINER_DIR=$LUSTRE_DIR/containers
export VENV_DIR=$LUSTRE_DIR/venvs

export HF_HOME=$LUSTRE_DIR/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$LUSTRE_DIR/cache/torch
export TRITON_CACHE_DIR=$LUSTRE_DIR/cache/triton
export PIP_CACHE_DIR=$LUSTRE_DIR/cache/pip
export UV_CACHE_DIR=$LUSTRE_DIR/cache/uv
export XDG_CACHE_HOME=$LUSTRE_DIR/cache/xdg
export TMPDIR=${TMPDIR:-/tmp}
export WANDB_DIR=$LOG_DIR
export WANDB_CACHE_DIR=$LUSTRE_DIR/cache/wandb
export WANDB_DATA_DIR=$LUSTRE_DIR/cache/wandb-data
export VLLM_CACHE_ROOT=$LUSTRE_DIR/cache/vllm
