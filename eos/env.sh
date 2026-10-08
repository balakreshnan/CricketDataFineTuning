# EOS environment: sourced from ~/.bashrc and at the top of every job script.
# Rule: code lives and runs from $HOME/clustercodes; everything large (weights, HF cache,
# datasets, checkpoints, logs, containers, pip/torch caches) lives on Lustre.
export LUSTRE_DIR=/lustre/fsw/general_sa/bbalakreshna
export CODE_DIR=$HOME/clustercodes

export MODELS_DIR=$LUSTRE_DIR/models
export DATA_DIR=$LUSTRE_DIR/datasets
export CKPT_DIR=$LUSTRE_DIR/ckpts
export LOG_DIR=$LUSTRE_DIR/logs
export CONTAINER_DIR=$LUSTRE_DIR/containers

export HF_HOME=$LUSTRE_DIR/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export TORCH_HOME=$LUSTRE_DIR/cache/torch
export TRITON_CACHE_DIR=$LUSTRE_DIR/cache/triton
export PIP_CACHE_DIR=$LUSTRE_DIR/cache/pip
export XDG_CACHE_HOME=$LUSTRE_DIR/cache/xdg
export WANDB_DIR=$LOG_DIR
export VLLM_CACHE_ROOT=$LUSTRE_DIR/cache/vllm
