# Environment for every NeMo RL process on Rhea. Sourced inside the container on each node (driver, Ray head, Ray
# workers), so all of them see the same Python environments, build settings and Vera Rubin fixes.
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemorl/nemorl.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export MODEL_DIR NRL_REPO
UV=$GYM_VENV/bin/uv
unset PIP_CONSTRAINT VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=$NRL_VENVS/driver UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
export PATH=$UV_PROJECT_ENVIRONMENT/bin:$GYM_VENV/bin:$PATH     # ray, and uv (NeMo RL calls uv to build worker environments)
export NEMO_GYM_VENV_DIR=$NRL_VENVS/gym
# Build settings from NeMo RL docker/Dockerfile (kernels built from source target sm_90 and sm_100).
export TORCH_CUDA_ARCH_LIST="9.0 10.0" CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export MAX_JOBS=64 NVTE_BUILD_MAX_JOBS=32
# Vera Rubin: the stock Triton cannot target sm_107a; nemorl/rubinfix patches it at interpreter start-up.
export PYTHONPATH=$CODE_DIR/nemorl/rubinfix${PYTHONPATH:+:$PYTHONPATH}
# Transformer Engine must load the pip cuDNN of the worker environment, not the one of the container.
for v in "$NEMO_RL_VENV_DIR"/*AutomodelPolicyWorker* "$NEMO_RL_VENV_DIR"/*Vllm* "$UV_PROJECT_ENVIRONMENT"; do
  c=$v/lib/python3.13/site-packages/nvidia/cudnn
  if [[ -d $c/lib ]]; then
    export LD_LIBRARY_PATH=$c/lib:${LD_LIBRARY_PATH:-}
    export CUDNN_HOME=${CUDNN_HOME:-$c} CUDNN_PATH=${CUDNN_PATH:-$c}
  fi
done
export HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$NRL_ROOT/hf-datasets RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export WANDB_PROJECT=nemotron35-cricket-nemo-rl-rhea
mkdir -p "$HF_DATASETS_CACHE"
