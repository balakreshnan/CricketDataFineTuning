#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -J general_sa-nemo-rl.probe
# Does the stock NeMo RL software stack run on Vera Rubin?
#   1. clone NeMo RL (with submodules) to Lustre
#   2. build its locked environment with the vLLM extra (torch 2.14 cu130, vLLM 0.33) with uv, on Lustre
#   3. run GPU checks in that environment, and the same checks in the Rubin PyTorch image for comparison
# Export STEPS to run a subset, e.g. STEPS="check".
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemorl/nemorl.env"
export STEPS="${STEPS:-clone sync check}" EXTRA="${EXTRA:-vllm}" TRITON_ARCH="${TRITON_ARCH:-sm80}"
mkdir -p "$NRL_ROOT" "$NRL_VENVS"
echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname -s); steps: $STEPS; extra: $EXTRA"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root bash -c '
  source '"$CODE_DIR"'/nemorl/nemorl.env
  UV=$GYM_VENV/bin/uv
  export UV_PROJECT_ENVIRONMENT=$NRL_VENVS/probe-$EXTRA UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
  unset PIP_CONSTRAINT
  # Build settings taken from NeMo RL docker/Dockerfile: kernels built from source (DeepEP, mamba-ssm, flash-attn,
  # Transformer Engine) are compiled for sm_90 and sm_100 (Rubin runs sm_100 code), against the pip cuDNN.
  SP=$UV_PROJECT_ENVIRONMENT/lib/python3.13/site-packages
  export TORCH_CUDA_ARCH_LIST="9.0 10.0" CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
  export CUDNN_HOME=$SP/nvidia/cudnn CUDNN_PATH=$SP/nvidia/cudnn LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
  export MAX_JOBS=64 NVTE_BUILD_MAX_JOBS=32
  if [[ " $STEPS " == *" clone "* ]]; then
    echo "### clone"
    if [[ ! -d $NRL_REPO/.git ]]; then
      git clone --depth 1 --branch "$NRL_REF" --recurse-submodules --shallow-submodules https://github.com/NVIDIA-NeMo/RL.git "$NRL_REPO" 2>&1 | tail -6
    fi
    cd "$NRL_REPO" && git log -1 --format="%H %cd %s" && git submodule status | cut -c1-120 && du -sh . | cut -f1
  fi
  cd "$NRL_REPO" || exit 1
  if [[ " $STEPS " == *" sync "* ]]; then
    echo "### uv sync --locked --extra $EXTRA  ($(date +%T))"
    $UV --version
    $UV sync --locked --extra $EXTRA --no-dev > "$NRL_ROOT/sync-$EXTRA-$SLURM_JOB_ID.log" 2>&1
    echo "uv sync exit code $?  ($(date +%T))"
    grep -n -i -E "error|failed|no matching|not supported|unsatisf" "$NRL_ROOT/sync-$EXTRA-$SLURM_JOB_ID.log" | head -40 | cut -c1-400
    tail -15 "$NRL_ROOT/sync-$EXTRA-$SLURM_JOB_ID.log" | cut -c1-400
  fi
  if [[ " $STEPS " == *" check "* ]]; then
    echo "### checks in the NeMo RL environment ($UV_PROJECT_ENVIRONMENT)"
    if [[ -x $UV_PROJECT_ENVIRONMENT/bin/python ]]; then
      $UV pip list --python $UV_PROJECT_ENVIRONMENT/bin/python 2>/dev/null | grep -i -E "^(torch|triton|vllm|transformers|flashinfer|ray|nemo|peft|nvidia-cudnn|mamba|causal)" | tr "\n" ";"; echo
      # The stock Triton does not know Rubin (sm_107a), and its sm_100 code path emits Blackwell-only instructions.
      # nemorl/rubinfix makes it compile generic sm_80 code, which Rubin runs.
      PYTHONPATH='"$CODE_DIR"'/nemorl/rubinfix $UV_PROJECT_ENVIRONMENT/bin/python '"$CODE_DIR"'/nemorl/rubin_check.py "$MODEL_DIR" > "$NRL_ROOT/check-$EXTRA-$SLURM_JOB_ID.log" 2>&1; echo "check exit code $?"
      grep -a -E "^(python|torch|gpu|PASS|FAIL|     |rubinfix)" "$NRL_ROOT/check-$EXTRA-$SLURM_JOB_ID.log" | cut -c1-700; echo "--- last lines"; tail -6 "$NRL_ROOT/check-$EXTRA-$SLURM_JOB_ID.log" | cut -c1-500
    else
      echo "no environment was built"
    fi
    echo "### same checks in the Rubin PyTorch image (the one rlvr.py uses)"
    source "$VENV/bin/activate"
    [[ $EXTRA == vllm ]] && python '"$CODE_DIR"'/nemorl/rubin_check.py "$MODEL_DIR" 2>&1 | grep -a -E "^(python|torch|gpu|PASS|FAIL|     )" | cut -c1-700
  fi
'
echo "[$(date +%T)] probe done"
