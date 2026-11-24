#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J general_sa-nemotron-rhea.images
# Find out which container images work on Rhea's Vera Rubin nodes for this project:
#   1. the PyTorch image already on Lustre (training)       -> GPU smoke test
#   2. the Rubin build of vLLM (serving for NeMo Gym eval)  -> import to Lustre if missing, GPU + Triton smoke test
#   3. NeMo Gym                                             -> can pip install nemo-gym into the PyTorch image's Python?
# Nothing here is fatal: every check prints PASS or FAIL so that one job answers all three.
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemotron/nemotron.env"
mkdir -p "$CONTAINER_DIR" "$NT_LOGS"
echo "[$(date +%T)] node $(hostname -s) $(uname -m); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr '\n' ';')"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

probe='
import platform, sys, torch
print("python", sys.version.split()[0], platform.machine(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), "arch list", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().abs().mean()) > 0)
y = torch.nn.functional.scaled_dot_product_attention(x[None, None, :64, :64], x[None, None, :64, :64], x[None, None, :64, :64])
print("sdpa ok", tuple(y.shape))
try:
    import triton, triton.language as tl
    @triton.jit
    def _k(p, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(p + o, tl.load(p + o) + 1.0)
    t = torch.zeros(64, device="cuda")
    _k[(1,)](t, 64)
    print("triton", triton.__version__, "kernel ok", float(t.sum()) == 64.0)
except Exception as e:
    print("triton FAIL", type(e).__name__, str(e)[:200])
for m in ("vllm", "transformers", "peft", "flashinfer"):
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not importable:", type(e).__name__)
'
run_in() {  # run_in <sqsh> <label>
  echo "=== $2: $1"
  srun --container-image="$1" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --no-container-remap-root \
       env TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID python3 -c "$probe" 2>&1 | grep -v -E "Warning|warn" | tail -14 \
    && echo "RESULT $2: PASS (see lines above for any FAIL)" || echo "RESULT $2: FAIL"
}

run_in "$IMAGE_SQSH" "pytorch image (training)"

if [[ ! -f "$VLLM_SQSH" ]]; then
  echo "[$(date +%T)] importing $VLLM_IMAGE_REMOTE -> $VLLM_SQSH"
  export ENROOT_CACHE_PATH=$LUSTRE_DIR/cache/enroot ENROOT_DATA_PATH=/tmp/enroot-data-$SLURM_JOB_ID
  mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
  enroot import -o "$VLLM_SQSH.part" "docker://${VLLM_IMAGE_REMOTE/\//#}" 2>&1 | grep -v "failed to set capabilities" | tail -8
  [[ -f "$VLLM_SQSH.part" ]] && mv "$VLLM_SQSH.part" "$VLLM_SQSH"
fi
if [[ -f "$VLLM_SQSH" ]]; then
  ls -lh "$VLLM_SQSH"
  run_in "$VLLM_SQSH" "vllm rubin image (serving)"
  srun --container-image="$VLLM_SQSH" --no-container-remap-root bash -c 'vllm --version 2>&1 | tail -1; python3 -c "import vllm.model_executor.models.registry as r; ms=[k for k in r.ModelRegistry.get_supported_archs() if \"emotron\" in k or \"Nemotron\" in k]; print(\"vllm nemotron archs:\", ms)" 2>&1 | tail -1'
else
  echo "RESULT vllm rubin image (serving): FAIL (import produced no file; no arm64 build or no access)"
fi

echo "=== NeMo Gym $GYM_VERSION"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --no-container-remap-root bash -c '
  python3 --version
  python3 -m pip download --no-deps -d /tmp/gymdl nemo-gym=='"$GYM_VERSION"' 2>&1 | tail -3
  ls /tmp/gymdl 2>/dev/null
  python3 -m pip install --dry-run nemo-gym=='"$GYM_VERSION"' 2>&1 | tail -4
' && echo "RESULT nemo-gym pip: see lines above" || echo "RESULT nemo-gym pip: FAIL"
echo "[$(date +%T)] images probe done"
