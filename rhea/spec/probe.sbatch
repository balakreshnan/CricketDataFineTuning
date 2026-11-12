#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J general_sa-cricket-rhea.specprobe
# Probe: can Dynamo's SGLang runtime serve Qwen3.8-27B on Rhea (Vera Rubin, aarch64), and does speculative decoding work?
#   A) import the Dynamo SGLang image to Lustre
#   B) inside it: GPU support, sglang/dynamo versions, speculative flags
#   C) three plain SGLang servers on GPUs 0-2 (no speculation / MTP head (NEXTN) / n-gram), same small benchmark each
#   D) Dynamo frontend + dynamo.sglang worker on GPU 3
# Nothing here is fatal: every stage logs what happened and the job carries on.
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
export DYNAMO_SQSH=$CONTAINER_DIR/dynamo-sglang-runtime-1.5.1-arm64.sqsh
export OUT=$PROJ/spec/probe-$SLURM_JOB_ID RUBINFIX=${RUBINFIX:-1}
mkdir -p "$OUT" "$LUSTRE_DIR/cache/enroot-cache"
echo "[$(date +%T)] node $(hostname) out $OUT"

if [[ ! -f $DYNAMO_SQSH ]]; then
  export ENROOT_TEMP_PATH=/raid/enroot-tmp-$SLURM_JOB_ID ENROOT_CACHE_PATH=$LUSTRE_DIR/cache/enroot-cache
  mkdir -p "$ENROOT_TEMP_PATH" 2>/dev/null || { export ENROOT_TEMP_PATH=/tmp/enroot-tmp-$SLURM_JOB_ID; mkdir -p "$ENROOT_TEMP_PATH"; }
  echo "[$(date +%T)] importing nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
  enroot import -o "$DYNAMO_SQSH.part" "docker://nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.5.1" 2>&1 | grep -v "failed to set capabilities" | tail -5
  if [[ ! -f $DYNAMO_SQSH.part ]]; then
    # the stored NGC key was rejected (401); this image is public, so retry without credentials
    echo "[$(date +%T)] retrying anonymously"
    mkdir -p /tmp/enroot-nocreds-$SLURM_JOB_ID
    ENROOT_CONFIG_PATH=/tmp/enroot-nocreds-$SLURM_JOB_ID enroot import -o "$DYNAMO_SQSH.part" "docker://nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.5.1" 2>&1 | grep -v "failed to set capabilities" | tail -5
  fi
  [[ -f $DYNAMO_SQSH.part ]] && mv "$DYNAMO_SQSH.part" "$DYNAMO_SQSH"
  rm -rf "$ENROOT_TEMP_PATH"
fi
ls -lh "$DYNAMO_SQSH" || { echo "IMAGE IMPORT FAILED"; exit 1; }

srun --container-image="$DYNAMO_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root bash -c '
CODE=/home/bbalakreshna/clustercodes
EVAL='"$DATA_LOCAL"'/data/eval-00000.jsonl
MODEL='"$MODEL_DIR"'
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-$SLURM_JOB_ID
if [[ "${RUBINFIX:-1}" == 1 ]]; then   # compile Triton / FlashInfer kernels for sm_100a (see spec/rubinfix/sitecustomize.py)
  export PYTHONPATH=$CODE/spec/rubinfix:${PYTHONPATH:-} FLASHINFER_CUDA_ARCH_LIST=10.0
  python -c "import triton.backends.nvidia.compiler as c; print(\"rubinfix: triton arch for capability 107 ->\", c.sm_arch_from_capability(107))"
  grep -n -E "sm_arch_from_capability|def make_ptx|gpu-name|mcpu|proc =" /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/compiler.py | head -12
fi
echo "===== B) environment"
uname -m; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch list", torch.cuda.get_arch_list())
try:
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print("gpu matmul ok", float((x @ x).float().abs().mean()) > 0, torch.cuda.get_device_capability(0))
except Exception as e:
    print("GPU OP FAILED:", repr(e)[:300])
import sglang; print("sglang", sglang.__version__)
PY
pip list 2>/dev/null | grep -i -E "^(ai-dynamo|ai-dynamo-runtime|sglang|flashinfer-python|transformers|torch|triton) "
echo "--- sglang speculative flags"; python -m sglang.launch_server --help 2>&1 | grep -i -A3 -E "^\s+--speculative" | grep -v "^--$" | head -60
echo "--- SpeculativeAlgorithm values"; f=$(python -c "import sglang.srt.speculative.spec_info as m; print(m.__file__)" 2>/dev/null); grep -n -E "^\s+[A-Z_0-9]+ = |class Spec" "$f" | head -20
echo "--- qwen3_5 / mtp support in sglang"; d=$(python -c "import sglang,os; print(os.path.dirname(sglang.__file__))"); grep -rl -i -E "qwen3_5|Qwen3_5" $d --include=*.py | head -8; grep -rl -i "mtp" $d/srt/models 2>/dev/null | head -8
echo "--- dynamo.sglang speculative passthrough"; python -m dynamo.sglang --help 2>&1 | grep -i -c specul

start() {  # name gpu port extra-args...
  local name=$1 gpu=$2 port=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu python -m sglang.launch_server --model-path "$MODEL" --served-model-name qwen38 \
    --host 127.0.0.1 --port $port --context-length 16384 --mem-fraction-static 0.8 "$@" > $OUT/server-$name.log 2>&1 &
  echo $! > $OUT/pid-$name
}
wait_up() {  # name port -> 0 if healthy within 15 min
  for i in $(seq 90); do
    curl -sf -m 5 localhost:$2/v1/models >/dev/null && return 0
    kill -0 $(cat $OUT/pid-$1) 2>/dev/null || return 1
    sleep 10
  done; return 1
}
echo "===== C) plain SGLang servers"
start base  0 30000
start mtp   1 30100 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
start ngram 2 30200 --speculative-algorithm NGRAM --speculative-num-draft-tokens 8
for spec in "base 30000" "mtp 30100" "ngram 30200"; do
  set -- $spec
  if wait_up $1 $2; then
    echo "[$(date +%T)] $1 server UP"
    python $CODE/spec/bench_client.py http://localhost:$2 qwen38 $EVAL $1 32 32 2048 | tee -a $OUT/bench.jsonl
    grep -i -E "accept|spec" $OUT/server-$1.log | tail -3 | cut -c1-300
  else
    echo "[$(date +%T)] $1 server FAILED TO START; log tail:"; grep -v -E "^\s*$" $OUT/server-$1.log | tail -25 | cut -c1-300
  fi
  kill $(cat $OUT/pid-$1) 2>/dev/null
done
sleep 5
echo "===== D) Dynamo frontend + dynamo.sglang worker"
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/dynamo-kv-$SLURM_JOB_ID DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq DYN_NAMESPACE=cricket DYN_SELF_HOST_METADATA=0
python -m dynamo.frontend --http-port 8000 > $OUT/dynamo-frontend.log 2>&1 & fp=$!
CUDA_VISIBLE_DEVICES=3 python -m dynamo.sglang --model-path "$MODEL" --served-model-name qwen38 --tp-size 1 \
  --port 30300 --context-length 16384 --mem-fraction-static 0.8 > $OUT/dynamo-worker.log 2>&1 & wp=$!
ok=1; for i in $(seq 90); do curl -sf -m 5 localhost:8000/v1/models | grep -q qwen38 && { ok=0; break; }; kill -0 $wp 2>/dev/null || break; sleep 10; done
if [[ $ok == 0 ]]; then
  sleep 30; echo "[$(date +%T)] Dynamo UP"
  python $CODE/spec/bench_client.py http://localhost:8000 qwen38 $EVAL dynamo-base 32 32 2048 | tee -a $OUT/bench.jsonl
else
  echo "Dynamo worker FAILED; log tail:"; grep -v -E "^\s*$" $OUT/dynamo-worker.log | tail -25 | cut -c1-300
fi
kill $fp $wp 2>/dev/null
echo "[$(date +%T)] probe done"
'
echo "[$(date +%T)] job finished"
