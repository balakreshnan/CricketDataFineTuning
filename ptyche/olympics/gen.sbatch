#!/bin/bash
#SBATCH -A general_sa
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -J general_sa-olympics.datagen
# Reasoning-trace generation with NVIDIA Dynamo (SGLang backend) hosting Nemotron-3.5-Lightning.
# Every node is a self-contained Dynamo deployment: one frontend (OpenAI API on :8000, KV-aware router) and one
# SGLang worker per GPU (TP1, the 30B-A3B model fits one 189 GB GB200 many times over), discovered through a
# node-local file store, so no etcd/NATS is needed. gen_client.py then samples K traces per question for this
# node's shard of the test and train questions.
#   sbatch -N <nodes> -t 04:00:00 -p tcpo gen.sbatch
#   env: K (4), CONCURRENCY (512 per node), MAX_TOKENS (16384), LIMIT (0 = all; smoke tests), TAG (output dir name), SPLITS ("test train"), MODEL_DIR (default: base)
set -euo pipefail
source /home/bbalakreshna/clustercodes/olympics/config.env
export MODEL_DIR=${MODEL_DIR:-$HF_HOME/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION}   # or a fine-tuned export
export MAX_TOKENS=${MAX_TOKENS:-16384} CONCURRENCY=${CONCURRENCY:-512} K=${K:-4} LIMIT=${LIMIT:-0} TAG=${TAG:-full} SPLITS=${SPLITS:-"test train"} GPUS_PER_NODE ROUTER_MODE=${ROUTER_MODE:-kv}
export OUT=$PROJ/gen/$TAG PROJ
mkdir -p "$OUT/logs"
echo "[$(date +%T)] job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ($SLURM_NNODES nodes) K=$K LIMIT=$LIMIT out=$OUT"

srun --container-image="$DYNAMO_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" \
     --container-mount-home --no-container-remap-root --container-workdir="$RUN_FROM" bash -c '
set -euo pipefail
CODE_DIR=/home/bbalakreshna/clustercodes/olympics          # ~/.bashrc (env.sh) may reset it in the container
node=$SLURM_NODEID; log=$OUT/logs/node$node
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/dynamo-kv-$SLURM_JOB_ID DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_NAMESPACE=olympics HF_HUB_OFFLINE=1 DYN_SELF_HOST_METADATA=0
# node-local compile caches (Lustre caches race across processes)
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-$SLURM_JOB_ID

python -m dynamo.frontend --http-port 8000 --router-mode $ROUTER_MODE > $log.frontend.log 2>&1 &
pids=$!
for g in $(seq 0 $((GPUS_PER_NODE - 1))); do
  # model-card recipe for Blackwell (flashinfer Mamba kernels, fp16 SSM state with stochastic rounding),
  # with a bigger decode CUDA graph since we batch hundreds of requests per GPU
  CUDA_VISIBLE_DEVICES=$g python -m dynamo.sglang \
    --model-path "$MODEL_DIR" --served-model-name nemotron-3.5-lightning --tp-size 1 \
    --port $((30000 + 100 * g)) --nccl-port $((31000 + 100 * g)) \
    --context-length 32768 --mem-fraction-static 0.85 \
    --mamba-backend flashinfer --mamba-ssm-dtype float16 \
    --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
    --cuda-graph-max-bs-decode 256 \
    --dyn-reasoning-parser nemotron_v3 \
    > $log.worker$g.log 2>&1 &
  pids="$pids $!"
done

# ready = the frontend lists the model (first worker registered); then give the others time to finish loading
for i in $(seq 180); do
  curl -sf localhost:8000/v1/models | grep -q nemotron && break
  sleep 10
done
curl -sf localhost:8000/v1/models | grep -q nemotron || { echo "node $node: model never registered"; tail -30 $log.worker0.log; exit 1; }
sleep 90
echo "[node $node $(date +%T)] Dynamo up after ~$((i * 10 + 90))s"

for split in $SPLITS; do
  python $CODE_DIR/gen_client.py --questions $PROJ/data/questions_$split.jsonl --out $OUT/$split.shard$node.jsonl \
    --shard $node/$SLURM_NNODES --k $K --limit $LIMIT --concurrency $CONCURRENCY --max-tokens $MAX_TOKENS 2>&1 | tee $log.client-$split.log
done
echo "[node $node $(date +%T)] done"
kill $pids 2>/dev/null || true   # by PID: pkill -f would also match this bash -c, whose text names them
'
echo "[$(date +%T)] datagen finished"
