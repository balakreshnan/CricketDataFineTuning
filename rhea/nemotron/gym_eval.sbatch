#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -J general_sa-nemotron-rhea.gymeval
# NeMo Gym evaluation of the base model and the model with the RLVR adapter in the cricket environment.
#   1. vLLM (Rubin image) serves the base model and, as a LoRA module, the adapter of run $RUN_NAME, on GPU 0
#   2. NeMo Gym (venv on Lustre) runs the cricket_qa environment against it: $REPEATS answers per question
#   3. gym_summary.py writes per-question-type summaries and logs them to W&B
# GPU utilisation, memory and power are logged every 15 s to $OUT/gpulog.
# Export: RUN_NAME (required), LIMIT (0 = all 600 questions), REPEATS (4), CONC (64), MAXTOK (4096), TAGS ("base tuned")
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemotron/nemotron.env"
if [[ -f "$SECRETS_FILE" ]]; then set -a; source "$SECRETS_FILE"; set +a; fi
export RUN_NAME LIMIT=${LIMIT:-0} REPEATS=${REPEATS:-4} CONC=${CONC:-64} MAXTOK=${MAXTOK:-4096} TAGS="${TAGS:-base tuned}"
export ADAPTER=$NT_DIR/runs/$RUN_NAME/adapter OUT=$NT_DIR/runs/$RUN_NAME/gym GYM_WORK=$NT_DIR/gym PORT=8000
export WANDB_PROJECT=$NT_WANDB_PROJECT GYM_RUN_NAME=$RUN_NAME-nemo-gym
mkdir -p "$OUT/gpulog" "$GYM_WORK"
echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname -s): NeMo Gym eval of $RUN_NAME; adapter $ADAPTER; out $OUT"
srun --overlap bash -c 'exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 15 > "'"$OUT"'/gpulog/$(hostname -s)-'"$SLURM_JOB_ID"'.csv"' >/dev/null 2>&1 &
GPULOG_PID=$!

# ---- 1. vLLM ----
lora=""; [[ " $TAGS " == *" tuned "* ]] && lora="--enable-lora --max-lora-rank 64 --lora-modules tuned=$ADAPTER"
srun --overlap --container-image="$VLLM_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --no-container-remap-root bash -c '
  export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HOME='"$HF_HOME"' TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID VLLM_CACHE_ROOT=/tmp/vllm-$SLURM_JOB_ID
  exec vllm serve '"$MODEL_DIR"' --served-model-name base --host 127.0.0.1 --port '"$PORT"' --api-key dummy \
    --max-model-len 16384 --max-num-seqs 128 --enable-prefix-caching --async-scheduling \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
    --reasoning-parser nemotron_v3 '"$lora"' '"${VLLM_EXTRA:-}"'
' > "$OUT/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
ok=0
for i in $(seq 1 120); do
  if curl -sf -H "Authorization: Bearer dummy" "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" 2>/dev/null; then ok=1; break; fi
  kill -0 $VLLM_PID 2>/dev/null || break
  sleep 10
done
if [[ $ok != 1 ]]; then
  echo "[$(date +%T)] vLLM did not come up; last lines of its log:"; grep -v -i "warn" "$OUT/vllm-$SLURM_JOB_ID.log" | tail -40
  kill $GPULOG_PID $VLLM_PID 2>/dev/null; exit 1
fi
echo "[$(date +%T)] vLLM is up: $(python3 -c "import json;print([m['id'] for m in json.load(open('$OUT/models.json'))['data']])")"

# ---- 2. NeMo Gym ----
srun --overlap --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home --no-container-remap-root bash -c '
  set -uo pipefail
  source '"$CODE_DIR"'/nemotron/nemotron.env
  export UV_PYTHON_INSTALL_DIR=$LUSTRE_DIR/cache/uv-python UV_CACHE_DIR=$LUSTRE_DIR/cache/uv
  source "$GYM_VENV/bin/activate"
  python '"$CODE_DIR"'/nemotron/gym_prepare.py '"$CODE_DIR"' "$DATA_LOCAL" "$GYM_WORK"
  cd "$GYM_WORK"
  lim=""; [[ $LIMIT != 0 ]] && lim="--limit $LIMIT"
  for tag in $TAGS; do
    [[ -s $OUT/$tag.jsonl && $(wc -l < $OUT/$tag.jsonl) -ge $(( (LIMIT == 0 ? 600 : LIMIT) * REPEATS )) ]] && { echo "$tag already collected"; continue; }
    rm -rf "$OUT/$tag"   # NeMo Gym keeps prepared data and its statistics here; stale ones stop the run
    echo "[$(date +%T)] gym eval run: $tag"
    gym eval run --resources-server cricket_qa --model-type vllm_model --model-url http://127.0.0.1:$PORT/v1 --model $tag --model-api-key dummy \
      --agent cricket_qa_simple_agent --split validation --output "$OUT/$tag.jsonl" \
      --num-repeats $REPEATS --concurrency $CONC --temperature 1.0 --top-p 0.95 --max-output-tokens $MAXTOK $lim > "$OUT/gym-$tag-$SLURM_JOB_ID.log" 2>&1
    echo "gym exit code $?"; grep -n -E "Error|error|Traceback|Exception|mean/reward|pass@|accuracy" "$OUT/gym-$tag-$SLURM_JOB_ID.log" | tail -25 | cut -c1-300; tail -12 "$OUT/gym-$tag-$SLURM_JOB_ID.log" | cut -c1-300
    ls -la "$OUT" | grep "$tag"
  done
'
kill $VLLM_PID 2>/dev/null

# ---- 3. summary and W&B (training venv has wandb) ----
srun --overlap --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home --no-container-remap-root \
     --container-env=WANDB_API_KEY,WANDB_ENTITY,WANDB_PROJECT,WANDB_DIR,GYM_RUN_NAME bash -c '
  source '"$CODE_DIR"'/nemotron/nemotron.env
  source "$VENV/bin/activate"
  export WANDB_PROJECT=$NT_WANDB_PROJECT
  have=""; for tag in $TAGS; do [[ -s $OUT/$tag.jsonl ]] && have="$have $tag"; done
  [[ -n $have ]] && python '"$CODE_DIR"'/nemotron/gym_summary.py "$OUT" $have
'
kill $GPULOG_PID 2>/dev/null
echo "[$(date +%T)] gym eval done"
