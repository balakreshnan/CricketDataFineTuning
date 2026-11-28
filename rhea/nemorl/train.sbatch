#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -J general_sa-nemo-rl.train
# NeMo RL GRPO + LoRA on Nemotron 3.5 Lightning 30B with the cricket NeMo Gym environment, on 1 or more nodes
# (4 GPUs each; pass -N to sbatch).
#   1 node : NeMo RL starts its own local Ray cluster.
#   N nodes: this script starts a Ray head on the first node and a Ray worker on every other node (same resources
#            that NeMo RL ray.sub registers), and NeMo RL attaches to that cluster.
# NeMo RL serves the policy with vLLM on the training GPUs and runs NeMo Gym on the first node.
# GPU utilisation, memory, power and temperature are logged every 15 s on every node to <run>/gpulog.
# Export: RUN_NAME (required), NRL_STEPS (3), NRL_PROMPTS (16 per node), NRL_SAVE_PERIOD (10), NRL_WANDB (true), EXTRA_ARGS,
#         NRL_SRC_DATA (cricket dataset) and NRL_DATA (task files) to train on another dataset in the same layout,
#         NRL_SEED (42), NRL_LEN_COEF (0.4; reward of a correct answer = 1 - coef * output_tokens / 4096)
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemorl/nemorl.env"
N=$SLURM_JOB_NUM_NODES
export RUN_NAME NRL_STEPS=${NRL_STEPS:-3} NRL_PROMPTS=${NRL_PROMPTS:-$((16 * N))} NRL_SAVE_PERIOD=${NRL_SAVE_PERIOD:-10}
export NRL_WANDB=${NRL_WANDB:-true} EXTRA_ARGS="${EXTRA_ARGS:-}" NRL_NODES=$N
export NRL_SEED=${NRL_SEED:-42} NRL_LEN_COEF=${NRL_LEN_COEF:-0.4}
export NRL_GYM_CFG=cricket_qa_len$NRL_LEN_COEF.yaml
export NRL_RUN_DIR=$NRL_ROOT/runs/$RUN_NAME NRL_DATA=${NRL_DATA:-$NRL_ROOT/data}
export NRL_SRC_DATA=${NRL_SRC_DATA:-$DATA_LOCAL}   # dataset folder in the cricket layout (data/train-*.jsonl, data/eval-00000.jsonl)
HEAD=$(hostname -s); export HEAD_IP=$(hostname -i | awk "{print \$1}") RAY_PORT=1200
mkdir -p "$NRL_RUN_DIR/gpulog" "$NRL_RUN_DIR/ray" "$NRL_DATA" "$NRL_VENVS"
echo "[$(date +%T)] job $SLURM_JOB_ID, $N node(s) $SLURM_JOB_NODELIST, head $HEAD ($HEAD_IP): NeMo RL run $RUN_NAME, $NRL_STEPS steps x $NRL_PROMPTS questions x 8 answers, seed $NRL_SEED, length coefficient $NRL_LEN_COEF; out $NRL_RUN_DIR"
srun --overlap -N "$N" --ntasks-per-node=1 bash -c 'exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 15 > "'"$NRL_RUN_DIR"'/gpulog/$(hostname -s)-'"$SLURM_JOB_ID"'.csv"' >/dev/null 2>&1 &
GPULOG_PID=$!
CONT=(--container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home --no-container-remap-root)
WORKERS_PID=""
if (( N > 1 )); then
  # Ray workers on the other nodes; they retry until the head is up, then block for the life of the job.
  srun --overlap -N $((N - 1)) -n $((N - 1)) --ntasks-per-node=1 -x "$HEAD" "${CONT[@]}" bash -c '
    source '"$CODE_DIR"'/nemorl/node_env.sh
    rank=$((10#$(hostname -s | tr -dc 0-9)))
    for try in $(seq 1 60); do
      ray stop >/dev/null 2>&1; rm -rf /tmp/ray/session_* 2>/dev/null
      ray start --address "$HEAD_IP:$RAY_PORT" --disable-usage-stats --min-worker-port=2000 --max-worker-port=2999 \
          --resources="{\"slurm_managed_ray_cluster\": 1, \"worker_units\": 4, \"topo_rank\": $rank}" --block
      echo "[$(date +%T)] $(hostname -s): ray worker exited, retry $try"; sleep 10
    done
  ' > "$NRL_RUN_DIR/ray/workers-$SLURM_JOB_ID.log" 2>&1 &
  WORKERS_PID=$!
fi
srun --overlap -N 1 -n 1 -w "$HEAD" "${CONT[@]}" bash -c '
  set -uo pipefail
  source '"$CODE_DIR"'/nemorl/node_env.sh
  export NRL_RUN_DIR NRL_DATA RUN_NAME NRL_STEPS NRL_PROMPTS NRL_SAVE_PERIOD NRL_WANDB NRL_NODES NRL_SEED NRL_LEN_COEF NRL_GYM_CFG NRL_SRC_DATA   # read by cricket_rhea.yaml
  export WANDB_DIR=$NRL_RUN_DIR
  if (( NRL_NODES > 1 )); then
    rank=$((10#$(hostname -s | tr -dc 0-9)))
    ray stop >/dev/null 2>&1; rm -rf /tmp/ray/session_* 2>/dev/null
    ray start --head --node-ip-address="$HEAD_IP" --port=$RAY_PORT --disable-usage-stats --include-dashboard=False \
        --min-worker-port=2000 --max-worker-port=2999 \
        --resources="{\"slurm_managed_ray_cluster\": 1, \"worker_units\": 4, \"topo_rank\": $rank}" > "$NRL_RUN_DIR/ray/head-$SLURM_JOB_ID.log" 2>&1
    echo "[$(date +%T)] ray head exit code $?"
    export RAY_ADDRESS=$HEAD_IP:$RAY_PORT
    for i in $(seq 1 60); do
      n=$(python -c "import ray; ray.init(address=\"$RAY_ADDRESS\", logging_level=\"ERROR\"); print(sum(1 for x in ray.nodes() if x[\"Alive\"]), int(ray.cluster_resources().get(\"GPU\", 0)))" 2>/dev/null | tail -1)
      [[ "${n%% *}" == "$NRL_NODES" ]] && break; sleep 5
    done
    echo "[$(date +%T)] ray cluster: nodes and GPUs = $n (expected $NRL_NODES nodes)"
    [[ "${n%% *}" == "$NRL_NODES" ]] || { echo "ray cluster incomplete"; tail -20 "$NRL_RUN_DIR"/ray/*-$SLURM_JOB_ID.log; exit 1; }
  else
    export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID; mkdir -p "$RAY_TMPDIR"
  fi
  python3 '"$CODE_DIR"'/nemorl/prepare.py '"$CODE_DIR"' "$NRL_SRC_DATA" "$NRL_REPO" "$NRL_DATA"
  cp '"$CODE_DIR"'/nemorl/cricket_rhea.yaml "$NRL_REPO/examples/nemo_gym/cricket_rhea.yaml"
  cd "$NRL_REPO"
  echo "[$(date +%T)] starting NeMo RL"
  $UV run --locked --no-dev python examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/cricket_rhea.yaml $EXTRA_ARGS \
      > "$NRL_RUN_DIR/train-$SLURM_JOB_ID.log" 2>&1
  echo "[$(date +%T)] NeMo RL exit code $?"
  grep -a -n -E "^Traceback|^[A-Za-z.]*Error:|=== Step|Avg Reward|Mean Generation|Total step time|Saving checkpoint for" "$NRL_RUN_DIR/train-$SLURM_JOB_ID.log" | tail -60 | cut -c1-300
  echo "--- last lines"; grep -a -v "^\s*$" "$NRL_RUN_DIR/train-$SLURM_JOB_ID.log" | tail -15 | cut -c1-300
  ls "$NRL_RUN_DIR/checkpoints" 2>/dev/null | head -30
  (( NRL_NODES > 1 )) && ray stop >/dev/null 2>&1
'
[[ -n $WORKERS_PID ]] && kill $WORKERS_PID 2>/dev/null
kill $GPULOG_PID 2>/dev/null
echo "[$(date +%T)] train job done"
