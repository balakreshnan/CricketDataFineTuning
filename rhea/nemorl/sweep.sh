#!/bin/bash
# Seed x length-coefficient sweep of the NeMo RL cricket run (run on the Rhea login node from ~/clustercodes).
#   usage: nemorl/sweep.sh "<coefficients>" "<seeds>" [steps]     e.g. nemorl/sweep.sh "0 0.1 0.4 1.0" "1 2 3" 60
# Per run: a 4-node, 90-minute training job (plus a resume job for coefficients below 0.4, whose long answers make
# steps slow), then a one-node NeMo Gym evaluation of the final adapter (600 questions x 4 answers).
# Optional environment: PREFIX (run-name prefix, default "sweep"), FIRST_DEP (job id the training jobs wait for),
# RESUME=0 (no resume jobs), and NRL_SRC_DATA / NRL_DATA / EVAL_DATA_DIR to use another dataset (exported to the jobs).
# Run names: <prefix>-c<coef>-s<seed>. Job ids are appended to $NRL_ROOT/sweep_jobs.tsv.
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemorl/nemorl.env"
COEFS=$1; SEEDS=$2; STEPS=${3:-60}
mkdir -p "$NRL_LOGS"
export NT_WANDB_PROJECT=nemotron35-cricket-nemo-rl-rhea
for c in $COEFS; do for s in $SEEDS; do
  run=${PREFIX:-sweep}-c$c-s$s
  ex="ALL,RUN_NAME=$run,NRL_STEPS=$STEPS,NRL_SAVE_PERIOD=5,NRL_WANDB=true,NRL_SEED=$s,NRL_LEN_COEF=$c"
  j=$(sbatch --parsable -N 4 -t 1:30:00 ${FIRST_DEP:+-d afterok:$FIRST_DEP} -o "$NRL_LOGS/%x-%j.out" --export="$ex" nemorl/train.sbatch)
  jobs=$j
  if [[ ${RESUME:-1} == 1 ]] && awk "BEGIN{exit !($c < 0.4)}"; then
    j=$(sbatch --parsable -N 4 -t 1:30:00 -d afterany:$j -o "$NRL_LOGS/%x-%j.out" --export="$ex" nemorl/train.sbatch)
    jobs="$jobs $j"
  fi
  e=$(RUN_NAME=$run TAGS=tuned ADAPTER=$NRL_ROOT/runs/$run/checkpoints/step_$STEPS/policy/weights/model \
      OUT=$NT_DIR/runs/$run/gym sbatch --parsable -t 0:30:00 -d afterany:$j -o "$NRL_LOGS/%x-%j.out" --export=ALL nemotron/gym_eval.sbatch)
  echo -e "$run\t$c\t$s\t$STEPS\ttrain:$jobs\teval:$e" | tee -a "$NRL_ROOT/sweep_jobs.tsv"
done; done
