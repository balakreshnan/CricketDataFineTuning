#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -J general_sa-gsm8k.data
# Download GSM8K to Lustre and convert it to the cricket dataset layout (gsm8k_prepare.py).
set -euo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/config.env"
python3 "$CODE_DIR/gsm8k/gsm8k_prepare.py" "$DATA_DIR/GSM8K"
wc -l "$DATA_DIR"/GSM8K/data/*.jsonl; head -c 700 "$DATA_DIR/GSM8K/data/eval-00000.jsonl"; echo
