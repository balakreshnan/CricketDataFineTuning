#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -J general_sa-cricket-rhea.probe
# What is a Rhea compute node? GPUs, python, container tooling, local scratch.
echo "== $(hostname) $(uname -m)"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi | head -15
echo "== python"; for p in python3 python3.12 python3.11 uv pip3; do command -v $p && $p --version 2>&1 | head -1; done
python3 -c "import venv, ensurepip; print('venv+ensurepip ok')" 2>&1
echo "== container"; command -v enroot; srun --help 2>&1 | grep -m3 container
echo "== scratch"; df -h /tmp /raid /scratch 2>/dev/null; nproc; free -g | head -2
echo "== net"; curl -sI -m 10 https://huggingface.co | head -1; curl -sI -m 10 https://download.pytorch.org/whl/cu128 | head -1; curl -sI -m 10 https://nvcr.io/v2/ | head -1
echo "== cuda libs"; ls /usr/local | head; ldconfig -p | grep -m3 -E "libcuda|libnvidia-ml"
