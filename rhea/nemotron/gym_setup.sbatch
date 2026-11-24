#!/bin/bash
#SBATCH -A general_sa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH -J general_sa-nemotron-rhea.gymsetup
# Install NeMo Gym into a venv on Lustre and scaffold an environment, printing what the CLI offers and generates.
set -uo pipefail
source "${CODE_DIR:-/home/bbalakreshna/clustercodes}/nemotron/nemotron.env"
export GYM_WORK=$NT_DIR/gym
mkdir -p "$GYM_WORK"
srun --container-image="$IMAGE_SQSH" --container-mounts="$LUSTRE_DIR:$LUSTRE_DIR" --container-mount-home \
     --no-container-remap-root bash -c '
  source '"$CODE_DIR"'/nemotron/nemotron.env
  export GYM_WORK=$NT_DIR/gym
  # NeMo Gym 0.7.0 needs Python 3.13: on Python 3.14 the hydra CLI fails ("badly formed help string").
  # uv fetches a standalone Python 3.13 for aarch64 and builds the venv with it; everything stays on Lustre.
  export UV_PYTHON_INSTALL_DIR=$LUSTRE_DIR/cache/uv-python UV_CACHE_DIR=$LUSTRE_DIR/cache/uv
  if [[ ! -x $GYM_VENV/bin/gym ]] || ! "$GYM_VENV/bin/python" -c "import sys; sys.exit(sys.version_info[:2] != (3, 13))"; then
    rm -rf "$GYM_VENV" /tmp/uvboot
    python3 -m venv /tmp/uvboot && /tmp/uvboot/bin/pip install -q uv
    /tmp/uvboot/bin/uv venv --python 3.13 "$GYM_VENV" 2>&1 | tail -3
    /tmp/uvboot/bin/uv pip install --python "$GYM_VENV/bin/python" "nemo-gym[ray]==$GYM_VERSION" uv 2>&1 | tail -4
  fi
  source "$GYM_VENV/bin/activate"
  echo "### version"; gym --version; python --version; which uv ray 2>&1
  echo "### gym --help"; gym --help 2>&1 | head -60
  echo "### gym env --help"; gym env --help 2>&1 | head -40
  echo "### gym eval run --help"; gym eval run --help 2>&1 | head -120
  echo "### gym env start --help"; gym env start --help 2>&1 | head -60
  cd "$GYM_WORK"
  echo "### init"; rm -rf resources_servers/cricket_scaffold; gym env init --resources-server cricket_scaffold 2>&1 | tail -20
  echo "### tree"; find . -maxdepth 4 -not -path "*/.venv/*" -not -name "*.lock" | head -80
  for f in resources_servers/cricket_scaffold/app.py resources_servers/cricket_scaffold/configs/*.yaml resources_servers/cricket_scaffold/pyproject.toml resources_servers/cricket_scaffold/requirements.txt; do
    [[ -f $f ]] && { echo "### FILE $f"; cat "$f"; }
  done
  echo "### example data"; head -c 1500 resources_servers/cricket_scaffold/data/example.jsonl 2>/dev/null; echo
  echo "### package layout"; P=$(python -c "import nemo_gym, os; print(os.path.dirname(nemo_gym.__file__))"); echo $P; ls $P | head -60
  ls $P/.. | grep -i -E "resources_servers|responses_api|benchmarks" ; find $P/.. -maxdepth 3 -path "*vllm_model*" | head -20
  echo "### vllm model config"; find $P/.. -maxdepth 5 -name "vllm_model.yaml" | head -3 | while read f; do echo "# $f"; cat "$f"; done
  echo "### simple agent config"; find $P/.. -maxdepth 5 -name "simple_agent.yaml" | head -1 | while read f; do echo "# $f"; cat "$f"; done
  echo "### list"; gym list resources-servers 2>&1 | head -30; gym list models 2>&1 | head -20
'
echo "[$(date +%T)] gym setup done"
