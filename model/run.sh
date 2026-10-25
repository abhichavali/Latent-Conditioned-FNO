#!/bin/bash
# Restart loop around train.py: resumes from the latest checkpoint until the run writes DONE.  Launch detached:
#   setsid nohup bash run.sh config/A.yaml > /dev/null 2>&1 < /dev/null & disown
# Log: <run_root>/<name>/stdout.log.  Stop: pkill -f '[r]un.sh config/A.yaml'; pkill -f '[t]rain.py --config config/A.yaml'
cd "$(dirname "$0")"
source /venv/main/bin/activate
CFG=$1
shift
RUN=$(python -c "import sys, yaml; sys.path.insert(0, '.'); from train import load_config as l; c = l('$CFG'); print(c['run_root'] + '/' + c['name'])")
mkdir -p "$RUN"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.85}
while [ ! -f "$RUN/DONE" ]; do
  echo "$(date) start $CFG $*" >> "$RUN/stdout.log"
  python train.py --config "$CFG" "$@" >> "$RUN/stdout.log" 2>&1
  code=$?
  [ -f "$RUN/DONE" ] && break
  echo "$(date) exited $code, restarting in 30s" >> "$RUN/stdout.log"
  sleep 30
done
