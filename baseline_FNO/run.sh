#!/bin/bash
# Detached end-to-end baseline on the GPU box: waits for any dataset generation on this GPU, then data -> sweep -> report.
#   setsid nohup bash baseline_FNO/run.sh > baseline_FNO/run.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
while pgrep -f "[p]ython generate.py" > /dev/null; do sleep 60; done
source /venv/main/bin/activate
python data.py --out /root/baseline_data && python train.py --data /root/baseline_data --out results
