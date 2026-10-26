#!/bin/bash
# v3 queue on pdebox3, runs unattended: guard the training run (NaN, deadline), then few-shot evaluation, rehearsal
# k-shot, summary.  Resumable: rerunning continues (fewshot_ctx.py resumes from state.npz, skips DONE runs).
#   setsid nohup bash queue_ctx.sh > /workspace/runs/ctx_queue.log 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
source /venv/main/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8  # 0.9 left no room to load a 2 GB jit executable (fewshot OOM)
RUN=/workspace/runs/ctx
OUT=/workspace/results/ctx
DEADLINE=${DEADLINE:-0245}   # UTC HHMM: stop training here (SIGTERM checkpoints) and evaluate the latest checkpoint
V1_CUTOFF=${V1_CUTOFF:-0345} # UTC HHMM: start the v1-random-sims phase only before this (credit)
mkdir -p $OUT
say() { echo "$(date -u +%H:%M:%S) $*"; }
in_window() { n=$(date -u +%H%M); [ "$n" -ge "$1" ] && [ "$n" -lt 1200 ]; }  # after HHMM tonight (run starts ~17 UTC)
stop_training() { pkill -f "[r]un.sh config/ctx.yaml"; sleep 2; pkill -TERM -f "[t]rain_ctx.py --config"; sleep 120; }

while [ ! -f $RUN/DONE ] && [ ! -f $RUN/STOPPED_NAN ] && [ ! -f $RUN/STOPPED_DEADLINE ]; do
  sleep 300
  # NaN guard: the last 20 train records (>= 2000 steps, or 20 restarts that each aborted) mostly non-finite
  if python -c "
import json, sys
L = [json.loads(l) for l in open('$RUN/log.jsonl') if '\"train/nonfinite\"' in l][-20:]
sys.exit(0 if len(L) == 20 and all(r['train/nonfinite'] >= 50 for r in L) else 1)" 2>/dev/null; then
    say "NaN guard: stopping training"; touch $RUN/STOPPED_NAN; stop_training
  fi
  if in_window $DEADLINE; then say "deadline: stopping training"; touch $RUN/STOPPED_DEADLINE; stop_training; fi
done
[ -f $RUN/STOPPED_NAN ] && { say "training NaN; no evaluation"; exit 1; }
CK=$(ls $RUN/ckpt/step_*.eqx | tail -n 1)
say "evaluating $CK"
job() {  # 3 tries each
  for t in 1 2 3; do python fewshot_ctx.py "$@" --run $RUN --ckpt $CK && return 0; say "failed ($t): $*"; sleep 60; done
}
# (a), (b) on every target first (headline), then the 515M fine-tune chain (c): same acquisitions (amortized only)
for tg in swe ap fire; do job run --target $tg --method active --skip_c --out $OUT/active/${tg}_activeab; done
job rehearsal --out $OUT
python fewshot_ctx.py summary --results $OUT
for tg in swe ap fire; do job run --target $tg --method active --out $OUT/active/${tg}_active; done
python fewshot_ctx.py summary --results $OUT
if ! in_window $V1_CUTOFF; then
  for tg in swe ap fire; do job run --target $tg --method v1random --out $OUT/active/${tg}_v1random; done
  python fewshot_ctx.py summary --results $OUT
fi
touch $OUT/QUEUE_DONE
say "queue done"
