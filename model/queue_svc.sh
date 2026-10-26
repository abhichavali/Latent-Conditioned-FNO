#!/bin/bash
# Scratch vs v3 on fresh equations (scratch_vs_ctx.py), pdebox4, unattended and resumable (each stage skips what is
# done; scratch runs resume from their checkpoints):
#   setsid nohup bash queue_svc.sh > /workspace/logs/svc.log 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
O=/workspace/results/scratch_vs_ctx
CK=/workspace/runs/ctx/ckpt/step_00110000.eqx
CK_BYTES=6254597140
HOURS=${HOURS:-3}  # no scratch run starts later than this after the first launch (GPU budget)
mkdir -p $O/logs
[ -f $O/deadline ] || echo $(( $(date +%s) + HOURS * 3600 )) > $O/deadline
DL=$(cat $O/deadline)
say() { echo "$(date -u +%H:%M:%S) $*"; }
retry() { for t in 1 2 3; do "$@" && return 0; say "failed ($t): $*"; sleep 30; done; return 1; }

# 1 dataset C (gen_C.sh: 6 two-field + 6 three-field equations, seed 99, B-style flags, 192 windows each)
while pgrep -f "[g]enerate.py" > /dev/null; do sleep 60; done
grep -q GENDONE /workspace/logs/gen_C.log || bash /workspace/gen_C.sh >> /workspace/logs/gen_C.log 2>&1
grep -q GENDONE /workspace/logs/gen_C.log || { say "generation failed"; exit 1; }
# 2 split
[ -f $O/split.json ] || python scratch_vs_ctx.py split --out $O || exit 1
# 3 v3 k-shot (alone on the GPU)
until [ "$(stat -c %s $CK 2>/dev/null)" = "$CK_BYTES" ]; do say "waiting for $CK"; sleep 120; done
[ -f $O/ctx_kshot.json ] || XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 retry python fewshot_ctx.py fresh --ckpt $CK --split $O/split.json --out $O
say "ctx done"
# 4 scratch: 2 workers share the GPU
for i in 0 1; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 retry python baselines.py eq --config config/scratch.yaml --split $O/split.json \
    --out $O/scratch --worker $i --workers 2 --deadline $DL > $O/logs/scratch_$i.log 2>&1 &
done
wait
say "scratch done"
# 5 summary
python scratch_vs_ctx.py summary --out $O && touch $O/DONE
say "queue done"
