#!/bin/bash
# E1 (FRAMES), E2, E3: judging + sealed analyses, after both queues print "QUEUE COMPLETE" (exact marker). Frozen judge.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/e123_judge.log
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8152/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
say "waiting for both E1–E3 queues"
until grep -qx ".* QUEUE COMPLETE" $W/logs/e123_gpu0.log 2>/dev/null && grep -qx ".* QUEUE COMPLETE" $W/logs/e123_gpu1.log 2>/dev/null; do
  grep -q "FAIL" $W/logs/e123_gpu0.log $W/logs/e123_gpu1.log 2>/dev/null && fail "a queue failed"; sleep 60; done
ok=0; while [ $ok -lt 20 ]; do G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; say "GPU $G free for 10 min"
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port 8152 --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_e123_judge.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8152/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "judge died"; done
say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_e123_judge.log | head -1))"
cd $W/scripts
for f in $W/runs/e1/frames_*_gen.jsonl $W/runs/e2/*_gen.jsonl $W/runs/e3/TOPIC_gen.jsonl; do
  $PY judge.py --gen $f --out ${f%_gen.jsonl}_judged.jsonl >> $LOG 2>&1 || fail "judge $f"; done
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "judge down"
for a in "e1 frames" e2 e3; do $PY analyze_e123.py $a >> $LOG 2>&1 || fail "analysis $a"; done
flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add runs/e1 runs/e2 runs/e3 && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "E1 (FRAMES), E2, E3 judged + sealed analyses (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "ALL COMPLETE"
