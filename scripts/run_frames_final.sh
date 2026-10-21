#!/bin/bash
# FRAMES final 624 (tag frames-final-freeze; pilot decision "no change"). Uses the frozen corpus, BM25 index, partition,
# K_FROZEN_frames and code unchanged. GPU gate (>= 60 GB free for 10 min; other processes never touched):
# agent server -> generation -> down -> judge server -> judge -> down -> sealed analysis -> evidence log -> run record.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/frames_final.log; mkdir -p $W/logs $W/runs/final
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:8121/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8122/v1
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
gate () { local ok=0; G=""
  while [ $ok -lt 20 ]; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
    if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
  done; say "GPU $G free for 10 min"; }
serve () {
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $1 --port $2 --dtype auto --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_frames_final_$2.log 2>&1 &
  S=$!
  until curl -sf http://127.0.0.1:$2/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server $1 died"; done
  say "server $1 ready on GPU $G ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_frames_final_$2.log | head -1))"; }
down () { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"; }
[ "$($GIT describe --tags --exact-match HEAD 2>/dev/null)" = frames-final-freeze ] || fail "HEAD is not frames-final-freeze"
cd $W/scripts
$PY verify_snapshot.py snapshot >> $LOG 2>&1 || fail "snapshot re-check"
$PY verify_snapshot.py index >> $LOG 2>&1 || fail "index re-check"
K=$($PY -c "import json;print(json.load(open('$W/data/K_FROZEN_frames.json'))['k_star'])")
say "k* = $K (frozen from the pilot probe); waiting for a GPU"
gate; serve $WQ_MODEL 8121
$PY run_webqa.py --dataset frames --split final --k $K --out $W/runs/final/frames_final_gen.jsonl >> $LOG 2>&1 || fail generation
down
$GIT add runs/final/frames_final_gen.jsonl && $GIT commit -q -m "FRAMES final generation (624)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
gate; serve $WQ_JUDGE_MODEL 8122
$PY judge.py --gen $W/runs/final/frames_final_gen.jsonl --out $W/runs/final/frames_final_judged.jsonl >> $LOG 2>&1 || fail judge
down
WQ_OUT=$W/runs/final $PY analyze.py frames final $W/runs/final/frames_final_gen.jsonl $W/runs/final/frames_final_judged.jsonl >> $LOG 2>&1 || fail analysis
WQ_OUT=$W/runs/final $PY evidence_log_frames.py final $W/runs/final/frames_final_gen.jsonl $W/runs/final/frames_final_judged.jsonl >> $LOG 2>&1 || fail "evidence log"
WQ_OUT=$W/runs/final $PY sensitivity_frames.py final $W/runs/final/frames_final_gen.jsonl $W/runs/final/frames_final_judged.jsonl >> $LOG 2>&1 || fail "sensitivity"
$PY run_record.py frames_final frames final $W/runs/final/frames_final_gen.jsonl $W/runs/final/RESULTS_frames_final.txt $LOG judge >> $LOG 2>&1 || fail "run record"
$GIT add runs/final/frames_final_judged.jsonl runs/final/RESULTS_frames_final.* runs/final/EVIDENCE_LOG_frames_final.* runs/final/SENSITIVITY_frames_final.txt runs/final/RUN_RECORD_frames_final.json && $GIT commit -q -m "FRAMES final judged + sealed analysis + evidence log + run record (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/final/RESULTS_frames_final.txt"
