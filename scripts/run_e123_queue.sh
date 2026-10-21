#!/bin/bash
# E1–E3 generation queue for ONE GPU (tag e123-freeze). Usage: run_e123_queue.sh <gpu> <port> <job> ...
#   job = e1:<frames|hotpot>:<MODE> | e2:<CELL> | e3 | an:e1:hotpot (EM analysis, no judge needed)
# Each generation job first runs a 5-question smoke (label-free: 0 errors required), then the full set (cache reused).
# GPU used only after >= 58 GB free for 10 min; other processes never touched; no retries. Ends with "QUEUE COMPLETE".
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/e123_gpu$G.log; mkdir -p $W/logs $W/runs/e1 $W/runs/e2 $W/runs/e3
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
commit () { flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add "${@:2}" && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; }
smoke () { $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $1 || fail "smoke $1"; }
[ "$(git -C $W describe --tags --exact-match e123-freeze 2>/dev/null)" = e123-freeze ] || fail "tag e123-freeze missing"
cd $W/scripts; say "queue start: $*"
ok=0; while [ $ok -lt 20 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; say "GPU $G free for 10 min"
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_e123_gpu$G.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server died"; done
say "server ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_e123_gpu$G.log | head -1))"
for job in "$@"; do
  IFS=: read -r e x y <<< "$job"
  case $e in
    e1) o=$W/runs/e1/${x}_${y}_gen.jsonl; $PY run_e1.py --dataset $x --mode $y --n 5 --out $o >> $LOG 2>&1 || fail "$job smoke"; smoke $o
        $PY run_e1.py --dataset $x --mode $y --out $o >> $LOG 2>&1 || fail "$job" ;;
    e2) o=$W/runs/e2/${x}_gen.jsonl; $PY run_e23.py --exp e2 --cell $x --n 5 --out $o >> $LOG 2>&1 || fail "$job smoke"; smoke $o
        $PY run_e23.py --exp e2 --cell $x --out $o >> $LOG 2>&1 || fail "$job" ;;
    e3) o=$W/runs/e3/TOPIC_gen.jsonl; $PY run_e23.py --exp e3 --n 5 --out $o >> $LOG 2>&1 || fail "$job smoke"; smoke $o
        $PY run_e23.py --exp e3 --out $o >> $LOG 2>&1 || fail "$job" ;;
    an) $PY analyze_e123.py e1 hotpot >> $LOG 2>&1 || fail "E1 hotpot analysis"
        commit "E1 HotpotQA: generation + sealed EM analysis (run once)" runs/e1/hotpot_* runs/e1/RESULTS_e1_hotpot.*; continue ;;
  esac
  say "$job generated"
done
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"
commit "E1–E3 generation (GPU$G)" runs/e1 runs/e2 runs/e3
say "QUEUE COMPLETE"
