#!/bin/bash
# HQ runs (DESIGN_HQ.md, tag hq-freeze). hq_queue.sh <gpu> <port> <MODE>...  (MODE = BASE | PEER_EV | PEER_FULL | REMOTE_Q | REMOTE_GEN)
# env UTIL (default 0.70) / MAXLEN (default 32768): lowered only when other users occupy the GPU (logged in DESIGN_HQ.md).
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/hq_gpu$G.log; O=$W/runs/hq; mkdir -p $O
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
U=${UTIL:-0.70}; ML=${MAXLEN:-32768}
need=$(python3 -c "print(int(81559*$U)+2000)")
waitgpu () { ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge "$need" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
[ -n "$(git -C $W tag --contains hq-freeze 2>/dev/null)" ] || { say "FAIL tag hq-freeze missing"; exit 1; }
cd $W/scripts; say "queue start: $* (util $U, max-model-len $ML)"
# shared GPU: memory can be claimed by another user between the free-memory check and our startup -> wait and retry
while :; do
  waitgpu
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization $U \
    --max-model-len $ML --generation-config vllm --disable-log-requests > $W/logs/vllm_hq_g$G.log 2>&1 &
  S=$!; up=0
  while kill -0 $S 2>/dev/null; do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }; sleep 5; done
  [ $up = 1 ] && break
  say "server start failed (memory claimed by another process?) - waiting to retry"; sleep 60
done
say "server ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_hq_g$G.log | head -1))"
for mode in "$@"; do
  o=$O/hq_${mode}_gen.jsonl
  if $PY hq_run.py --mode $mode --n 5 --out $o.smoke >> $LOG 2>&1 && $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $o.smoke; then
    rm -f $o.smoke; $PY hq_run.py --mode $mode --out $o >> $LOG 2>&1 && say "done $mode" || say "FAIL $mode"
  else say "FAIL smoke $mode"; fi
done
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"; say "QUEUE COMPLETE"
