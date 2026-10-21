#!/bin/bash
# RX queue for ONE GPU (tag rx-freeze). Usage: rx_queue.sh <gpu> <port> <job> ...
#   job = gen:<frames|hotpot>:<MODE>   generation (5-question smoke with 0 errors required, then the full set)
#       | judge                         swap the agent server for the frozen judge, judge every FRAMES RX file that has
#                                       no judged file yet, then bring the agent server back for the remaining jobs
# GPU used only after >= 58 GB free for 2 min; other users' processes are never touched; no retries; a failed job is
# logged and the queue moves on (its output is not used). Servers are up only while a job needs them.
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/rx_gpu$G.log; mkdir -p $W/logs $W/runs/rx
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:$((PORT+1))/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
S=""
stop () { [ -n "$S" ] && { kill $S 2>/dev/null; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"; }; S=""; }
die () { say "FAIL: $1"; stop; exit 1; }
waitgpu () { ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
agent () { [ -n "$S" ] && [ "$MODE" = agent ] && return; stop; waitgpu
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_rx_gpu$G.log 2>&1 &
  S=$!; MODE=agent
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || die "agent server died"; done
  say "agent ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_rx_gpu$G.log | head -1))"; }
judgesrv () { stop; waitgpu
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port $((PORT+1)) --dtype auto --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_rx_judge_gpu$G.log 2>&1 &
  S=$!; MODE=judge
  until curl -sf http://127.0.0.1:$((PORT+1))/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || die "judge server died"; done
  say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_rx_judge_gpu$G.log | head -1))"; }
[ "$(git -C $W describe --tags --exact-match rx-freeze 2>/dev/null)" = rx-freeze ] || [ -n "$(git -C $W tag --contains rx-freeze 2>/dev/null)" ] || die "tag rx-freeze missing"
cd $W/scripts; say "queue start: $*"
for job in "$@"; do
  IFS=: read -r t ds m <<< "$job"
  if [ "$t" = judge ]; then
    judgesrv
    for f in $W/runs/rx/frames_*_gen.jsonl; do j=${f%_gen.jsonl}_judged.jsonl
      [ -s "$j" ] && continue
      grep -q "^${f##*/} done" $W/runs/rx/DONE 2>/dev/null || continue
      $PY rx_judge.py --gen $f --out $j >> $LOG 2>&1 && say "judged ${f##*/}" || say "FAIL judge ${f##*/}"; done
    stop; continue
  fi
  agent
  o=$W/runs/rx/${ds}_${m}_gen.jsonl
  if $PY rx_run.py --dataset $ds --mode $m --n 5 --out $o.smoke >> $LOG 2>&1 && \
     $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $o.smoke; then
    rm -f $o.smoke
    if $PY rx_run.py --dataset $ds --mode $m --out $o >> $LOG 2>&1; then
      say "$ds:$m generated"; flock $W/runs/rx/DONE.lock sh -c "echo '${o##*/} done' >> $W/runs/rx/DONE"
    else say "FAIL $ds:$m"; fi
  else say "FAIL smoke $ds:$m"; fi
done
stop; say "QUEUE COMPLETE"
