#!/bin/bash
# After the 32B replication queue exits: if AgentWebBench is not finished, serve Qwen2.5-14B on both GPUs (ports 8161 /
# 8171, the ports awb_supervise.sh looks for) until the supervisor reports completion; then swap to the frozen judge on
# GPU0 and score all three variants with awb_eval.py. Servers are up only while needed; other users' processes untouched.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/awb_finish.log
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} PYTHONPATH=$W/.pylib:$W/scripts
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
waitgpu () { ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
Q32=$1
say "waiting for 32B queue (pid $Q32) to exit"; while kill -0 $Q32 2>/dev/null; do sleep 60; done
SUP=$(cat $W/runs/awb/PID_supervisor)
S0=""; S1=""
if kill -0 $SUP 2>/dev/null; then
  for g in 0 1; do waitgpu $g; p=$((8161 + 10*g))
    CUDA_VISIBLE_DEVICES=$g nohup $VPY/vllm serve Qwen/Qwen2.5-14B-Instruct --port $p --dtype bfloat16 --gpu-memory-utilization 0.70 \
      --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_awb_gpu$g.log 2>&1 &
    eval S$g=$!; done
  for p in 8161 8171; do until curl -sf http://127.0.0.1:$p/v1/models >/dev/null 2>&1; do sleep 5; done; done
  say "14B servers up for AWB (pids $S0 $S1)"
  while kill -0 $SUP 2>/dev/null; do sleep 60; done
  say "AWB supervisor exited: $(tail -n 1 $W/logs/awb_supervise.log)"
  for s in $S0 $S1; do kill $s; while kill -0 $s 2>/dev/null; do sleep 3; done; done; say "14B servers down"
fi
waitgpu 0
CUDA_VISIBLE_DEVICES=0 nohup $VPY/vllm serve Qwen/Qwen3-32B-FP8 --port 8162 --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_awb_judge.log 2>&1 &
J=$!; until curl -sf http://127.0.0.1:8162/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $J 2>/dev/null || { say "FAIL judge server died"; exit 1; }; done
say "judge up ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_awb_judge.log | head -1))"
export WQ_JUDGE_URL=http://127.0.0.1:8162/v1 WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8
for m in multi_agent multi_agent_docs multi_agent_summary; do
  $PY $W/scripts/awb_eval.py $m >> $LOG 2>&1 && say "evaluated $m" || say "FAIL eval $m"; done
kill $J; while kill -0 $J 2>/dev/null; do sleep 3; done; say "judge down; AWB FINISH COMPLETE"
