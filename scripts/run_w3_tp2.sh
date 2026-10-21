#!/bin/bash
# W3 infrastructure amendment I1: Qwen2.5-32B-Instruct (BF16, 62 GB) cannot hold a 32k-token KV cache on one 80 GB GPU
# (vLLM: 8.0 GiB needed, 6.0 GiB available), so it is served with tensor parallelism over both GPUs. This script runs
# after both single-GPU queues report QUEUE DONE; both GPUs must have >= 70 GB free for 10 min. The experiment protocol
# (prompts, seeds, arms, k* probe, max_tokens, max_model_len 32768, BF16) is unchanged.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/w3_tp2.log; PORT=8133
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} WQ_BASE_URL=http://127.0.0.1:$PORT/v1
export WQ_MODEL=Qwen/Qwen2.5-32B-Instruct; SH=qwen2.5-32b; D=$W/runs/backbones/$SH; mkdir -p $D
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
say "waiting for both single-GPU queues"
until grep -q "QUEUE DONE" $W/logs/w3_gpu0.log 2>/dev/null && grep -q "QUEUE DONE" $W/logs/w3_gpu1.log 2>/dev/null; do
  grep -q "FAIL" $W/logs/w3_gpu0.log $W/logs/w3_gpu1.log 2>/dev/null && fail "a single-GPU queue failed"; sleep 60; done
ok=0; while [ $ok -lt 20 ]; do
  n=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=71680' | wc -l)
  if [ "$n" -ge 2 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
say "both GPUs >= 70 GB free for 10 min"
CUDA_VISIBLE_DEVICES=0,1 nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_w3_tp2.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server died"; done
say "server ready TP=2 ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_tp2.log | head -1))"
cd $W/scripts
$PY run_webqa.py --dataset frames --split pilot --n 5 --k 2 --out $D/frames_smoke.jsonl >> $LOG 2>&1 || fail smoke
$PY run_webqa.py --dataset frames --split pilot --n 20 --k 1 --out $D/frames_probe.jsonl >> $LOG 2>&1 || fail probe
$PY token_probe_bb.py $D/frames_probe.jsonl $D >> $LOG 2>&1 || fail "token probe"
K=$($PY -c "import json;print(json.load(open('$D/K_FROZEN.json'))['k_star'])"); say "[$SH] k* = $K"
$PY run_webqa.py --dataset frames --split final --k $K --out $D/frames_final_gen.jsonl >> $LOG 2>&1 || fail final
$PY run_distribution.py --cond S_UNION --split final --out $D/S_UNION_final_gen.jsonl >> $LOG 2>&1 || fail S_UNION
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"
flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add runs/backbones/$SH && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "W3b backbone $SH (TP=2): smoke, probe, k*, FRAMES final + S_UNION generation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "QUEUE DONE"
