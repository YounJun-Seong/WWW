#!/bin/bash
# RX 32B replication of R1/R2 on FRAMES (Qwen2.5-32B-Instruct, BF16, TP=2 over both GPUs, util 0.85, max-model-len
# 32768: exactly the W3b serving config). Starts only when BOTH GPUs have >= 70 GB free for 2 min (i.e. after every
# single-GPU RX queue has released them). Then swaps to the frozen judge on GPU 0 and judges the 32B files.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/rx_32b.log; PORT=8191; O=$W/runs/rx32; mkdir -p $O
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-32B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:$((PORT+1))/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
S=""; stop () { [ -n "$S" ] && { kill $S 2>/dev/null; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"; }; S=""; }
die () { say "FAIL: $1"; stop; exit 1; }
say "waiting for both GPUs >= 70 GB free for 2 min"
ok=0; while [ $ok -lt 4 ]; do n=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=71680' | wc -l)
  if [ "$n" -ge 2 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
CUDA_VISIBLE_DEVICES=0,1 nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_rx_32b.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || die "32B server died"; done
say "32B ready TP=2 ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_rx_32b.log | head -1))"
cd $W/scripts
for m in R1_FULL R1_GOLDRM R1_GOLDREP R1_NONGOLDRM R2_I1a R2_I1b R2_I2 R2_I3 R2_I4; do
  o=$O/frames_${m}_gen.jsonl
  if $PY rx_run32.py --dataset frames --mode $m --n 5 --out $o.smoke >> $LOG 2>&1 && \
     $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $o.smoke; then
    rm -f $o.smoke
    $PY rx_run32.py --dataset frames --mode $m --out $o >> $LOG 2>&1 && say "frames:$m generated (32B)" || say "FAIL frames:$m (32B)"
  else say "FAIL smoke frames:$m (32B)"; fi
done
stop
ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
CUDA_VISIBLE_DEVICES=0 nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port $((PORT+1)) --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_rx_32b_judge.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$((PORT+1))/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || die "judge died"; done
say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_rx_32b_judge.log | head -1))"
for f in $O/frames_*_gen.jsonl; do $PY rx_judge.py --gen $f --out ${f%_gen.jsonl}_judged.jsonl >> $LOG 2>&1 && say "judged ${f##*/}" || say "FAIL judge ${f##*/}"; done
stop; say "32B QUEUE COMPLETE"
