#!/bin/bash
# HotpotQA final 1,000 (tag hotpot-final-freeze; pilot decision "no change"). One GPU, waits until some GPU has >= 60 GB
# free for 10 min; never touches other processes. generation (k* frozen) -> sealed analysis -> descriptive evidence log.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/hotpot_final.log; mkdir -p $W/logs $W/runs/final
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:8111/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
[ "$($GIT describe --tags --exact-match HEAD 2>/dev/null)" = hotpot-final-freeze ] || { say "FAIL: HEAD is not hotpot-final-freeze"; exit 1; }
say "waiting for a GPU"
ok=0; G=-1
while [ $ok -lt 20 ]; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
done
say "GPU $G free for 10 min"
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port 8111 --dtype bfloat16 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_hotpot_final.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8111/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || { say "FAIL: server died"; exit 1; }; done
say "server ready"
cd $W/scripts
K=$($PY -c "import json;print(json.load(open('$W/data/K_FROZEN_hotpot.json'))['k_star'])")
say "k* = $K (frozen from the pilot probe); final generation starting"
$PY run_webqa.py --dataset hotpot --split final --k $K --out $W/runs/final/hotpot_final_gen.jsonl >> $LOG 2>&1 || { say "FAIL: generation"; kill $S; exit 1; }
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"
$GIT add runs/final/hotpot_final_gen.jsonl && $GIT commit -q -m "HotpotQA final generation (1,000)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
WQ_OUT=$W/runs/final $PY analyze.py hotpot final $W/runs/final/hotpot_final_gen.jsonl >> $LOG 2>&1 || { say "FAIL: analysis"; exit 1; }
WQ_OUT=$W/runs/final $PY evidence_log.py hotpot final $W/runs/final/hotpot_final_gen.jsonl >> $LOG 2>&1 || { say "FAIL: evidence log"; exit 1; }
$GIT add runs/final/RESULTS_hotpot_final.* runs/final/EVIDENCE_LOG_hotpot_final.* && $GIT commit -q -m "HotpotQA final sealed analysis + descriptive evidence log (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/final/RESULTS_hotpot_final.txt"
