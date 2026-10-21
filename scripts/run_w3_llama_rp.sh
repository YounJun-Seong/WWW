#!/bin/bash
# W3 POST-HOC SENSITIVITY (decided by the user after seeing the label-free execution stats of the frozen Llama run:
# ~30% of calls hit the 1,024-token limit through greedy repetition loops). The frozen Llama-3.1-8B result is kept and
# reported unchanged; this is an additional, separately labelled run. Only change: server-side repetition_penalty 1.1
# (vLLM --override-generation-config). Same prompts, seeds, arms, k* = the frozen Llama k* (no re-probe), separate LLM
# cache (the cache key does not contain the server default). GPU0 only, between the Llama queue and the TP=2 run.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/w3_llama_rp.log; PORT=8134; G=0
SH=llama3.1-8b_rp1.1; D=$W/runs/backbones/$SH; mkdir -p $D
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=meta-llama/Llama-3.1-8B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1 WQ_CACHE=$W/data/llm_cache_llama_rp11
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8124/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
commit () { flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add "${@:2}" && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; }
gate () { local ok=0; while [ $ok -lt 20 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; say "GPU $G free for 10 min"; }
say "waiting for the frozen Llama queue (GPU0) to finish"
until grep -q "QUEUE DONE" $W/logs/w3_gpu0.log; do grep -q FAIL $W/logs/w3_gpu0.log && fail "GPU0 queue failed"; sleep 60; done
K=$($PY -c "import json;print(json.load(open('$W/runs/backbones/llama3.1-8b/K_FROZEN.json'))['k_star'])")
gate
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --override-generation-config '{"repetition_penalty": 1.1}' \
  --disable-log-requests > $W/logs/vllm_w3_llama_rp.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server died"; done
grep -q "repetition_penalty" $W/logs/vllm_w3_llama_rp.log || fail "override not visible in the server log"
say "server ready, repetition_penalty 1.1 ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_llama_rp.log | head -1)); k* = $K (frozen Llama)"
cd $W/scripts
$PY run_webqa.py --dataset frames --split final --k $K --out $D/frames_final_gen.jsonl >> $LOG 2>&1 || fail final
$PY run_distribution.py --cond S_UNION --split final --out $D/S_UNION_final_gen.jsonl >> $LOG 2>&1 || fail S_UNION
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"
commit "W3 post-hoc sensitivity: Llama-3.1-8B with repetition_penalty 1.1 (generation)" runs/backbones/$SH
say "generation DONE; waiting for the main judge stage to finish"
until grep -qE "DONE:|FAIL" <(tail -n +5 $W/logs/w3_judge.log); do sleep 60; done
ok=0; while [ $ok -lt 20 ]; do GJ=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$GJ" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
CUDA_VISIBLE_DEVICES=$GJ nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port 8124 --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_w3_llama_rp_judge.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8124/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "judge died"; done
say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_llama_rp_judge.log | head -1))"
WQ_CACHE=$W/data/llm_cache $PY judge.py --gen $D/frames_final_gen.jsonl --out $D/frames_final_judged.jsonl >> $LOG 2>&1 || fail "judge final"
WQ_CACHE=$W/data/llm_cache $PY judge.py --gen $D/S_UNION_final_gen.jsonl --out $D/S_UNION_final_judged.jsonl >> $LOG 2>&1 || fail "judge S_UNION"
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "judge down"
WQ_OUT=$D $PY analyze.py frames final $D/frames_final_gen.jsonl $D/frames_final_judged.jsonl > /dev/null 2>> $LOG || fail analyze.py
$PY analyze_backbone.py $D "$SH (POST-HOC sensitivity)" >> $LOG 2>&1 || fail analyze_backbone
commit "W3 post-hoc sensitivity: Llama rp1.1 judged + analysed" runs/backbones/$SH
say "DONE"
