#!/bin/bash
# Resume of the Llama rp1.1 post-hoc sensitivity: judge stage onward. The first judge start (17:30) failed with OOM:
# GPU0 had just been taken by the W4 agent server, because run_overlap.sh matched this run's "generation DONE" log line
# as "finished". No judged output existed. This waits for W4 to finish, then judges and analyses as specified.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/w3_llama_rp.log
SH=llama3.1-8b_rp1.1; D=$W/runs/backbones/$SH
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8124/v1 WQ_CACHE=$W/data/llm_cache
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL (resume): $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
say "resume: judge stage; waiting for W4 to finish"
until grep -qE "DONE:|FAIL" $W/logs/overlap.log; do sleep 60; done
ok=0; while [ $ok -lt 20 ]; do GJ=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$GJ" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
CUDA_VISIBLE_DEVICES=$GJ nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port 8124 --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_w3_llama_rp_judge2.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8124/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "judge died"; done
say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_llama_rp_judge2.log | head -1))"
cd $W/scripts
$PY judge.py --gen $D/frames_final_gen.jsonl --out $D/frames_final_judged.jsonl >> $LOG 2>&1 || fail "judge final"
$PY judge.py --gen $D/S_UNION_final_gen.jsonl --out $D/S_UNION_final_judged.jsonl >> $LOG 2>&1 || fail "judge S_UNION"
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "judge down"
WQ_OUT=$D $PY analyze.py frames final $D/frames_final_gen.jsonl $D/frames_final_judged.jsonl > /dev/null 2>> $LOG || fail analyze.py
$PY analyze_backbone.py $D "$SH (POST-HOC sensitivity)" >> $LOG 2>&1 || fail analyze_backbone
flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add runs/backbones/$SH && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "W3 post-hoc sensitivity: Llama rp1.1 judged + analysed (after resume)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "SENSITIVITY COMPLETE"
