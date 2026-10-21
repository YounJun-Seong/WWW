#!/bin/bash
# W3b judging + sealed analyses, after both GPU queues report QUEUE DONE. Judge = Qwen3-32B-FP8 (snapshot aa55da1),
# frozen prompt; any GPU with >= 60 GB free for 10 min. Per backbone: judge final + S_UNION -> analyze.py (full sealed
# FRAMES analysis) -> analyze_backbone.py -> commit. Stops on any failure.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/w3_judge.log
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8122/v1
BB="qwen2.5-32b llama3.1-8b mistral-small-24b qwen2.5-7b"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
say "waiting for both GPU queues"
until grep -q "QUEUE DONE" $W/logs/w3_gpu0.log 2>/dev/null && grep -q "QUEUE DONE" $W/logs/w3_gpu1.log 2>/dev/null \
      && grep -q "QUEUE DONE" $W/logs/w3_tp2.log 2>/dev/null; do      # amendment I1: also wait for the TP=2 32B run
  grep -q "FAIL" $W/logs/w3_gpu0.log $W/logs/w3_gpu1.log $W/logs/w3_tp2.log 2>/dev/null && say "a queue failed; judging what exists" && break; sleep 60; done
ok=0; while [ $ok -lt 20 ]; do G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; say "GPU $G free for 10 min"
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_JUDGE_MODEL --port 8122 --dtype auto --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_w3_judge.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8122/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "judge server died"; done
say "judge ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_judge.log | head -1))"
cd $W/scripts; done_bb=""
for b in $BB; do D=$W/runs/backbones/$b
  [ -f $D/S_UNION_final_gen.jsonl ] || { say "[$b] no generation; skipped"; continue; }
  $PY judge.py --gen $D/frames_final_gen.jsonl --out $D/frames_final_judged.jsonl >> $LOG 2>&1 || fail "[$b] judge final"
  $PY judge.py --gen $D/S_UNION_final_gen.jsonl --out $D/S_UNION_final_judged.jsonl >> $LOG 2>&1 || fail "[$b] judge S_UNION"
  done_bb="$done_bb $b"
done
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "judge down"
for b in $done_bb; do D=$W/runs/backbones/$b
  WQ_OUT=$D $PY analyze.py frames final $D/frames_final_gen.jsonl $D/frames_final_judged.jsonl > /dev/null 2>> $LOG || fail "[$b] analyze.py"
  $PY analyze_backbone.py $D $b >> $LOG 2>&1 || fail "[$b] analyze_backbone"
done
flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add runs/backbones && flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "W3b judged + sealed backbone analyses (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE:$done_bb"
