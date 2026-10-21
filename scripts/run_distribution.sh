#!/bin/bash
# Experiment W2 (tag dist-freeze): FRAMES evidence distribution. GPU gate (>= 60 GB free for 10 min; other processes never
# touched). Agent server -> pilot 200 x 3 conditions -> label-free execution check (stop on FAIL) -> final 624 x 3
# conditions -> down -> judge server -> judge -> down -> sealed analysis (run once). No retries; failures keep their logs.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/distribution.log; D=$W/runs/dist; mkdir -p $W/logs $D
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:8121/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8122/v1
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
C="S_UNION S_5 S_UNION_T7"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
gate () { local ok=0; G=""
  while [ $ok -lt 20 ]; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
    if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
  done; say "GPU $G free for 10 min"; }
serve () {
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $1 --port $2 --dtype auto --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_dist_$2.log 2>&1 &
  S=$!
  until curl -sf http://127.0.0.1:$2/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server $1 died"; done
  say "server $1 ready on GPU $G ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_dist_$2.log | head -1))"; }
down () { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"; }
[ "$($GIT describe --tags --exact-match HEAD 2>/dev/null)" = dist-freeze ] || fail "HEAD is not dist-freeze"
cd $W/scripts; say "start; waiting for a GPU"
gate; serve $WQ_MODEL 8121
for c in $C; do $PY run_distribution.py --cond $c --split pilot --out $D/${c}_pilot_gen.jsonl >> $LOG 2>&1 || fail "pilot $c"; done
$PY check_distribution.py pilot $D >> $LOG 2>&1 || fail "execution check (runs/dist/CHECK_pilot.txt)"
$GIT add runs/dist/*_pilot_gen.jsonl runs/dist/CHECK_pilot.txt && $GIT commit -q -m "W2 pilot 200 x 3 conditions + label-free execution check (PASS)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for c in $C; do $PY run_distribution.py --cond $c --split final --out $D/${c}_final_gen.jsonl >> $LOG 2>&1 || fail "final $c"; done
down
$GIT add runs/dist/*_final_gen.jsonl && $GIT commit -q -m "W2 final 624 x 3 conditions (generation)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
gate; serve $WQ_JUDGE_MODEL 8122
for c in $C; do $PY judge.py --gen $D/${c}_final_gen.jsonl --out $D/${c}_final_judged.jsonl >> $LOG 2>&1 || fail "judge $c"; done
down
$PY analyze_distribution.py $D >> $LOG 2>&1 || fail analysis
$GIT add runs/dist/*_final_judged.jsonl runs/dist/RESULTS_distribution.* && $GIT commit -q -m "W2 judged + sealed analysis (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/dist/RESULTS_distribution.txt"
