#!/bin/bash
# Experiment W4 (tag w4-freeze): evidence overlap o = 1..5 on FRAMES final 624, Qwen2.5-14B (o = 0 = the frozen D run).
# Starts after every W3 run (main judge stage and the Llama sensitivity) has finished. Two agent servers, one per GPU
# (each GPU >= 58 GB free for 10 min; other processes never touched): GPU0 runs o = 1, 3, 5 and GPU1 runs o = 2, 4.
# Then one judge server -> judge -> sealed analysis (run once). No retries.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/overlap.log; D=$W/runs/overlap; mkdir -p $D
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} WQ_MODEL=Qwen/Qwen2.5-14B-Instruct
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8143/v1
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; kill $S0 $S1 $SJ 2>/dev/null; exit 1; }
gate () { local g=$1 ok=0; while [ $ok -lt 20 ]; do f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; say "GPU $g free for 10 min"; }
serve () { CUDA_VISIBLE_DEVICES=$1 nohup $VPY/vllm serve $2 --port $3 --dtype auto --gpu-memory-utilization 0.70 --max-model-len 32768 \
  --generation-config vllm --disable-log-requests > $W/logs/vllm_overlap_$3.log 2>&1 & echo $!; }
up () { until curl -sf http://127.0.0.1:$1/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $2 2>/dev/null || fail "server $1 died"; done
  say "server :$1 ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_overlap_$1.log | head -1))"; }
[ "$($GIT describe --tags --exact-match w4-freeze 2>/dev/null)" = w4-freeze ] || fail "tag w4-freeze missing"
say "waiting for W3 (main judge + Llama sensitivity)"
until tail -n +5 $W/logs/w3_judge.log | grep -q "DONE:" && grep -qE "(DONE|FAIL)" $W/logs/w3_llama_rp.log; do
  tail -n +5 $W/logs/w3_judge.log | grep -q FAIL && fail "W3 judge failed"; sleep 60; done
gate 0; S0=$(serve 0 $WQ_MODEL 8141); gate 1; S1=$(serve 1 $WQ_MODEL 8142); up 8141 $S0; up 8142 $S1
cd $W/scripts
( for o in 1 3 5; do WQ_BASE_URL=http://127.0.0.1:8141/v1 $PY run_overlap.py --o $o --out $D/O${o}_final_gen.jsonl >> $LOG 2>&1 || exit 1; done ) & P0=$!
( for o in 2 4; do WQ_BASE_URL=http://127.0.0.1:8142/v1 $PY run_overlap.py --o $o --out $D/O${o}_final_gen.jsonl >> $LOG 2>&1 || exit 1; done ) & P1=$!
wait $P0 || fail "generation GPU0"; wait $P1 || fail "generation GPU1"
kill $S0 $S1; while kill -0 $S0 2>/dev/null || kill -0 $S1 2>/dev/null; do sleep 3; done; say "agent servers down"
$GIT add runs/overlap && $GIT commit -q -m "W4 overlap o=1..5 generation (FRAMES final)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
gate 0; SJ=$(serve 0 $WQ_JUDGE_MODEL 8143); up 8143 $SJ
for o in 1 2 3 4 5; do $PY judge.py --gen $D/O${o}_final_gen.jsonl --out $D/O${o}_final_judged.jsonl >> $LOG 2>&1 || fail "judge o=$o"; done
kill $SJ; while kill -0 $SJ 2>/dev/null; do sleep 3; done; SJ=""; say "judge down"
$PY analyze_overlap.py $D >> $LOG 2>&1 || fail analysis
$GIT add runs/overlap && $GIT commit -q -m "W4 overlap judged + sealed analysis (run once)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/overlap/RESULTS_overlap.txt"
