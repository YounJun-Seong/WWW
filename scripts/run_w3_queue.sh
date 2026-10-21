#!/bin/bash
# Experiment W3 queue for ONE GPU (tag w3-freeze). Usage: run_w3_queue.sh <gpu> <port> <job> [<job> ...]
#   job = bb:<hf_model>:<short>:<util>   FRAMES backbone replication: smoke 5 -> token probe 20 (tokens only) -> k* ->
#                                        final 624 (all arms, k*) -> S_UNION final 624 -> server down
#   job = hotdist                        HotpotQA W2 (Qwen2.5-14B): S_UNION, S_5, S_UNION_T7 on final 1,000 -> EM analysis
# Weights are copied from a shared HF cache ($SHARED_HF_HUB) into the private cache first (Mistral's duplicate
# consolidated.safetensors excluded). The GPU is used only after it has had enough free memory for 10 minutes; other
# processes are never touched. No retries; a failure stops this queue and keeps its logs.
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/w3_gpu$G.log; mkdir -p $W/logs $W/runs/backbones $W/runs/dist_hotpot
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}; SRC=${SHARED_HF_HUB:?set SHARED_HF_HUB}
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=$HUB WQ_BASE_URL=http://127.0.0.1:$PORT/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
commit () { flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com add "${@:2}" && \
  flock $W/.git/w3.lock git -C $W -c user.name=anon -c user.email=anon@example.com commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; }
fetch () {  # copy one model repo into the private cache, verifying blob sizes
  local m=models--${1/\//--}; [ -f $HUB/$m/.complete ] && return 0
  local skip=""; [ -L $SRC/$m/snapshots/*/consolidated.safetensors ] && skip=$(basename $(readlink $SRC/$m/snapshots/*/consolidated.safetensors))
  mkdir -p $HUB/$m/blobs; cp -a $SRC/$m/refs $SRC/$m/snapshots $HUB/$m/
  for b in $SRC/$m/blobs/*; do [ "$(basename $b)" = "$skip" ] && continue; cp -n $b $HUB/$m/blobs/; \
    [ $(stat -c %s $b) = $(stat -c %s $HUB/$m/blobs/$(basename $b)) ] || return 1; done
  [ -n "$skip" ] && rm -f $HUB/$m/snapshots/*/consolidated.safetensors
  touch $HUB/$m/.complete; }
gate () { local need=$1 ok=0
  while [ $ok -lt 20 ]; do
    f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$f" -ge "$need" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
  done; say "GPU $G has >= $need MiB free for 10 min"; }
serve () {  # model util
  local need=$(python3 -c "print(int($2*81559)+1024)")
  gate $need
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $1 --port $PORT --dtype bfloat16 --gpu-memory-utilization $2 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_w3_gpu$G.log 2>&1 &
  S=$!
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server $1 died"; done
  say "server $1 ready ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_w3_gpu$G.log | head -1))"; }
down () { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"; }
[ "$(git -C $W describe --tags --exact-match w3-freeze 2>/dev/null)" = w3-freeze ] || fail "tag w3-freeze missing"
cd $W/scripts; say "queue start: $*"
for job in "$@"; do
  if [ "$job" = hotdist ]; then
    export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct; D=$W/runs/dist_hotpot
    serve $WQ_MODEL 0.70
    for c in S_UNION S_5 S_UNION_T7; do $PY run_dist_hotpot.py --cond $c --out $D/${c}_final_gen.jsonl >> $LOG 2>&1 || fail "hotdist $c"; done
    down
    $PY analyze_dist_hotpot.py $D >> $LOG 2>&1 || fail "hotdist analysis"
    commit "W3a HotpotQA evidence distribution: generation + sealed EM analysis (run once)" runs/dist_hotpot
    say "hotdist DONE"; continue
  fi
  IFS=: read -r _ M SH U <<< "$job"; export WQ_MODEL=$M; D=$W/runs/backbones/$SH; mkdir -p $D
  say "[$SH] copying weights"; fetch $M || fail "[$SH] weight copy"
  serve $M $U
  $PY run_webqa.py --dataset frames --split pilot --n 5 --k 2 --out $D/frames_smoke.jsonl >> $LOG 2>&1 || fail "[$SH] smoke"
  $PY run_webqa.py --dataset frames --split pilot --n 20 --k 1 --out $D/frames_probe.jsonl >> $LOG 2>&1 || fail "[$SH] probe"
  $PY token_probe_bb.py $D/frames_probe.jsonl $D >> $LOG 2>&1 || fail "[$SH] token probe"
  K=$($PY -c "import json;print(json.load(open('$D/K_FROZEN.json'))['k_star'])"); say "[$SH] k* = $K"
  $PY run_webqa.py --dataset frames --split final --k $K --out $D/frames_final_gen.jsonl >> $LOG 2>&1 || fail "[$SH] final"
  $PY run_distribution.py --cond S_UNION --split final --out $D/S_UNION_final_gen.jsonl >> $LOG 2>&1 || fail "[$SH] S_UNION"
  down
  commit "W3b backbone $SH: smoke, probe, k*, FRAMES final + S_UNION generation" runs/backbones/$SH
  say "[$SH] generation DONE"
done
say "QUEUE DONE"
