#!/bin/bash
# BX3 execution-robustness regeneration (DESIGN_BX3.md, tag bx3-freeze). Usage: bx3_queue.sh <gpu> <port> <job> ...
#   job = rx:<ds>:<MODE> (rx_run.py) | bx:<ds>:<MODE> (bx_run.py) | judge (FRAMES files in runs/bx3, frozen judge + shared judge cache)
# Every generation job (and its smoke) gets its own fresh, empty cache dir. GPU used only after >= 58 GB free for 2 min.
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/bx3_gpu$G.log; O=$W/runs/bx3; mkdir -p $O $W/logs
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:$((PORT+1))/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
S=""
stop () { [ -n "$S" ] && { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"; }; S=""; }
waitgpu () { ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
up () { local model=$1 port=$2 lg=$3; stop; waitgpu
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $model --port $port --dtype ${4:-bfloat16} --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $lg 2>&1 &
  S=$!; until curl -sf http://127.0.0.1:$port/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || { say "FAIL server died"; exit 1; }; done
  say "server ready $model ($(grep -o 'snapshots/[0-9a-f]*' $lg | head -1))"; }
fresh () { d=$O/tmpcache/$1; rm -rf $d; mkdir -p $d; echo $d; }
[ -n "$(git -C $W tag --contains bx3-freeze 2>/dev/null)" ] || { say "FAIL tag bx3-freeze missing"; exit 1; }
cd $W/scripts; say "queue start: $*"; MODE_SRV=""
for job in "$@"; do
  if [ "$job" = judge ]; then
    up Qwen/Qwen3-32B-FP8 $((PORT+1)) $W/logs/vllm_bx3_judge_g$G.log auto; MODE_SRV=judge
    for f in $O/frames_*_gen.jsonl; do j=${f%_gen.jsonl}_judged.jsonl; [ -s "$j" ] && continue
      $PY rx_judge.py --gen $f --out $j >> $LOG 2>&1 && say "judged ${f##*/}" || say "FAIL judge ${f##*/}"; done
    stop; continue
  fi
  IFS=: read -r kind ds m <<< "$job"; SCRIPT=$([ $kind = rx ] && echo rx_run.py || echo bx_run.py)
  [ "$MODE_SRV" = agent ] || { up $WQ_MODEL $PORT $W/logs/vllm_bx3_g$G.log; MODE_SRV=agent; }
  o=$O/${ds}_${m}_gen.jsonl; tag=${ds}_${m}
  ( while :; do echo "$(date +%s) $(curl -s localhost:$PORT/metrics | grep -E '^vllm:num_requests_(running|waiting)' | awk '{print $2}' | tr '\n' ' ')"; sleep 5; done ) > $O/inflight_${tag}.log 2>&1 &
  SAMP=$!
  if WQ_CACHE=$(fresh smoke_$tag) $PY $SCRIPT --dataset $ds --mode $m --n 5 --out $o.smoke >> $LOG 2>&1 && \
     $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $o.smoke; then
    rm -f $o.smoke
    if WQ_CACHE=$(fresh $tag) $PY $SCRIPT --dataset $ds --mode $m --out $o >> $LOG 2>&1; then
      say "$ds:$m regenerated ($(ls $O/tmpcache/$tag | wc -l) fresh calls)"; else say "FAIL $ds:$m"; fi
  else say "FAIL smoke $ds:$m"; fi
  kill $SAMP 2>/dev/null
done
stop; say "QUEUE COMPLETE"
