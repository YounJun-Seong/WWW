#!/bin/bash
# BX4 runs (DESIGN_BX4.md, tag bx4-freeze).  bx4_queue.sh <gpu> <port> <job>...   job = gen:<ds>:<MODE> | suff:<ds> | judge
# env UTIL_AGENT / UTIL_JUDGE (default 0.70): vLLM memory fraction; lowered only when other users occupy the GPU.
G=$1; PORT=$2; shift 2
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/bx4_gpu$G.log; O=$W/runs/bx4; mkdir -p $O
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$PORT/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:$((PORT+1))/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
S=""; CUR=""
stop () { [ -n "$S" ] && { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"; }; S=""; CUR=""; }
waitgpu () { need=$(python3 -c "print(int(81559*$1)+2000)"); ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge "$need" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
up () { [ "$CUR" = "$1" ] && return; stop
  local mdl p dt u ml
  if [ $1 = agent ]; then mdl=$WQ_MODEL; p=$PORT; dt=bfloat16; u=${UTIL_AGENT:-0.70}; ml=${MAXLEN_AGENT:-32768}
  else mdl=$WQ_JUDGE_MODEL; p=$((PORT+1)); dt=auto; u=${UTIL_JUDGE:-0.70}; ml=${MAXLEN_JUDGE:-32768}; fi
  local extra=""; [ $1 = judge ] && extra=${VLLM_EXTRA_JUDGE:-}
  waitgpu $u; say "starting $1 server, gpu-memory-utilization $u, max-model-len $ml $([ $1 = judge ] && echo ${VLLM_EXTRA_JUDGE:-})"
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $mdl --port $p --dtype $dt --gpu-memory-utilization $u \
    --max-model-len $ml --generation-config vllm --disable-log-requests $extra > $W/logs/vllm_bx4_$1_g$G.log 2>&1 &
  S=$!; CUR=$1; until curl -sf http://127.0.0.1:$p/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || { say "FAIL server"; exit 1; }; done
  say "server ready $mdl ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_bx4_$1_g$G.log | head -1))"; }
run () { local o=$1; shift
  if $PY bx4_run.py "$@" --n 5 --out $o.smoke >> $LOG 2>&1 && $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and all(r['error'] is None for r in R)" $o.smoke; then
    rm -f $o.smoke; $PY bx4_run.py "$@" --out $o >> $LOG 2>&1 && say "done ${o##*/}" || say "FAIL ${o##*/}"
  else say "FAIL smoke ${o##*/}"; fi; }
[ -n "$(git -C $W tag --contains bx4-freeze 2>/dev/null)" ] || { say "FAIL tag bx4-freeze missing"; exit 1; }
cd $W/scripts; say "queue start: $*"
for job in "$@"; do
  IFS=: read -r t ds m <<< "$job"
  case $t in
    gen)   up agent; run $O/${ds}_${m}_gen.jsonl --dataset $ds --mode $m ;;
    suff)  up judge; run $O/suff_${ds}.jsonl --dataset $ds --mode SUFF ;;
    judge) up judge; for f in $O/frames_*_gen.jsonl; do j=${f%_gen.jsonl}_judged.jsonl; [ -s "$j" ] && continue
             $PY rx_judge.py --gen $f --out $j >> $LOG 2>&1 && say "judged ${f##*/}" || say "FAIL judge ${f##*/}"; done ;;
  esac
done
stop; say "QUEUE COMPLETE"
