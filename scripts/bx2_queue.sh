#!/bin/bash
# BX2 runs (DESIGN_BX2.md, tag bx2-freeze): B ownership-framing control, then C latency audit on dedicated servers.
# GPUs used only after >= 58 GB free for 2 min; other users' processes are never touched; servers up only while needed.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/bx2.log; O=$W/runs/bx2; mkdir -p $O $W/logs
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} WQ_MODEL=Qwen/Qwen2.5-14B-Instruct
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
waitgpu () { ok=0; while [ $ok -lt 4 ]; do f=$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$f" -ge 58000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 30; done; }
serve () { local g=$1 p=$2; shift 2; waitgpu $g
  CUDA_VISIBLE_DEVICES=$g nohup $VPY/vllm serve $WQ_MODEL --port $p --dtype bfloat16 --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests "$@" > $W/logs/vllm_bx2_g${g}_p$p.log 2>&1 &
  local s=$!; until curl -sf http://127.0.0.1:$p/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $s 2>/dev/null || { say "FAIL server $g:$p died"; exit 1; }; done
  say "server $g:$p ready $* ($(grep -o 'snapshots/[0-9a-f]*' $W/logs/vllm_bx2_g${g}_p$p.log | head -1))"; echo $s; }
stop () { kill $1; while kill -0 $1 2>/dev/null; do sleep 3; done; say "server pid $1 down"; }
[ -n "$(git -C $W tag --contains bx2-freeze 2>/dev/null)" ] || { say "FAIL tag bx2-freeze missing"; exit 1; }
cd $W/scripts; say "queue start"
# B: ownership-framing control (standard server settings, shared frozen cache)
S=$(serve 1 8171 | tail -1); export WQ_BASE_URL=http://127.0.0.1:8171/v1
if $PY bx2_run.py --n 5 --out $O/smoke_F_SPLIT_LOCAL.jsonl >> $LOG 2>&1 && $PY bx2_run.py --out $O/hotpot_F_SPLIT_LOCAL_gen.jsonl >> $LOG 2>&1; then
  say "B F_SPLIT_LOCAL generated"; else say "FAIL B F_SPLIT_LOCAL"; fi
stop $S
# C: latency on dedicated servers, prefix caching disabled, fresh cache per (rep, part)
S0=$(serve 0 8261 --no-enable-prefix-caching | tail -1); S1=$(serve 1 8271 --no-enable-prefix-caching | tail -1)
fresh () { d=$O/tmpcache/$1; rm -rf $d; mkdir -p $d; echo $d; }
WQ_CACHE=$(fresh smoke) WQ_BASE_URL=http://127.0.0.1:8261/v1 $PY bx2_latency.py --cond loaded --rep 0 --nq 2 --out $O/smoke_lat.jsonl >> $LOG 2>&1 \
  || { say "FAIL latency smoke"; stop $S0; stop $S1; exit 1; }
say "latency smoke ok"
for r in 1 2 3; do
  WQ_CACHE=$(fresh iso_r${r}_p0) WQ_BASE_URL=http://127.0.0.1:8261/v1 BX2_GPU=0 $PY bx2_latency.py --cond isolated --rep $r --part 0 --out $O/lat_isolated_r${r}_p0.jsonl >> $LOG 2>&1 &
  a=$!
  WQ_CACHE=$(fresh iso_r${r}_p1) WQ_BASE_URL=http://127.0.0.1:8271/v1 BX2_GPU=1 $PY bx2_latency.py --cond isolated --rep $r --part 1 --out $O/lat_isolated_r${r}_p1.jsonl >> $LOG 2>&1 &
  b=$!
  wait $a && wait $b && say "C isolated rep $r done" || say "FAIL C isolated rep $r"
done
stop $S1
for r in 1 2 3; do
  WQ_CACHE=$(fresh load_r${r}) WQ_BASE_URL=http://127.0.0.1:8261/v1 BX2_GPU=0 $PY bx2_latency.py --cond loaded --rep $r --inflight 16 --out $O/lat_loaded_r${r}_all.jsonl >> $LOG 2>&1 \
    && say "C loaded rep $r done" || say "FAIL C loaded rep $r"
done
stop $S0; rm -rf $O/tmpcache; say "BX2 QUEUE COMPLETE"
