#!/bin/bash
# BX2-C resumed after the uncached-client fix (DESIGN_BX2.md deviations log), on the already-running dedicated servers.
#   bx2_queue_c.sh <server0_pid:8261> <server1_pid:8271>
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}; LOG=$W/logs/bx2.log; O=$W/runs/bx2
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub} WQ_MODEL=Qwen/Qwen2.5-14B-Instruct
export WQ_CACHE=$O/tmpcache/unused; mkdir -p $WQ_CACHE
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
stop () { kill $1; while kill -0 $1 2>/dev/null; do sleep 3; done; say "server pid $1 down"; }
S0=$1; S1=$2; cd $W/scripts; say "C resumed (uncached client)"
WQ_BASE_URL=http://127.0.0.1:8261/v1 $PY bx2_latency.py --cond loaded --rep 0 --nq 2 --out $O/smoke_lat.jsonl >> $LOG 2>&1 \
  && $PY -c "import json,sys;R=[json.loads(l) for l in open(sys.argv[1])];assert R and not any(c['hit'] for r in R for c in r['calls']) and all(c['wall']>0.05 for r in R for c in r['calls'])" $O/smoke_lat.jsonl \
  || { say "FAIL latency smoke (uncached)"; exit 1; }
say "latency smoke ok (no hits, every call reached the server)"
for r in 1 2 3; do
  WQ_BASE_URL=http://127.0.0.1:8261/v1 BX2_GPU=0 $PY bx2_latency.py --cond isolated --rep $r --part 0 --out $O/lat_isolated_r${r}_p0.jsonl >> $LOG 2>&1 & a=$!
  WQ_BASE_URL=http://127.0.0.1:8271/v1 BX2_GPU=1 $PY bx2_latency.py --cond isolated --rep $r --part 1 --out $O/lat_isolated_r${r}_p1.jsonl >> $LOG 2>&1 & b=$!
  wait $a && wait $b && say "C isolated rep $r done" || say "FAIL C isolated rep $r"
done
stop $S1
for r in 1 2 3; do
  WQ_BASE_URL=http://127.0.0.1:8261/v1 BX2_GPU=0 $PY bx2_latency.py --cond loaded --rep $r --inflight 16 --out $O/lat_loaded_r${r}_all.jsonl >> $LOG 2>&1 \
    && say "C loaded rep $r done" || say "FAIL C loaded rep $r"
done
stop $S0; rm -rf $O/tmpcache; say "BX2 QUEUE COMPLETE"
