#!/bin/bash
# Re-run ONE RX generation job against an already-running agent server (same command and checks as rx_queue.sh).
# Used when a queued job was interrupted for a reason unrelated to the job itself. Usage: rx_rerun_one.sh <port> <ds> <MODE>
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
export PYTHONPATH=$W/.pylib:$W/scripts WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:$1/v1
o=$W/runs/rx/${2}_${3}_gen.jsonl; cd $W/scripts
if $PY rx_run.py --dataset $2 --mode $3 --out $o; then
  flock $W/runs/rx/DONE.lock sh -c "echo '${o##*/} done' >> $W/runs/rx/DONE"; echo "$(date '+%m-%d %H:%M:%S') $2:$3 generated (rerun)"
else echo "$(date '+%m-%d %H:%M:%S') FAIL rerun $2:$3"; fi
