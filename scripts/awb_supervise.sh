#!/bin/bash
# Keep the three AgentWebBench variants (full / docs / summary) running until each has 53 results. A variant whose process
# died (e.g. its 14B server was swapped out by an RX queue) is restarted against whichever 14B server is up; AgentWebBench
# skips questions that already have a result file, and a failed question writes none, so restarts are safe.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; cd $W
# a variant restarted 3 times in a row with no new result is given up (questions that fail every pass, e.g. persistent
# empty model output or search-API errors) and reported as missing.
declare -A last stall
while :; do
  left=0
  for v in full docs summary; do
    d=multi_agent; pf=runs/awb/PID; lg=logs/awb_multi_agent.log
    [ $v != full ] && { d=multi_agent_$v; pf=runs/awb/PID_$v; lg=logs/awb_multi_agent_$v.log; }
    n=$(ls runs/awb/results/$d 2>/dev/null | grep -c '^result_')
    [ "$n" -ge 53 ] && continue
    [ "${stall[$v]:-0}" -ge 3 ] && continue
    left=1
    pid=$(cat $pf 2>/dev/null); [ -n "$pid" ] && kill -0 $pid 2>/dev/null && continue
    ports="8161 8171"; [ $v = full ] && ports="8171 8161"     # spread the load: full on GPU1, docs/summary on GPU0
    for p in $ports; do curl -sf http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q Qwen2.5-14B && {
      [ "${last[$v]:-x}" = "$n" ] && stall[$v]=$(( ${stall[$v]:-0} + 1 )) || stall[$v]=0; last[$v]=$n
      [ "${stall[$v]}" -ge 3 ] && { echo "=== $(date '+%m-%d %H:%M:%S') GIVE UP $v at $n/53 (3 passes without progress) ===" | tee -a $lg; break; }
      echo "=== $(date '+%m-%d %H:%M:%S') supervisor restart $v on $p ($n/53 done) ===" | tee -a $lg
      AWB_MSG=$v nohup bash scripts/awb_run.sh multi_agent $p >> $lg 2>&1 & echo $! > $pf; break; }; done
  done
  [ $left = 0 ] && { echo "$(date '+%m-%d %H:%M:%S') AWB ALL VARIANTS COMPLETE (or given up)"; exit 0; }
  sleep 60
done
