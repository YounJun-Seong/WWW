#!/bin/bash
# End-of-run chain (replaces free-memory waiters, which could race a queue's own server swap):
#   GPU1: after queue1b (pid $1) exits -> rx_queue.sh 1 8171 <jobs...>
#   then after the GPU0 queue (pid $2) AND the GPU1 chain exit -> rx_queue_32b.sh -> awb_finish.sh
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; cd $W
Q1=$1; Q0=$2; shift 2
while kill -0 $Q1 2>/dev/null; do sleep 20; done
bash scripts/rx_queue.sh 1 8171 "$@" > logs/rx_queue1d.out 2>&1 &
C1=$!
while kill -0 $Q0 2>/dev/null || kill -0 $C1 2>/dev/null; do sleep 20; done
bash scripts/rx_queue_32b.sh > logs/rx_queue_32b.out 2>&1 &
Q32=$!; echo $Q32 > runs/rx/PID_32b
bash scripts/awb_finish.sh $Q32 > logs/awb_finish.out 2>&1
