#!/bin/bash
# Run every RX analysis whose inputs exist (generation done, and for FRAMES judged). Missing inputs -> skipped, listed.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
export PYTHONPATH=$W/.pylib:$W/scripts; cd $W/scripts
R=$W/runs/rx
have () { local ds=$1; shift; for m in "$@"; do [ -s $R/${ds}_${m}_gen.jsonl ] || return 1
  [ $ds = frames ] && { [ -s $R/${ds}_${m}_judged.jsonl ] || return 1; }; done; return 0; }
run () { echo "################ $*"; $PY rx_analyze.py "$@" 2>&1 | tail -40; }
for ds in frames hotpot; do
  have $ds R1_FULL R1_GOLDRM R1_GOLDREP R1_NONGOLDRM && run r1 $ds || echo "skip r1 $ds"
  have $ds R2_I1a R2_I1b R2_I2 R2_I3 R2_I4 && run r2 $ds || echo "skip r2 $ds"
  have $ds DPAD && { [ $ds = frames ] || have $ds OVL1 OVL2 OVL3 OVL4 OVL5; } && run r4 $ds || echo "skip r4 $ds"
  have $ds SEED2028_D SEED2028_SU && run r5 $ds || echo "skip r5 $ds"
  have $ds UNI && run r6 $ds || echo "skip r6 $ds"
  have $ds R1_FULL R10_HARDNEG R10_RANDNEG && { [ $ds = frames ] || have $ds R10_SHORT; } && run r10 $ds || echo "skip r10 $ds"
  have $ds AGG && run r11 $ds || echo "skip r11 $ds"
  have $ds R1_FULL R12_N3 R12_N6 R12_GOLDLAST && run r12 $ds || echo "skip r12 $ds"
done
have hotpot E2H_EXCL_LOW E2H_EXCL_HIGH E2H_SHAR_LOW E2H_SHAR_HIGH && run r3 hotpot || echo "skip r3 hotpot"
