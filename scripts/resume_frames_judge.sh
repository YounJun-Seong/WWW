#!/bin/bash
# FRAMES pilot, judge stage onward (resume after the judge server failed to start: refs/main missing in the private HF cache; no judge output existed).
# frozen build_corpus.py (hashes appended to the freeze note), then waits for a free GPU (>= 60 GB for 10 min; never
# touches other processes): agent server -> smoke (5) -> token probe (20, tokens only) -> K freeze -> pilot (200)
# -> agent server down -> judge server (Qwen3-32B-FP8) -> judge -> server down -> human sheet -> sealed analysis.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/frames_pilot.log; mkdir -p $W/logs $W/runs
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:8121/v1
export WQ_JUDGE_MODEL=Qwen/Qwen3-32B-FP8 WQ_JUDGE_URL=http://127.0.0.1:8122/v1
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
fail () { say "FAIL: $1"; [ -n "$S" ] && kill $S 2>/dev/null; exit 1; }
gate () { local ok=0; G=""
  while [ $ok -lt 20 ]; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
    if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
  done; say "GPU $G free for 10 min"; }
serve () {  # $1 model $2 port
  CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $1 --port $2 --dtype auto --gpu-memory-utilization 0.70 \
    --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_frames_$2.log 2>&1 &
  S=$!
  until curl -sf http://127.0.0.1:$2/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || fail "server $1 died"; done
  say "server $1 ready on GPU $G"; }
down () { kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; S=""; say "server down"; }
cd $W/scripts; say "resume: judge stage (refs/main restored for Qwen3-32B-FP8)"
gate; serve $WQ_JUDGE_MODEL 8122
$PY judge.py --gen $W/runs/frames_smoke.jsonl --out $W/runs/frames_smoke_judged.jsonl >> $LOG 2>&1 || fail "judge smoke"
$PY -c "import json;R=[json.loads(l) for l in open('$W/runs/frames_smoke_judged.jsonl')];raw=[v['raw'] for r in R for k,v in r.items() if isinstance(v,dict)];bad=[x for x in raw if x.upper() not in ('TRUE','FALSE','(EMPTY RESPONSE)')];print('judge format ok' if not bad else 'judge format issues', bad[:5]);assert not bad" >> $LOG 2>&1 || fail "judge format"
$PY judge.py --gen $W/runs/frames_pilot_gen.jsonl --out $W/runs/frames_pilot_judged.jsonl >> $LOG 2>&1 || fail judge
down
$PY human_sheet.py --judged $W/runs/frames_pilot_judged.jsonl >> $LOG 2>&1 || fail "human sheet"
$GIT add runs/frames_smoke_judged.jsonl runs/frames_pilot_judged.jsonl data/human_validation_sheet.csv data/human_validation_key.json && $GIT commit -q -m "FRAMES pilot judged; human validation sheet (50, stratified)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
WQ_OUT=$W/runs $PY analyze.py frames pilot $W/runs/frames_pilot_gen.jsonl $W/runs/frames_pilot_judged.jsonl >> $LOG 2>&1 || fail analysis
$PY run_record.py frames_pilot frames pilot $W/runs/frames_pilot_gen.jsonl $W/runs/RESULTS_frames_pilot.txt $LOG judge >> $LOG 2>&1 || fail "run record"
$GIT add runs/RESULTS_frames_pilot.* runs/RUN_RECORD_frames_pilot.json && $GIT commit -q -m "FRAMES pilot sealed analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/RESULTS_frames_pilot.txt"
