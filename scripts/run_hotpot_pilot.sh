#!/bin/bash
# HotpotQA pilot (DESIGN_DRAFT.md §5), one GPU. Waits until some GPU has >= 60 GB free for 10 min; never touches other
# processes. smoke (5) -> token probe (20, tokens only) -> K freeze -> pilot (500) -> sealed analysis. Stops on failure.
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; PY=${PYTHON:-python3}
VPY=${VLLM_BIN:-$(dirname "$(command -v vllm)")}; LOG=$W/logs/hotpot_pilot.log; mkdir -p $W/logs $W/runs
export PYTHONPATH=$W/.pylib:$W/scripts HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WQ_MODEL=Qwen/Qwen2.5-14B-Instruct WQ_BASE_URL=http://127.0.0.1:8111/v1
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
GIT="git -C $W -c user.name=anon -c user.email=anon@example.com"
say "waiting for a GPU"
ok=0; G=-1
while [ $ok -lt 20 ]; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=61440{print $1; exit}')
  if [ -n "$G" ]; then ok=$((ok+1)); else ok=0; fi; sleep 30
done
say "GPU $G free for 10 min"
CUDA_VISIBLE_DEVICES=$G nohup $VPY/vllm serve $WQ_MODEL --port 8111 --dtype bfloat16 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --generation-config vllm --disable-log-requests > $W/logs/vllm_hotpot.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8111/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $S 2>/dev/null || { say "FAIL: server died"; exit 1; }; done
say "server ready"
cd $W/scripts
fail () { say "FAIL: $1"; kill $S; exit 1; }
$PY run_webqa.py --dataset hotpot --split pilot --n 5 --k 2 --out $W/runs/hotpot_smoke.jsonl >> $LOG 2>&1 || fail smoke
$PY -c "import json;R=[json.loads(l) for l in open('$W/runs/hotpot_smoke.jsonl')];assert all(r['error'] is None for r in R);print('smoke answers parsed:',sum(r[x]['answer'] is not None for r in R for x in ('A0','B0','POOL','A1m','B1m','A1n','B1n')),'/',7*len(R))" >> $LOG 2>&1 || fail "smoke check"
$PY run_webqa.py --dataset hotpot --split pilot --n 20 --k 1 --out $W/runs/hotpot_probe.jsonl >> $LOG 2>&1 || fail probe
$PY token_probe.py hotpot $W/runs/hotpot_probe.jsonl >> $LOG 2>&1 || fail "token probe"
$GIT add runs/hotpot_smoke.jsonl runs/hotpot_probe.jsonl data/K_FROZEN_hotpot.json && $GIT commit -q -m "HotpotQA smoke + token probe (tokens only); K frozen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
K=$($PY -c "import json;print(json.load(open('$W/data/K_FROZEN_hotpot.json'))['k_star'])")
say "k* = $K; pilot starting"
$PY run_webqa.py --dataset hotpot --split pilot --k $K --out $W/runs/hotpot_pilot_gen.jsonl >> $LOG 2>&1 || fail pilot
kill $S; while kill -0 $S 2>/dev/null; do sleep 3; done; say "server down"
$GIT add runs/hotpot_pilot_gen.jsonl && $GIT commit -q -m "HotpotQA pilot generation (500)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
WQ_OUT=$W/runs $PY analyze.py hotpot pilot $W/runs/hotpot_pilot_gen.jsonl >> $LOG 2>&1 || { say "FAIL: analysis"; exit 1; }
$GIT add runs/RESULTS_hotpot_pilot.* && $GIT commit -q -m "HotpotQA pilot sealed analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "DONE: runs/RESULTS_hotpot_pilot.txt"
