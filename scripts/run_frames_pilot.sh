#!/bin/bash
# FRAMES pilot chain (DESIGN_DRAFT.md §5). Waits for the Wikipedia snapshot fetch to finish, builds the corpus with the
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

say "waiting for the first Wikipedia fetch pass to finish"
until grep -q "^done:" $W/data/fetch.log; do sleep 60; done
say "fetch pass 1: $(grep '^done:' $W/data/fetch.log)"
cd $W/scripts
for pass in 2 3; do       # amendment W1: fixed link parsing adds the split/decoded titles; a further pass only re-tries fetch errors
  $PY fetch_wiki.py > $W/data/fetch_pass$pass.log 2>&1 || fail "fetch pass $pass"
  say "fetch pass $pass: $(grep '^done:' $W/data/fetch_pass$pass.log)"
  $PY verify_snapshot.py snapshot >> $LOG 2>&1 && break
  [ $pass = 3 ] && fail "snapshot verification (data/SNAPSHOT_VERIFY_snapshot.txt)"
done
say "snapshot verified"
$PY build_corpus.py > $W/logs/build_corpus.log 2>&1 || fail build_corpus
$PY verify_snapshot.py index >> $LOG 2>&1 || fail "BM25 index verification (data/SNAPSHOT_VERIFY_index.txt)"
say "BM25 index verified"
{ echo; echo "### Corpus artifacts (built $(date '+%Y-%m-%d %H:%M') by the frozen build_corpus.py, before any FRAMES generation)"; echo '```'; cat $W/logs/build_corpus.log; echo '```'; } >> $W/DESIGN_DRAFT.md
$GIT add DESIGN_DRAFT.md data/SNAPSHOT_VERIFY_*.txt data/wiki_manifest.jsonl data/partition.json data/retrieval_frames.jsonl logs/build_corpus.log && $GIT commit -q -m "FRAMES corpus built (frozen code); hashes appended to the freeze note

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say "corpus built; waiting for a GPU"
gate; serve $WQ_MODEL 8121
$PY run_webqa.py --dataset frames --split pilot --n 5 --k 2 --out $W/runs/frames_smoke.jsonl >> $LOG 2>&1 || fail smoke
$PY -c "import json;R=[json.loads(l) for l in open('$W/runs/frames_smoke.jsonl')];assert all(r['error'] is None for r in R);print('smoke answers parsed:',sum(r[x]['answer'] is not None for r in R for x in ('A0','B0','POOL','A1m','B1m','A1n','B1n')),'/',7*len(R))" >> $LOG 2>&1 || fail "smoke check"
$PY run_webqa.py --dataset frames --split pilot --n 20 --k 1 --out $W/runs/frames_probe.jsonl >> $LOG 2>&1 || fail probe
$PY token_probe.py frames $W/runs/frames_probe.jsonl >> $LOG 2>&1 || fail "token probe"
$GIT add runs/frames_smoke.jsonl runs/frames_probe.jsonl data/K_FROZEN_frames.json && $GIT commit -q -m "FRAMES smoke + token probe (tokens only); K frozen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
K=$($PY -c "import json;print(json.load(open('$W/data/K_FROZEN_frames.json'))['k_star'])")
say "k* = $K; pilot generation starting"
$PY run_webqa.py --dataset frames --split pilot --k $K --out $W/runs/frames_pilot_gen.jsonl >> $LOG 2>&1 || fail pilot
down
$GIT add runs/frames_pilot_gen.jsonl && $GIT commit -q -m "FRAMES pilot generation (200)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
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
