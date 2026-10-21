#!/bin/bash
# RX Exp 9 — AgentWebBench QA (53) external diagnostic transfer. Runs AgentWebBench's own code (commit in external/AWB_COMMIT,
# only change: CONCURRENT_NUM 32 -> 8) with our agent model (Qwen2.5-14B-Instruct) served locally, against whichever RX agent
# server is up. Methods: classical (centralized retrieval) and multi_agent (website content agents). Judging is done
# separately (awb_eval.py): AgentWebBench's judge prompt, but our local judge model, because no OpenAI key is available.
#   awb_run.sh <method> [port]      env AWB_MSG=docs|summary -> message-content variant, outputs under <method>_<AWB_MSG>
W=${WQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$W/external/AgentWebBench; PY=${PYTHON:-python3}
M=$1; P=${2:-}; D=$M; [ "${AWB_MSG:-full}" != full ] && D=${M}_$AWB_MSG
if [ -z "$P" ]; then for p in 8171 8161; do curl -sf http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q Qwen2.5-14B && { P=$p; break; }; done; fi
[ -n "$P" ] || { echo "no agent server up"; exit 1; }
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export PYTHONPATH=${AWB_PYLIB}:$W/.pylib:$A PYTHONUNBUFFERED=1 AGENTIC_WEB_SAVE_PREFIX=$W/runs/awb
cd $A
$PY $W/scripts/awb_main_wrapper.py --batch_file data/qa/test_53.json --answer_dir $W/runs/awb/results/$D --log_dir $W/runs/awb/logs/$D \
  --method $M --task_type qa --api_type local --model_id Qwen/Qwen2.5-14B-Instruct --url http://127.0.0.1:$P/v1
