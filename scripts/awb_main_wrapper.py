"""Run AgentWebBench's unmodified main.py after loading its tokenizer once in the main thread. AgentWebBench loads the
tokenizer lazily inside worker threads; concurrent first imports of transformers raced ("cannot import name
AutoTokenizer"). Pre-loading removes the race without changing any AgentWebBench logic.

RX message-content variants (env AWB_MSG; default `full` = AgentWebBench unchanged). A content agent's reply to the user
agent is {"website", "summary", "documents"}; the variants strip one field after the content agent has finished, so the
content agent's own search/selection is identical and only what is TRANSMITTED changes:
  docs     evidence-only: retrieved documents, no summary (claim) — the analogue of our EV message
  summary  claim-only: the content agent's summary, no documents — the analogue of an answer/claim message

Context fix (applied 10-15 ~17:50, all variants): AgentWebBench requests a fixed max_tokens 8192 while allowing up to 28000
input tokens, which overflows Qwen2.5-14B's 32768 context (HTTP 400 on every retry -> the question can never finish).
max_tokens is capped at 32768 - input - 256. Calls whose input was <= 24320 tokens (all calls of every question that
completed before the fix, since any longer call failed) get exactly the original 8192.

Search-API fix (applied 10-15 ~19:25, all variants): the public ClueWeb22 search API now answers HTTP 404 for some
website/query pairs (e.g. websites it does not index); AgentWebBench does not catch it (content-agent futures are
.result()-ed without a handler), so one 404 kills the whole question on every pass. A 404 is returned as an empty result
list ("no documents"). Other HTTP errors still raise. Questions completed before the fix never hit a 404 (it was fatal),
so their trajectories are unaffected; runs/awb/FIX_TIMES records when each fix went in."""
import os
import runpy
import sys

from awbench.utils import token_calculator

token_calculator._get_tokenizer()
from awbench.agents import content_agent as _ca, llm_client as _lc, user_agent as _ua  # noqa: E402
_call = _lc.call_llm


def call_llm(api_type, client, model_name, prompt, max_tokens=_lc.MAX_OUTPUT_TOKENS, **kw):
    n = len(token_calculator.tokenize_ids(_lc._truncate_prompt_if_needed(prompt, model_name)))
    return _call(api_type, client, model_name, prompt, max_tokens=max(256, min(max_tokens, 32768 - n - 256)), **kw)
for _m in (_lc, _ca, _ua):
    _m.call_llm = call_llm

import requests  # noqa: E402
from awbench import websites as _ws  # noqa: E402
from awbench.utils import awbench_api as _api  # noqa: E402
_search = _api.search


def search(query, k, website=None):
    try:
        return _search(query, k, website=website)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"[RX] search 404 -> empty result (website={website})", flush=True)
            return []
        raise
_api.search = search
for _m in (_ca, _ws):
    _m.awbench_search = search
MSG = os.environ.get("AWB_MSG", "full")
if MSG != "full":
    from awbench.agents import content_agent
    _orig = content_agent.ContentAgent.retrieve_documents

    def retrieve_documents(self, *a, **k):
        r = _orig(self, *a, **k)
        if MSG == "docs":
            r.pop("summary", None)
            if not r.get("documents"):
                r["documents"] = []
        elif MSG == "summary":
            r.pop("documents", None)
        return r
    content_agent.ContentAgent.retrieve_documents = retrieve_documents
    print(f"[RX] AWB_MSG={MSG}: content-agent replies stripped accordingly", flush=True)
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
