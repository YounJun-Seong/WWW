"""RX Exp 9 — AgentWebBench QA (53) external DIAGNOSTIC transfer: scoring + evidence trace. Not a replication of the
two-agent protocol (different architecture, n = 53), and only the multi_agent method could run: the public search API now
requires a `website` parameter, so AgentWebBench's centralized `classical` method is unavailable.

Scoring: (1) token F1 (SQuAD normalisation); (2) AgentWebBench's own QA judge prompt (judge_prompt_mhqa) run with OUR
local judge model (Qwen3-32B-FP8, T=0, thinking off) because no OpenAI key is available — a stated deviation.
Trace (descriptive, not causal), per question, using the dataset's `seen_doc_ids` (documents seen while the question was
built — a noisy relevant set, NOT verified answer support):
  searched     some content agent's search results contained a seen document
  transmitted  some content agent returned a seen document to the user agent ("Retrieved document <id>")
  correct      the final answer is judged correct
  python awb_eval.py [multi_agent|multi_agent_docs|multi_agent_summary]   (env WQ_JUDGE_URL, WQ_JUDGE_MODEL)
      ->  runs/awb/RESULTS_awb[_<variant>].{txt,json}
In the `summary` (claim-only) variant documents are never transmitted, so `transmitted` is forced False there."""
import glob
import json
import os
import re
import sys

ROOT = os.environ.get("WQ_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
A = f"{ROOT}/external/AgentWebBench"
sys.path.insert(0, f"{A}/awbench/evaluation/qa"); sys.path.insert(0, f"{ROOT}/scripts")
from common import chat, f1 as tok_f1  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "multi_agent"
SUF = "" if M == "multi_agent" else M[len("multi_agent"):]
data = {x["id"]: x for x in json.load(open(f"{A}/data/qa/test_53.json"))}
res_files = {int(re.search(r"result_(\d+)\.json$", f).group(1)): f for f in glob.glob(f"{ROOT}/runs/awb/results/{M}/result_*.json")}


def judge(q, gold, pred):
    """AgentWebBench's judge_prompt_mhqa, parsed as AgentWebBench parses it (JSON `judgement`, else keyword fallback)."""
    from prompt import judge_prompt_mhqa
    p = judge_prompt_mhqa.format(question=q, gt_answer=gold, pred_answer=pred)
    c = chat([{"role": "user", "content": p}], 0.0, 512, 0, base_url=os.environ["WQ_JUDGE_URL"],
             model=os.environ["WQ_JUDGE_MODEL"], extra={"chat_template_kwargs": {"enable_thinking": False}})
    t = c["texts"][0]
    m = re.search(r'"judgement"\s*:\s*"(correct|incorrect)', t, re.I)
    if m:
        return m.group(1).lower() == "correct", t
    lo = t.lower()
    if "incorrect" in lo:
        return False, t
    return ("correct" in lo) or None, t


def trace(qid, seen):
    searched = transmitted = False
    for f in glob.glob(f"{ROOT}/runs/awb/logs/{M}/content_agent_*_{qid}_*.log"):
        s = open(f, errors="ignore").read()
        ids_found = set(re.findall(r'"doc_id":\s*"(clueweb22-[^"]+)"', s))
        ids_sent = set(re.findall(r"Retrieved document (clueweb22-\S+)", s))
        searched |= bool(ids_found & seen); transmitted |= bool(ids_sent & seen) and not M.endswith("_summary")
    return searched, transmitted


rows = []
for qid, x in sorted(data.items()):
    r = json.load(open(res_files[qid])) if qid in res_files else None
    pred = (r or {}).get("answer") or ""
    v, raw = judge(x["question"], x["answer"], pred) if (r and os.environ.get("WQ_JUDGE_URL")) else (None, "")
    s, t = trace(qid, set(x["seen_doc_ids"]))
    rows.append(dict(id=qid, source=x["source"], has_result=r is not None, f1=tok_f1(pred, x["answer"]), judged=v,
                     searched_seen=s, transmitted_seen=t, turns=(r or {}).get("turns"), search_count=(r or {}).get("search count")))
n = len(rows); have = [r for r in rows if r["has_result"]]; jd = [r for r in have if r["judged"] is not None]
acc = sum(r["judged"] for r in jd) / len(jd) if jd else None
out = dict(method=M, n=n, with_result=len(have), judged=len(jd), accuracy_judge=acc,
           f1=sum(r["f1"] for r in have) / max(1, len(have)),
           searched_seen=sum(r["searched_seen"] for r in have), transmitted_seen=sum(r["transmitted_seen"] for r in have),
           correct_given_transmitted=(sum(1 for r in jd if r["transmitted_seen"] and r["judged"]) / max(1, sum(1 for r in jd if r["transmitted_seen"]))),
           correct_given_not_transmitted=(sum(1 for r in jd if not r["transmitted_seen"] and r["judged"]) / max(1, sum(1 for r in jd if not r["transmitted_seen"]))),
           by_source={s: dict(n=sum(1 for r in have if r["source"] == s), correct=sum(1 for r in jd if r["source"] == s and r["judged"]))
                      for s in sorted({r["source"] for r in rows})},
           accuracy_all53_missing_as_wrong=(sum(1 for r in jd if r["judged"]) / n if jd else None),
           deviations=["questions without a result (AgentWebBench's own prompt truncation raises on malformed document JSON, "
                       "or search-API errors, on every retry) are reported as missing and counted wrong in accuracy_all53","centralized `classical` method unavailable: public search API requires a website parameter",
                       "judge = AgentWebBench judge prompt with our local Qwen3-32B-FP8, not their OpenAI judge",
                       "AgentWebBench CONCURRENT_NUM 32 -> 8; Qwen2.5-14B-Instruct max input 28000 tokens"],
           rows=rows)
L = [f"=== RX Exp 9 AgentWebBench QA external diagnostic, method {M}, Qwen2.5-14B-Instruct (POST-HOC; n={n}) ===",
     f"results {len(have)}/{n}; judged {len(jd)}; accuracy (judge) {acc if acc is None else round(acc, 3)} on answered, "
     f"{out['accuracy_all53_missing_as_wrong'] if acc is None else round(out['accuracy_all53_missing_as_wrong'], 3)} on all {n} (missing = wrong); token F1 {out['f1']:.3f}",
     f"seen documents reached a content agent's search results: {out['searched_seen']}/{len(have)}; returned to the user agent: {out['transmitted_seen']}/{len(have)}",
     f"P(correct | seen doc transmitted) {out['correct_given_transmitted']:.3f}; P(correct | not transmitted) {out['correct_given_not_transmitted']:.3f}",
     "by source: " + ", ".join(f"{s} {v['correct']}/{v['n']}" for s, v in out["by_source"].items()),
     *[f"deviation: {d}" for d in out["deviations"]]]
json.dump(out, open(f"{ROOT}/runs/awb/RESULTS_awb{SUF}.json", "w"), indent=1)
open(f"{ROOT}/runs/awb/RESULTS_awb{SUF}.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
