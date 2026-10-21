"""Per-question evidence log for HotpotQA (descriptive; separate from the sealed analysis, which it never changes).
Fixed before the final run (pilot-decision record in DESIGN_DRAFT.md).

  python evidence_log.py hotpot final runs/hotpot_final_gen.jsonl   ->  runs/EVIDENCE_LOG_hotpot_final.{jsonl,txt}

Each agent's MAS revision (A1m, B1m) is checked separately, at four stages:
  availability   which gold paragraphs the agent holds, which its partner holds
  transmission   the MAS revision prompt is re-rendered from the frozen template; for each gold paragraph held by the
                 partner, whether its full text is in that prompt, and whether the call finished normally
  utilization    whether the revised answer is EM-correct; whether it cites a partner gold passage id (P6–P10), and
                 whether it simply equals the partner's round-0 answer (answer adoption)
  outcome        a single label, assigned in this order:
                   unknown_unparsed          the revised answer could not be parsed, or the call hit the token limit
                   no_availability           no gold paragraph in the agent's own set or its partner's
                   not_transmitted           a partner-held gold paragraph is missing from the revision prompt
                   correct                   EM 1
                   integration_failure       every gold paragraph was in the prompt, and the answer has F1 0
                   unknown_partial_match     EM 0 but F1 > 0 (it may be a formatting mismatch, so it is not called a failure)
                   unknown_insufficient      some gold paragraph was in neither set
The question-level MAS-final outcome uses the same order over the combined answer, with transmission required for
both agents.
"""
import json
import os
import re
import sys
from collections import Counter

from common import ROOT, f1, fmt_passages, normalize, prompt

CITE = re.compile(r"\[P(\d+)\]")


def agent_record(q, g, me, other, gold, lab):
    rev = g[f"{me}1m"]
    msgs = prompt("revise_mas", question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=g[f"{me}0"]["text"],
                  partner_answer=g[f"{other}0"]["answer"], partner_reasoning=g[f"{other}0"]["text"],
                  partner_evidence=fmt_passages(q[other], start=6))
    body = "\n".join(m["content"] for m in msgs)
    own_gold, part_gold = set(q[f"{me}_gold"]), set(q[f"{other}_gold"])
    part_ids = {6 + i for i, p in enumerate(q[other]) if p["title"] in gold}
    sent = {p["title"]: p["text"] in body for p in q[other] if p["title"] in gold}
    ans = rev["answer"]
    rec = dict(agent=me, correct0=normalize(g[f"{me}0"]["answer"] or "") == normalize(lab),
               own_gold=sorted(own_gold), partner_gold=sorted(part_gold),
               transmitted=all(sent.values()) if sent else None, transmitted_titles=sent,
               finish=rev["finish"], input_tokens=rev["input_tokens"], answer=ans,
               em=normalize(ans or "") == normalize(lab), f1=f1(ans or "", lab),
               cites_partner_gold=bool(part_ids & {int(x) for x in CITE.findall(rev["text"] or "")}),
               adopted_partner_answer=bool(ans) and normalize(ans) == normalize(g[f"{other}0"]["answer"] or ""))
    union_ok = gold <= own_gold | part_gold
    if ans is None or rev["finish"] != "stop":
        o = "unknown_unparsed"
    elif not (own_gold | part_gold):
        o = "no_availability"
    elif sent and not all(sent.values()):
        o = "not_transmitted"
    elif rec["em"]:
        o = "correct"
    elif union_ok and rec["f1"] == 0:
        o = "integration_failure"
    elif rec["f1"] > 0:
        o = "unknown_partial_match"
    else:
        o = "unknown_insufficient"
    rec["outcome"] = o
    return rec


def main():
    ds, split, gen = sys.argv[1], sys.argv[2], sys.argv[3]
    assert ds == "hotpot", "FRAMES uses retrieval_frames.jsonl provenance; this log is HotpotQA-only"
    out_dir = os.environ.get("WQ_OUT", f"{ROOT}/runs")
    lab = json.load(open(f"{ROOT}/data/hotpot_{split}_labels.json"))
    Q = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/hotpot_{split}.jsonl"))}
    rows = []
    for g in sorted(map(json.loads, open(gen)), key=lambda g: g["idx"]):
        q, y = Q[g["idx"]], lab[str(g["idx"])]
        gold = set(q["gold_titles"])
        ag = {me: agent_record(q, g, me, other, gold, y) for me, other in (("A", "B"), ("B", "A"))}
        fa = g["MAS"]["answer"]
        calls = [g["A1m"], g["B1m"], *g["MAS"]["calls"]]
        if fa is None or any(c["finish"] != "stop" for c in calls):
            fo = "unknown_unparsed"
        elif any(a["transmitted"] is False for a in ag.values()):
            fo = "not_transmitted"
        elif normalize(fa) == normalize(y):
            fo = "correct"
        elif f1(fa, y) == 0:
            fo = "integration_failure"
        else:
            fo = "unknown_partial_match"
        disc = ag["A"]["correct0"] != ag["B"]["correct0"]
        rows.append(dict(idx=g["idx"], gold_titles=sorted(gold), discordant0=disc,
                         wrong_agent0=("A" if not ag["A"]["correct0"] else "B") if disc else None,
                         mas_answer=fa, mas_synthesized=g["MAS"]["synthesized"], mas_outcome=fo, agents=ag))
    with open(f"{out_dir}/EVIDENCE_LOG_{ds}_{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    wr = [r["agents"][r["wrong_agent0"]] for r in rows if r["discordant0"]]
    L = [f"=== evidence log (descriptive) {ds} {split}: n={len(rows)} ===",
         "MAS-final outcome: " + json.dumps(Counter(r["mas_outcome"] for r in rows)),
         f"discordant round-0 pairs: {len(wr)}; wrong agent's MAS revision outcome: " + json.dumps(Counter(a["outcome"] for a in wr)),
         f"  partner holds gold {sum(bool(a['partner_gold']) for a in wr)}; transmitted {sum(a['transmitted'] is True for a in wr)}; "
         f"corrected {sum(a['em'] for a in wr)}; of corrected: cites partner gold passage {sum(a['em'] and a['cites_partner_gold'] for a in wr)}, "
         f"equals partner round-0 answer {sum(a['em'] and a['adopted_partner_answer'] for a in wr)}",
         "all agent revisions outcome: " + json.dumps(Counter(a["outcome"] for r in rows for a in r["agents"].values()))]
    open(f"{out_dir}/EVIDENCE_LOG_{ds}_{split}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
