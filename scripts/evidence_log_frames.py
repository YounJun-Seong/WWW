"""Per-question evidence log for FRAMES (descriptive; separate from the sealed analysis, which it never changes).
Fixed before the FRAMES final run. The HotpotQA counterpart is evidence_log.py (unchanged).

  python evidence_log_frames.py final runs/final/frames_final_gen.jsonl runs/final/frames_final_judged.jsonl
    ->  $WQ_OUT/EVIDENCE_LOG_frames_<split>.{jsonl,txt}

Correctness is the automatic judge's verdict. It is not human-validated, so every rate below is judge-based.
Per discordant round-0 pair (one agent judged correct, the other wrong), for the wrong agent's MAS revision, with the
denominator of each rate:
  held          the partner's source holds at least one gold page                                      / discordant pairs
  retrieved     the partner's top-5 holds at least one gold-page passage                               / discordant pairs
  transmitted   every partner gold-page passage is verbatim in the re-rendered MAS revision prompt, and
                the revision call finished normally                                                    / retrieved
  corrected     the wrong agent's MAS revision is judged correct                                       / transmitted
                (baseline: corrected among pairs whose partner retrieved no gold                       / not retrieved)
  cites         the corrected revision cites a partner gold-page passage id (P6–P10)                   / corrected
Question-level taxonomy for MAS-wrong questions (denominator: MAS-wrong). Labels, assigned in this order:
  unknown_unparsed     the MAS answer is missing, or any MAS call hit the token limit
  retrieval_miss       no gold page in A's or B's top-5
  insufficient_evidence some, but not all, gold pages are in A ∪ B top-5
  integration_failure  every gold page is in A ∪ B top-5
"""
import json
import os
import re
import sys
from collections import Counter

from common import ROOT, fmt_passages, prompt
from run_webqa import load

CITE = re.compile(r"\[P(\d+)\]")


def main():
    split, gen, judged = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = os.environ.get("WQ_OUT", f"{ROOT}/runs")
    Q = {q["idx"]: q for q in load("frames", split)}
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    J = {j["idx"]: j for j in map(json.loads, open(judged))}
    rows = []
    for g in sorted(map(json.loads, open(gen)), key=lambda g: g["idx"]):
        i, q, r, j = g["idx"], Q[g["idx"]], R[g["idx"]], J[g["idx"]]
        gold = set(r["gold_pages"]); ret_union = set(r["A_gold_retrieved"]) | set(r["B_gold_retrieved"])
        ok = {x: bool(j[x]["verdict"]) for x in ("A0", "B0", "A1m", "B1m", "MAS")}
        row = dict(idx=i, n_gold=len(gold), gold_owner=r["gold_owner"], A_gold_retrieved=r["A_gold_retrieved"],
                   B_gold_retrieved=r["B_gold_retrieved"], judge=ok, discordant0=ok["A0"] != ok["B0"])
        if row["discordant0"]:
            w = "A" if not ok["A0"] else "B"; p = "B" if w == "A" else "A"
            rev = g[f"{w}1m"]
            body = "\n".join(m["content"] for m in prompt(
                "revise_mas", question=q["question"], evidence=fmt_passages(q[w]), own_reasoning=g[f"{w}0"]["text"],
                partner_answer=g[f"{p}0"]["answer"], partner_reasoning=g[f"{p}0"]["text"],
                partner_evidence=fmt_passages(q[p], start=6)))
            pg = [(6 + k, ps) for k, ps in enumerate(q[p]) if ps["page"] in gold]
            held = any(o == p for o in r["gold_owner"].values())
            retrieved = bool(pg)
            transmitted = (all(ps["text"] in body for _, ps in pg) and rev["finish"] == "stop") if retrieved else None
            cited = {int(x) for x in CITE.findall(rev["text"] or "")}
            row["pair"] = dict(wrong=w, held=held, retrieved=retrieved, transmitted=transmitted,
                               corrected=ok[f"{w}1m"], finish=rev["finish"], answer_parsed=rev["answer"] is not None,
                               cites_partner_gold=bool(cited & {k for k, _ in pg}))
        if not ok["MAS"]:
            calls = [g["A1m"], g["B1m"], *g["MAS"]["calls"]]
            if g["MAS"]["answer"] is None or any(c["finish"] != "stop" for c in calls):
                t = "unknown_unparsed"
            elif not ret_union:
                t = "retrieval_miss"
            elif len(ret_union) < len(gold):
                t = "insufficient_evidence"
            else:
                t = "integration_failure"
            row["mas_wrong_type"] = t
        rows.append(row)
    with open(f"{out_dir}/EVIDENCE_LOG_frames_{split}.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    P = [x["pair"] for x in rows if x["discordant0"]]
    D = len(P); ret = [x for x in P if x["retrieved"]]; tr = [x for x in ret if x["transmitted"]]
    cor = [x for x in tr if x["corrected"]]; nret = [x for x in P if not x["retrieved"]]
    frac = lambda a, b: f"{a}/{b}" + (f" = {a / b:.3f}" if b else "")
    W = [x for x in rows if not x["judge"]["MAS"]]
    L = [f"=== evidence log (descriptive, judge-based; not human-validated) frames {split}: n={len(rows)} ===",
         f"discordant round-0 pairs D = {D}",
         f"  held        {frac(sum(x['held'] for x in P), D)}   (denominator: D)",
         f"  retrieved   {frac(len(ret), D)}   (denominator: D)",
         f"  transmitted {frac(len(tr), len(ret))}   (denominator: retrieved)",
         f"  corrected   {frac(len(cor), len(tr))}   (denominator: transmitted)",
         f"  corrected | partner retrieved no gold {frac(sum(x['corrected'] for x in nret), len(nret))}   (denominator: not retrieved)",
         f"  cites partner gold passage {frac(sum(x['cites_partner_gold'] for x in cor), len(cor))}   (denominator: corrected)",
         f"MAS-wrong questions = {len(W)}; types (denominator: MAS-wrong): "
         + ", ".join(f"{k} {frac(v, len(W))}" for k, v in sorted(Counter(x['mas_wrong_type'] for x in W).items()))]
    open(f"{out_dir}/EVIDENCE_LOG_frames_{split}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
