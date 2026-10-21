"""Experiment W2 on HotpotQA final 1,000 (Experiment W3a; Qwen2.5-14B). Generation only; gold is never read.
Same roles, prompts, combiner and seeds as run_distribution.py; only the paragraph assignment changes:
  S_UNION     A = B = all 10 paragraphs (A's 5 + B's 5; the team's union is unchanged)
  S_5         A = B = A's 5 paragraphs (the per-agent amount is unchanged; there is no retrieval in HotpotQA)
  S_UNION_T7  as S_UNION, round 0 at T = 0.7
The Distributed condition is the existing HotpotQA final run (runs/final/hotpot_final_gen.jsonl), reused unchanged.
  python run_dist_hotpot.py --cond S_UNION --out runs/dist_hotpot/S_UNION_final_gen.jsonl [--n N]"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import fmt_passages, prompt
from run_webqa import SLOT, call, combine, load

CONDS = ("S_UNION", "S_5", "S_UNION_T7", "D")   # "D" only verifies prompt identity via the cache


def assign(q, cond):
    if cond == "D":
        return q
    ev = q["A"] + q["B"] if cond in ("S_UNION", "S_UNION_T7") else q["A"]
    return dict(q, A=list(ev), B=list(ev))


def run_q(q, cond):
    q = assign(q, cond)
    t0 = 0.7 if cond == "S_UNION_T7" else 0.0
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"], cond=cond, titles=dict(A=[p["title"] for p in q["A"]], B=[p["title"] for p in q["B"]]))
    out["A0"] = call(prompt("answer", question=q["question"], evidence=ev["A"]), t0, s0 + SLOT["A0"])
    out["B0"] = call(prompt("answer", question=q["question"], evidence=ev["B"]), t0, s0 + SLOT["B0"])
    for me, other, sm, sn in (("A", "B", "A1m", "A1n"), ("B", "A", "B1m", "B1n")):
        own, part = out[f"{me}0"], out[f"{other}0"]
        out[sm] = call(prompt("revise_mas", question=q["question"], evidence=ev[me], own_reasoning=own["text"],
                              partner_answer=part["answer"], partner_reasoning=part["text"],
                              partner_evidence=fmt_passages(q[other], start=len(q[me]) + 1)), 0.0, s0 + SLOT[sm])
        out[sn] = call(prompt("revise_nocomm", question=q["question"], evidence=ev[me], own_reasoning=own["text"]),
                       0.0, s0 + SLOT[sn])
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    out["NOCOMM"] = combine(q, out["A1n"], out["B1n"], s0 + SLOT["SYNn"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = load("hotpot", "final")
    Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            return dict(run_q(q, a.cond), error=None)
        except Exception:
            return dict(idx=q["idx"], cond=a.cond, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"hotpot {a.cond} final: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
