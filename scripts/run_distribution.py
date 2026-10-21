"""Evidence-distribution experiment on FRAMES (DESIGN_DRAFT.md, Experiment W2). Generation only; gold is never read.
The questions, model, prompts, combiner and seeds are the frozen ones. Only the assignment of evidence to the two agents
changes:
  S_UNION     A = B = A top-5 ∪ B top-5 from the Distributed run (the same 10 passages; the team's union is unchanged)
  S_5         A = B = the pooled BM25 top-5 (U[:5]; the per-agent amount of evidence is unchanged)
  S_UNION_T7  as S_UNION, but round 0 is sampled at T = 0.7 (seeds per slot as frozen), so the two agents can disagree
The Distributed condition is the existing FRAMES final run, reused unchanged.
Roles: A0, B0, A1m, B1m, A1n, B1n, MAS, NOCOMM. Partner-passage ids start after the agent's own passages, so in the
Distributed layout they stay P6–P10, exactly as in run_webqa.py.

  python run_distribution.py --cond S_UNION --split final --out runs/dist/S_UNION_final_gen.jsonl
"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import fmt_passages, prompt
from run_webqa import SLOT, call, combine, load

CONDS = ("S_UNION", "S_5", "S_UNION_T7", "D")   # "D" = Distributed layout, used only to verify prompt identity via the cache


def assign(q, cond):
    if cond == "D":
        return q
    if cond in ("S_UNION", "S_UNION_T7"):
        ev = q["A"] + q["B"]
    else:
        ev = q["U"][:5]
    return dict(q, A=ev, B=list(ev))


def run_q(q, cond):
    q = assign(q, cond)
    t0 = 0.7 if cond == "S_UNION_T7" else 0.0
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"], cond=cond, pids=dict(A=[p["pid"] for p in q["A"]], B=[p["pid"] for p in q["B"]]))
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
    ap.add_argument("--split", required=True, choices=["pilot", "final"])
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = load("frames", a.split)
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
    print(f"{a.cond} {a.split}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
