"""Experiment W4: evidence-overlap intervention on FRAMES final 624 (Qwen2.5-14B). Generation only; gold never read.
Each agent always holds 5 passages. With overlap o ∈ {1..5} shared passages:
  A gets A[:5−⌊o/2⌋] + B[:⌊o/2⌋]      B gets B[:5−⌈o/2⌉] + A[:⌈o/2⌉]
so |A ∩ B| = o and |A ∪ B| = 10 − o (A, B = the frozen per-source BM25 top-5 lists; sources are disjoint).
o = 0 is exactly the Distributed layout (the existing FRAMES final run; verified through the cache).
Roles, prompts, combiner and seeds as run_distribution.py (partner ids start after the agent's own passages).
  python run_overlap.py --o 3 --out runs/overlap/O3_final_gen.jsonl [--n N]"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import fmt_passages, prompt
from run_webqa import SLOT, call, combine, load


def assign(q, o):
    A, B = q["A"], q["B"]
    a = A[:5 - o // 2] + B[:o // 2]
    b = B[:5 - (o + 1) // 2] + A[:(o + 1) // 2]
    return dict(q, A=a, B=b)


def run_q(q, o):
    q = assign(q, o)
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"], o=o, pids=dict(A=[p["pid"] for p in q["A"]], B=[p["pid"] for p in q["B"]]))
    out["A0"] = call(prompt("answer", question=q["question"], evidence=ev["A"]), 0.0, s0 + SLOT["A0"])
    out["B0"] = call(prompt("answer", question=q["question"], evidence=ev["B"]), 0.0, s0 + SLOT["B0"])
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
    ap.add_argument("--o", type=int, required=True, choices=range(0, 6))
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = load("frames", "final")
    Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            return dict(run_q(q, a.o), error=None)
        except Exception:
            return dict(idx=q["idx"], o=a.o, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"overlap o={a.o}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
