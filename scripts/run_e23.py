"""E2 / E3 generation (DESIGN_DRAFT.md, Experiments E2 and E3). Gold answers are never read; E2's placement uses only the
dataset's gold-page links (data/e2_assign.jsonl, built in advance). Roles, prompts, combiner and seeds are those of
run_distribution.py (A0, B0, A1m, B1m, A1n, B1n, MAS, NOCOMM); only the per-agent passage lists change.
  E2:  python run_e23.py --exp e2 --cell EXCL_LOW --out runs/e2/EXCL_LOW_gen.jsonl
  E3:  python run_e23.py --exp e3 --out runs/e3/TOPIC_gen.jsonl"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, prompt
from run_webqa import SLOT, call, combine, load


def run_q(q):
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"], pids=dict(A=[p["pid"] for p in q["A"]], B=[p["pid"] for p in q["B"]]))
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
    ap.add_argument("--exp", required=True, choices=["e2", "e3"])
    ap.add_argument("--cell", default=None, choices=[None, "EXCL_LOW", "EXCL_HIGH", "SHAR_LOW", "SHAR_HIGH"])
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = {p["pid"]: p for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    base = {q["idx"]: q for q in load("frames", "final")}
    if a.exp == "e2":
        A = [x for x in map(json.loads, open(f"{ROOT}/data/e2_assign.jsonl")) if x["eligible"]]
        Q = [dict(base[x["idx"]], A=[P[j] for j in x["cells"][a.cell]["A"]], B=[P[j] for j in x["cells"][a.cell]["B"]]) for x in A]
    else:
        R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_topic.jsonl"))}
        Q = [dict(q, A=[P[j] for j in R[i]["A"]], B=[P[j] for j in R[i]["B"]]) for i, q in sorted(base.items())]
    Q = sorted(Q, key=lambda q: q["idx"]); Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            return dict(run_q(q), error=None)
        except Exception:
            return dict(idx=q["idx"], error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(dict(r, exp=a.exp, cell=a.cell)) + "\n")
    print(f"{a.exp} {a.cell or 'TOPIC'}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
