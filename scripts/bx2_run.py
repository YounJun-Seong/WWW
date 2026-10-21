"""BX2-B ownership-framing control on HotpotQA B3_F_SPLIT (DESIGN_BX2.md §B). Round-0 A0 / B0 of B3_F_SPLIT are reused;
the revision prompt is `revise_mas_local`, which differs from `revise_mas` in exactly one line (checked below). Same
passages, order, numbering, partner answer / reasoning, seed slots and combiner as the CONTROL (the B3_F_SPLIT run).
  python bx2_run.py --out runs/bx2/hotpot_F_SPLIT_LOCAL_gen.jsonl [--n N]"""
import argparse
import difflib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, prompt
from run_webqa import SLOT, call, combine, load
from rx_run import rd


def check_one_line_diff():
    a = open(f"{ROOT}/prompts/revise_mas.txt").read().split("\n")
    b = open(f"{ROOT}/prompts/revise_mas_local.txt").read().split("\n")
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert d == ["-Their evidence passages:", "+More of your own evidence passages:"], d
    return d


def run_q(q, cell, g):
    s0 = 7_000_000 + q["idx"] * 1000
    q = dict(q, A=cell["A"], B=cell["B"])
    assert [p["title"] for p in q["A"]] == g["ids"]["A"] and [p["title"] for p in q["B"]] == g["ids"]["B"]
    out = dict(idx=q["idx"], mode="F_SPLIT_LOCAL", holder=g["holder"], ids=g["ids"], gold=g["gold"], A0=g["A0"], B0=g["B0"])
    for me, other, sm in (("A", "B", "A1m"), ("B", "A", "B1m")):
        own, part = g[f"{me}0"], g[f"{other}0"]
        out[sm] = call(prompt("revise_mas_local", question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=own["text"],
                              partner_answer=part["answer"], partner_reasoning=part["text"],
                              partner_evidence=fmt_passages(q[other], start=len(q[me]) + 1)), 0.0, s0 + SLOT[sm])
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("prompt diff:", check_one_line_diff())
    Q = {q["idx"]: q for q in load("hotpot", "final")}
    cells = rd("data/bx/b3_hotpot.jsonl"); G = rd("runs/bx/hotpot_B3_F_SPLIT_gen.jsonl")
    ids = sorted(G); ids = ids if a.n < 0 else ids[:a.n]

    def job(i):
        try:
            return dict(run_q(Q[i], cells[i]["cells"]["F_SPLIT"], G[i]), error=None)
        except Exception:
            return dict(idx=i, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, ids))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"BX2 F_SPLIT_LOCAL: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
