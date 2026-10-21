"""WWW Web QA arms (DESIGN_DRAFT.md §2). Generation only; gold is never read here.

  python run_webqa.py --dataset frames|hotpot --split pilot|final [--n N] [--k K] [--only round0|...] --out f.jsonl

Per question (all arms share the same round-0 calls through the cache):
  A0, B0           Single-A / Single-B (T=0)
  POOL             Pooled single, k=10 (T=0)
  A1m, B1m         MAS revisions with the partner's answer, reasoning and passages (T=0)
  A1n, B1n         No-comm revisions, own evidence only (T=0)
  MAS, NOCOMM      final combiner: agree after normalization -> that answer, else one synthesizer call (T=0)
  SC               k Pooled-prompt samples at T=0.7 (separate n=1 requests), majority over normalized answers
  PLACEBO_A        one extra Single-A draw at T=0.7 (G2 placebo)
Seeds: 7·10⁶ + idx·1000 + call slot (fixed per role, so every arm's calls are reproducible and shared).
"""
import argparse
import json
import os
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import ROOT, answer_of, chat, fmt_passages, normalize, prompt

MAXTOK = 1024
SLOT = dict(A0=0, B0=1, POOL=2, A1m=3, B1m=4, A1n=5, B1n=6, SYNm=7, SYNn=8, PLC=9, SC=100)


def load(dataset, split):
    if dataset == "frames":
        df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
        ids = json.load(open(f"{ROOT}/data/frames_split.json"))[split]
        P = {p["pid"]: p for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
        R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
        return [dict(idx=i, question=df.loc[i, "Prompt"], A=[P[j] for j in R[i]["A"]], B=[P[j] for j in R[i]["B"]],
                     U=[P[j] for j in R[i]["U"]]) for i in ids]
    rows = [json.loads(l) for l in open(f"{ROOT}/data/hotpot_{split}.jsonl")]
    return [dict(idx=r["idx"], question=r["question"], A=r["A"], B=r["B"], U=r["A"] + r["B"]) for r in rows]


def call(msgs, temp, seed):
    c = chat(msgs, temp, MAXTOK, seed)
    t = c["texts"][0]
    return dict(text=t, answer=answer_of(t), finish=c["finish"][0], input_tokens=c["input_tokens"], output_tokens=c["output_tokens"])


def combine(q, a, b, seed):
    if a["answer"] and b["answer"] and normalize(a["answer"]) == normalize(b["answer"]):
        return dict(answer=a["answer"], synthesized=False, calls=[])
    s = call(prompt("synthesize", question=q["question"], answer_1=a["answer"], reasoning_1=a["text"],
                    answer_2=b["answer"], reasoning_2=b["text"]), 0.0, seed)
    return dict(answer=s["answer"], synthesized=True, calls=[s])


def run_q(q, k, only):
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"])
    out["A0"] = call(prompt("answer", question=q["question"], evidence=ev["A"]), 0.0, s0 + SLOT["A0"])
    out["B0"] = call(prompt("answer", question=q["question"], evidence=ev["B"]), 0.0, s0 + SLOT["B0"])
    if only == "round0":
        return out
    out["POOL"] = call(prompt("answer", question=q["question"], evidence=fmt_passages(q["U"])), 0.0, s0 + SLOT["POOL"])
    for me, other, sm, sn in (("A", "B", "A1m", "A1n"), ("B", "A", "B1m", "B1n")):
        own, part = out[f"{me}0"], out[f"{other}0"]
        out[sm] = call(prompt("revise_mas", question=q["question"], evidence=ev[me], own_reasoning=own["text"],
                              partner_answer=part["answer"], partner_reasoning=part["text"],
                              partner_evidence=fmt_passages(q[other], start=6)), 0.0, s0 + SLOT[sm])
        out[sn] = call(prompt("revise_nocomm", question=q["question"], evidence=ev[me], own_reasoning=own["text"]),
                       0.0, s0 + SLOT[sn])
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    out["NOCOMM"] = combine(q, out["A1n"], out["B1n"], s0 + SLOT["SYNn"])
    out["PLACEBO_A"] = call(prompt("answer", question=q["question"], evidence=ev["A"]), 0.7, s0 + SLOT["PLC"])
    if k:
        pooled = prompt("answer", question=q["question"], evidence=fmt_passages(q["U"]))
        samples = [call(pooled, 0.7, s0 + SLOT["SC"] + j) for j in range(k)]
        norm = [normalize(s["answer"]) for s in samples if s["answer"]]
        c = Counter(norm)
        maj = next((s["answer"] for s in samples if s["answer"] and c[normalize(s["answer"])] == max(c.values())), None) if c else None
        out["SC"] = dict(answer=maj, k=k, samples=samples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["frames", "hotpot"])
    ap.add_argument("--split", required=True, choices=["pilot", "final"])
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--k", type=int, default=0, help="SC samples (0 = skip SC)")
    ap.add_argument("--only", default=None, choices=[None, "round0"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = load(a.dataset, a.split)
    Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            return dict(run_q(q, a.k, a.only), error=None)
        except Exception:
            return dict(idx=q["idx"], error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{a.dataset} {a.split}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
