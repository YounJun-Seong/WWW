"""FRAMES judge (DESIGN_DRAFT.md §3): Qwen3-32B-FP8, T=0, thinking disabled, frozen prompts/judge.txt -> TRUE/FALSE.
Judges every role's extracted answer (fallback: last 600 chars of the text). Reads gold from frames_test.tsv.

  python judge.py --gen gen.jsonl --out judged.jsonl   (env WQ_JUDGE_URL, WQ_JUDGE_MODEL)
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import ROOT, chat, prompt

ROLES = ("A0", "B0", "POOL", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM", "SC", "PLACEBO_A")


def response_of(r):
    if r is None:
        return ""
    if r.get("answer"):
        return r["answer"]
    return (r.get("text") or "")[-600:]


def verdict(question, gold, resp):
    if not resp.strip():
        return dict(verdict=False, raw="(empty response)")
    c = chat(prompt("judge", question=question, gold=gold, response=resp), 0.0, 8, 0,
             base_url=os.environ["WQ_JUDGE_URL"], model=os.environ["WQ_JUDGE_MODEL"],
             extra={"chat_template_kwargs": {"enable_thinking": False}})
    raw = c["texts"][0].strip()
    return dict(verdict=raw.upper().startswith("TRUE"), raw=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    G = [json.loads(l) for l in open(a.gen)]

    def job(g):
        q, gold = df.loc[g["idx"], "Prompt"], str(df.loc[g["idx"], "Answer"])
        out = dict(idx=g["idx"])
        for role in ROLES:
            if role in g:
                resp = response_of(g[role])
                out[role] = dict(response=resp, **verdict(q, gold, resp))
        if "SC" in g:
            out["SC_samples"] = [verdict(q, gold, response_of(s))["verdict"] for s in g["SC"]["samples"]]
        return out
    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(job, [g for g in G if g.get("error") is None]))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"judged {len(rows)} questions")


if __name__ == "__main__":
    main()
