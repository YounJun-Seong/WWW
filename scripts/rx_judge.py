"""FRAMES judging for RX generation files. Uses the frozen judge unchanged (judge.verdict / judge.response_of: same
prompt, model Qwen3-32B-FP8, T=0, thinking off, same cache), so any answer judged before gets the identical verdict.
Unlike judge.py (fixed role list), it judges every top-level role that carries an answer, because RX files use new role
names (UNI_*, ALWAYS_*, EVSYN_*). Skipped questions are carried through as skipped.
  python rx_judge.py --gen runs/rx/frames_X_gen.jsonl --out runs/rx/frames_X_judged.jsonl"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import ROOT
from judge import response_of, verdict

SKIP = {"idx", "mode", "meta", "ids", "gold", "error", "skipped"}


def roles(g):
    return [k for k, v in g.items() if k not in SKIP and isinstance(v, dict) and ("answer" in v or "text" in v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    G = [json.loads(l) for l in open(a.gen)]
    bad = [g["idx"] for g in G if g.get("error")]
    assert not bad, f"generation errors in {a.gen}: {bad[:5]}"

    def job(g):
        if g.get("skipped"):
            return dict(idx=g["idx"], skipped=True)
        q, gold = df.loc[g["idx"], "Prompt"], str(df.loc[g["idx"], "Answer"])
        out = dict(idx=g["idx"])
        for r in roles(g):
            resp = response_of(g[r]); out[r] = dict(response=resp, **verdict(q, gold, resp))
            if "samples" in g[r]:
                out[f"{r}_samples"] = [verdict(q, gold, response_of(s))["verdict"] for s in g[r]["samples"]]
        return out
    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(job, G))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"judged {sum(not r.get('skipped') for r in rows)} questions ({sum(bool(r.get('skipped')) for r in rows)} skipped)")


if __name__ == "__main__":
    main()
