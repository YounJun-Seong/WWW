"""Pre-registered sensitivity analysis for FRAMES (registered before the final run, after the render-date leak was found).
Excludes every question whose retrieved passages (A top-5, B top-5 or U top-10) contain a render-time "(age N)" string
(MediaWiki action=parse renders the frozen revision with the current date). It recomputes P-int, P-util and
MAS − SC-k with the sealed analysis's rules: paired bootstrap over questions, 10,000 resamples, seed 0.

  python sensitivity_frames.py final runs/final/frames_final_gen.jsonl runs/final/frames_final_judged.jsonl
    ->  $WQ_OUT/SENSITIVITY_frames_<split>.txt
"""
import json
import os
import re
import sys

import numpy as np

from common import ROOT

AGE = re.compile(r"\(age \d+\)")


def main():
    split, gen, judged = sys.argv[1:4]
    out_dir = os.environ.get("WQ_OUT", f"{ROOT}/runs")
    P = {p["pid"]: p["text"] for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    J = {j["idx"]: j for j in map(json.loads, open(judged))}
    ids = sorted(g["idx"] for g in map(json.loads, open(gen)))
    leak = {i for i in ids if any(AGE.search(P[j]) for s in "ABU" for j in R[i][s])}
    keep = [i for i in ids if i not in leak]
    rng = np.random.default_rng(0)
    C = {r: np.array([float(J[i][r]["verdict"]) for i in keep]) for r in ("MAS", "NOCOMM", "POOL", "SC")}

    def boot(d):
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        return f"{d.mean() * 100:+.1f} pp [{np.percentile(bs, 2.5) * 100:+.1f}, {np.percentile(bs, 97.5) * 100:+.1f}]"
    L = [f"=== FRAMES {split} sensitivity (judge-based): excluded {len(leak)} of {len(ids)} questions with a render-time "
         f"'(age N)' string in any retrieved passage; n = {len(keep)} ===",
         f"P-int  MAS − No-comm {boot(C['MAS'] - C['NOCOMM'])}",
         f"P-util MAS − Pooled  {boot(C['MAS'] - C['POOL'])}",
         f"MAS − SC-k           {boot(C['MAS'] - C['SC'])}",
         "excluded idx: " + json.dumps(sorted(leak))]
    open(f"{out_dir}/SENSITIVITY_frames_{split}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
