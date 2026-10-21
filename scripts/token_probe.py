"""k* for SC (DESIGN_DRAFT.md §2) from tokens only: T_MAS = mean tokens of A0+B0+A1m+B1m(+synth) per question;
t_s = mean tokens of one Pooled-prompt T=0.7 sample (the probe run uses --k 1). Writes data/K_FROZEN_<dataset>.json."""
import json
import sys

import numpy as np

from common import ROOT

tok = lambda r: r["input_tokens"] + r["output_tokens"]


def main():
    ds, path = sys.argv[1], sys.argv[2]
    G = [json.loads(l) for l in open(path) if json.loads(l)["error"] is None]
    T = np.mean([sum(tok(g[r]) for r in ("A0", "B0", "A1m", "B1m")) + sum(tok(c) for c in g["MAS"]["calls"]) for g in G])
    t_s = np.mean([tok(g["SC"]["samples"][0]) for g in G])
    k = int(np.arange(1, 101)[np.argmin(np.abs(np.arange(1, 101) * t_s - T))])
    out = dict(dataset=ds, n_probe=len(G), T_MAS=float(T), t_s=float(t_s), k_star=k, predicted_ratio=float(T / (k * t_s)))
    json.dump(out, open(f"{ROOT}/data/K_FROZEN_{ds}.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
