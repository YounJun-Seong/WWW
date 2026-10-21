"""Backbone version of token_probe.py (same formula; tokens only): writes <out_dir>/K_FROZEN.json instead of
data/K_FROZEN_<dataset>.json, so the frozen Qwen2.5-14B k* is never overwritten.
  python token_probe_bb.py <probe.jsonl> <out_dir>"""
import json
import sys

import numpy as np

tok = lambda r: r["input_tokens"] + r["output_tokens"]


def main():
    path, out_dir = sys.argv[1], sys.argv[2]
    G = [g for g in map(json.loads, open(path)) if g["error"] is None]
    T = np.mean([sum(tok(g[r]) for r in ("A0", "B0", "A1m", "B1m")) + sum(tok(c) for c in g["MAS"]["calls"]) for g in G])
    t_s = np.mean([tok(g["SC"]["samples"][0]) for g in G])
    k = int(np.arange(1, 101)[np.argmin(np.abs(np.arange(1, 101) * t_s - T))])
    out = dict(n_probe=len(G), T_MAS=float(T), t_s=float(t_s), k_star=k, predicted_ratio=float(T / (k * t_s)))
    json.dump(out, open(f"{out_dir}/K_FROZEN.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
