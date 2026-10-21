"""BX3 post-hoc execution robustness audit (DESIGN_BX3.md): original vs regenerated estimates for the key contrasts.
  python bx3_analyze.py   -> runs/bx3/RESULTS_bx3.{txt,json}"""
import json
import os

import numpy as np

from common import ROOT, normalize
from rx_analyze import Score, rd

OUT = f"{ROOT}/runs/bx3"
SRC = dict(R1_FULL="runs/rx", R1_GOLDREP="runs/rx", PEER_EV_5="runs/bx", CQ_Q_5="runs/bx", CQ_GEN_5="runs/bx",
           B3_F_SPLIT="runs/bx", B3_F_SHAR="runs/bx")


def load(ds, arm, regen):
    d = OUT if regen else f"{ROOT}/{SRC[arm]}"
    g = rd(f"{d}/{ds}_{arm}_gen.jsonl")
    assert all(x.get("error") is None for x in g.values()), (ds, arm, regen)
    return g, Score(ds).fn(f"{d}/{ds}_{arm}_judged.jsonl")


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); idx = np.random.default_rng(0).integers(0, len(x), (10000, len(x)))
    return [float(x.mean()), *map(float, np.percentile(x[idx].mean(1), [lo, hi]))]


def contrast(ds, a, b, kind):
    """per-question values: 'team' = MAS; 'agent' = mean of A1m, B1m; 'pint' = MAS − NOCOMM."""
    out = {}
    for regen in (False, True):
        (ga, ca), (gb, cb) = load(ds, a, regen), load(ds, b, regen)
        ids = sorted(i for i in ga if i in gb and not ga[i].get("skipped") and not gb[i].get("skipped"))
        v = {"team": lambda g, c, i: c(i, g[i], "MAS"),
             "agent": lambda g, c, i: (c(i, g[i], "A1m") + c(i, g[i], "B1m")) / 2,
             "pint": lambda g, c, i: c(i, g[i], "MAS") - c(i, g[i], "NOCOMM")}[kind]
        d = [v(ga, ca, i) - v(gb, cb, i) for i in ids]
        out["regen" if regen else "orig"] = dict(n=len(ids), ci95=ci(d), ci90=ci(d, 5, 95))
    return out


def agreement(ds, arm):
    (go, _), (gr, _) = load(ds, arm, False), load(ds, arm, True)
    roles = [r for r in ("A0", "B0", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM") if r in next(iter(gr.values()))]
    ids = sorted(i for i in go if i in gr and not go[i].get("skipped"))
    same = lambda r: float(np.mean([normalize(go[i][r].get("answer") or "") == normalize(gr[i][r].get("answer") or "") for i in ids]))
    return {r: same(r) for r in roles}


def main():
    C = [("R1 FULL − GOLDREP (team final)", ds, "R1_FULL", "R1_GOLDREP", "team") for ds in ("frames", "hotpot")]
    C += [("B3 P-int(F_SPLIT) − P-int(F_SHAR)", "hotpot", "B3_F_SPLIT", "B3_F_SHAR", "pint")]
    C += [("B4-P1 CQ_Q_5 − PEER_EV_5 (agent)", ds, "CQ_Q_5", "PEER_EV_5", "agent") for ds in ("frames", "hotpot")]
    C += [("B4-P2 CQ_GEN_5 − PEER_EV_5 (agent)", ds, "CQ_GEN_5", "PEER_EV_5", "agent") for ds in ("frames", "hotpot")]
    res = dict(contrasts=[], agreement={})
    L = ["=== BX3 post-hoc execution robustness audit: original (primary, unchanged) vs one regeneration (fresh caches) ===",
         f"{'contrast':38s} {'ds':7s} {'original 95% CI':>26s} {'regenerated 95% CI':>26s}  direction  significance"]
    sig = lambda c: "+" if c[1] > 0 else ("−" if c[2] < 0 else "0")
    for name, ds, a, b, kind in C:
        r = contrast(ds, a, b, kind); o, g = r["orig"]["ci95"], r["regen"]["ci95"]
        row = dict(contrast=name, dataset=ds, **r, same_direction=bool(np.sign(o[0]) == np.sign(g[0]) or abs(g[0]) < 1e-9),
                   same_significance=sig(o) == sig(g))
        if name.startswith("B4-P1"):
            row["equivalence_regen"] = dict(ci95=bool(g[1] > -0.02 and g[2] < 0.02),
                                            tost90=bool(r["regen"]["ci90"][1] > -0.02 and r["regen"]["ci90"][2] < 0.02))
        res["contrasts"].append(row)
        f = lambda c: f"{100 * c[0]:+6.1f} [{100 * c[1]:+5.1f}, {100 * c[2]:+5.1f}]"
        L.append(f"{name:38s} {ds:7s} {f(o):>26s} {f(g):>26s}  {'same' if row['same_direction'] else 'FLIP':9s}  "
                 f"{sig(o)} -> {sig(g)}" + (f"  | regen equivalence ±2pp: 95% {row['equivalence_regen']['ci95']}, TOST90 {row['equivalence_regen']['tost90']}"
                                            if "equivalence_regen" in row else ""))
    L.append("answer agreement, original vs regenerated (normalised answer equal):")
    for ds, arms in (("frames", ["R1_FULL", "R1_GOLDREP", "PEER_EV_5", "CQ_Q_5", "CQ_GEN_5"]),
                     ("hotpot", ["R1_FULL", "R1_GOLDREP", "PEER_EV_5", "CQ_Q_5", "CQ_GEN_5", "B3_F_SPLIT", "B3_F_SHAR"])):
        for a in arms:
            res["agreement"][f"{ds}:{a}"] = ag = agreement(ds, a)
            L.append(f"  {ds:7s} {a:11s} " + ", ".join(f"{k} {v:.3f}" for k, v in ag.items()))
    L.append("One regeneration = one additional execution: sensitivity to execution, not invariance across conditions.")
    json.dump(res, open(f"{OUT}/RESULTS_bx3.json", "w"), indent=1)
    open(f"{OUT}/RESULTS_bx3.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
