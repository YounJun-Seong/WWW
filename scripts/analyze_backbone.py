"""Backbone replication on FRAMES final 624 (Experiment W3b), run once per backbone after judging. Judge-based.
Paired bootstrap over questions: 10,000 resamples, seed 0, one shared index matrix.
PRIMARY (per backbone, two estimands): P-int = MAS − No-comm (Distributed); DiD(S_UNION) = [MAS − No-comm]_D −
[MAS − No-comm]_S_UNION.  SECONDARY: P-util = MAS − Pooled; MAS − SC-k (token ratio); arm accuracies; label-free
execution stats (answer-parse rate per role, truncation share) — reported, never used to exclude a backbone.
Reading (fixed in advance): "replicates" if both primary 95% CIs lie above 0; "partially replicates" if exactly one
does; otherwise "does not replicate". Backbones are reported separately and never pooled.
  python analyze_backbone.py <backbone_dir> <label>  ->  <backbone_dir>/RESULTS_backbone.{txt,json}"""
import json
import sys

import numpy as np

B = 10000
ROLES = ("A0", "B0", "POOL", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM", "SC", "PLACEBO_A")
tok = lambda r: r["input_tokens"] + r["output_tokens"]


def main():
    d, label = sys.argv[1], sys.argv[2]
    rd = lambda p: {x["idx"]: x for x in map(json.loads, open(f"{d}/{p}"))}
    G, J, GS, JS = rd("frames_final_gen.jsonl"), rd("frames_final_judged.jsonl"), rd("S_UNION_final_gen.jsonl"), rd("S_UNION_final_judged.jsonl")
    ids = sorted(G); n = len(ids)
    assert sorted(GS) == ids
    err = sum(G[i]["error"] is not None for i in ids) + sum(GS[i]["error"] is not None for i in ids)
    assert err == 0, f"{err} generation errors"
    V = {r: np.array([float(J[i][r]["verdict"]) for i in ids]) for r in ROLES}
    VS = {r: np.array([float(JS[i][r]["verdict"]) for i in ids]) for r in ("MAS", "NOCOMM")}
    idx = np.random.default_rng(0).integers(0, n, (B, n))

    def ci(x):
        bs = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pint, pint_s = V["MAS"] - V["NOCOMM"], VS["MAS"] - VS["NOCOMM"]
    T = dict(MAS=np.mean([tok(G[i]["A0"]) + tok(G[i]["B0"]) + tok(G[i]["A1m"]) + tok(G[i]["B1m"]) + sum(tok(c) for c in G[i]["MAS"]["calls"]) for i in ids]),
             SC=np.mean([sum(tok(s) for s in G[i]["SC"]["samples"]) for i in ids]), POOL=np.mean([tok(G[i]["POOL"]) for i in ids]))
    calls = [G[i][r] for i in ids for r in ("A0", "B0", "POOL", "A1m", "B1m", "A1n", "B1n")]
    res = dict(backbone=label, n=n,
               primary=dict(P_int=ci(pint), DiD_S_UNION=ci(pint - pint_s)),
               secondary=dict(P_util=ci(V["MAS"] - V["POOL"]), MAS_minus_SC=ci(V["MAS"] - V["SC"]),
                              token_ratio_MAS_SC=float(T["MAS"] / T["SC"]), P_int_S_UNION=ci(pint_s),
                              k_star=G[ids[0]]["SC"]["k"]),
               accuracy={r: float(V[r].mean()) for r in ROLES} | {"S_UNION_MAS": float(VS["MAS"].mean()), "S_UNION_NOCOMM": float(VS["NOCOMM"].mean())},
               tokens={k: float(v) for k, v in T.items()},
               execution=dict(parse_rate={r: float(np.mean([G[i][r]["answer"] is not None for i in ids])) for r in ("A0", "B0", "POOL", "A1m", "B1m", "A1n", "B1n")},
                              truncated_share=float(np.mean([c["finish"] != "stop" for c in calls]))))
    above = [res["primary"][k][1] > 0 for k in ("P_int", "DiD_S_UNION")]
    res["reading"] = "replicates" if all(above) else ("partially replicates" if any(above) else "does not replicate")
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    a = res["accuracy"]
    L = [f"=== backbone {label}: FRAMES final n={n} (judge-based) -> {res['reading']} ===",
         f"PRIMARY  P-int MAS − No-comm = {pp(res['primary']['P_int'])}",
         f"PRIMARY  DiD(S_UNION)        = {pp(res['primary']['DiD_S_UNION'])}   (S_UNION P-int {pp(res['secondary']['P_int_S_UNION'])})",
         f"P-util MAS − Pooled {pp(res['secondary']['P_util'])}; MAS − SC-{res['secondary']['k_star']} {pp(res['secondary']['MAS_minus_SC'])} (token ratio {res['secondary']['token_ratio_MAS_SC']:.2f})",
         "accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in a.items()),
         "tokens/q: " + ", ".join(f"{k} {v:.0f}" for k, v in res["tokens"].items()),
         f"execution: parse min {min(res['execution']['parse_rate'].values()):.3f}, truncated {res['execution']['truncated_share']:.4f}"]
    open(f"{d}/RESULTS_backbone.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{d}/RESULTS_backbone.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
