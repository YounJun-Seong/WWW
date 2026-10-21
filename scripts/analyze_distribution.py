"""Sealed analysis of the evidence-distribution experiment on FRAMES final 624 (Experiment W2). Run once.
Judge-based correctness (Qwen3-32B-FP8; the judge was human-validated on 50 pilot items: agreement 0.98, kappa 0.959).
Paired bootstrap over questions: 10,000 resamples, seed 0, one shared index matrix for every contrast.

PRIMARY   DiD(S_UNION) = [MAS − No-comm]_Distributed − [MAS − No-comm]_S_UNION
SECONDARY DiD(S_5), DiD(S_UNION_T7); P-int within each condition; MAS_cond − MAS_D; No-comm_cond − No-comm_D;
          MAS_cond − Pooled; round-0 answer agreement; κ0 → κ_MAS; BRR/HRR/DI on discordant round-0 pairs; tokens.
Reading (fixed in advance): distribution moderates the communication effect if the DiD 95% CI lies above 0; the effect
is reversed if it lies below 0; otherwise it is not established.

  python analyze_distribution.py runs/dist  ->  runs/dist/RESULTS_distribution.{txt,json}
"""
import json
import sys

import numpy as np

from common import ROOT, normalize

B = 10000
CONDS = ("S_UNION", "S_5", "S_UNION_T7")


def tok(r):
    return r["input_tokens"] + r["output_tokens"]


def main():
    d = sys.argv[1]
    def load(gen, judged):
        G = {g["idx"]: g for g in map(json.loads, open(gen))}
        J = {j["idx"]: j for j in map(json.loads, open(judged))}
        assert all(g["error"] is None for g in G.values())
        return G, J
    data = {"D": load(f"{ROOT}/runs/final/frames_final_gen.jsonl", f"{ROOT}/runs/final/frames_final_judged.jsonl")}
    for c in CONDS:
        data[c] = load(f"{d}/{c}_final_gen.jsonl", f"{d}/{c}_final_judged.jsonl")
    ids = sorted(data["D"][0])
    assert all(sorted(data[c][0]) == ids for c in CONDS), "question sets differ"
    n = len(ids)
    V = {c: {r: np.array([float(data[c][1][i][r]["verdict"]) for i in ids]) for r in ("A0", "B0", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM")}
         for c in data}
    pool = np.array([float(data["D"][1][i]["POOL"]["verdict"]) for i in ids])
    idx = np.random.default_rng(0).integers(0, n, (B, n))

    def ci(x):
        bs = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pint = {c: V[c]["MAS"] - V[c]["NOCOMM"] for c in data}
    res = dict(n=n, primary=dict(DiD_S_UNION=ci(pint["D"] - pint["S_UNION"])), secondary={}, conditions={})
    for c in ("S_5", "S_UNION_T7"):
        res["secondary"][f"DiD_{c}"] = ci(pint["D"] - pint[c])
    for c in data:
        G = data[c][0]; v = V[c]
        a, b = v["A0"], v["B0"]
        disc = [(k, "A" if a[k] == 0 else "B") for k in range(n) if a[k] != b[k]]
        brr = np.array([v[f"{w}1m"][k] for k, w in disc]); hrr = np.array([1 - v[f"{'B' if w == 'A' else 'A'}1m"][k] for k, w in disc])
        k0 = np.maximum(a, b); km = np.maximum(v["A1m"], v["B1m"])
        res["conditions"][c] = dict(
            acc={r: float(v[r].mean()) for r in v}, P_int=ci(pint[c]),
            MAS_minus_Pooled=ci(v["MAS"] - pool),
            MAS_minus_MAS_D=ci(v["MAS"] - V["D"]["MAS"]) if c != "D" else None,
            NOCOMM_minus_NOCOMM_D=ci(v["NOCOMM"] - V["D"]["NOCOMM"]) if c != "D" else None,
            round0_answer_agreement=float(np.mean([bool(G[i]["A0"]["answer"]) and normalize(G[i]["A0"]["answer"]) == normalize(G[i]["B0"]["answer"] or "") for i in ids])),
            kappa0=float(k0.mean()), kappa_MAS=float(km.mean()), gain=float(((k0 == 0) & (km == 1)).mean()), loss=float(((k0 == 1) & (km == 0)).mean()),
            n_discordant=len(disc), BRR=float(brr.mean()) if disc else None, HRR=float(hrr.mean()) if disc else None,
            tokens=dict(MAS=float(np.mean([tok(G[i]["A0"]) + tok(G[i]["B0"]) + tok(G[i]["A1m"]) + tok(G[i]["B1m"]) + sum(tok(x) for x in G[i]["MAS"]["calls"]) for i in ids])),
                        NOCOMM=float(np.mean([tok(G[i]["A0"]) + tok(G[i]["B0"]) + tok(G[i]["A1n"]) + tok(G[i]["B1n"]) + sum(tok(x) for x in G[i]["NOCOMM"]["calls"]) for i in ids]))),
            synthesized_MAS=float(np.mean([G[i]["MAS"]["synthesized"] for i in ids])))
    lo, hi = res["primary"]["DiD_S_UNION"][1:]
    res["reading"] = "distribution moderates the communication effect" if lo > 0 else ("reversed" if hi < 0 else "not established")
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L = [f"=== FRAMES final evidence-distribution experiment (judge-based), n={n} ===",
         f"PRIMARY  DiD(S_UNION) = [MAS−No-comm]_D − [MAS−No-comm]_S_UNION = {pp(res['primary']['DiD_S_UNION'])}  -> {res['reading']}",
         *[f"secondary DiD({c}) = {pp(res['secondary'][f'DiD_{c}'])}" for c in ("S_5", "S_UNION_T7")]]
    for c, r in res["conditions"].items():
        L.append(f"[{c}] MAS {r['acc']['MAS']:.3f} No-comm {r['acc']['NOCOMM']:.3f} A0 {r['acc']['A0']:.3f} B0 {r['acc']['B0']:.3f}; "
                 f"P-int {pp(r['P_int'])}; MAS−Pooled {pp(r['MAS_minus_Pooled'])}"
                 + (f"; MAS−MAS_D {pp(r['MAS_minus_MAS_D'])}; NoComm−NoComm_D {pp(r['NOCOMM_minus_NOCOMM_D'])}" if c != "D" else ""))
        L.append(f"     round-0 agreement {r['round0_answer_agreement']:.3f}; κ0 {r['kappa0']:.3f} → MAS {r['kappa_MAS']:.3f} (gain {r['gain']:.3f}, loss {r['loss']:.3f}); "
                 f"discordant {r['n_discordant']} BRR {r['BRR']} HRR {r['HRR']}; tokens/q MAS {r['tokens']['MAS']:.0f} No-comm {r['tokens']['NOCOMM']:.0f}; synth {r['synthesized_MAS']:.3f}")
    open(f"{d}/RESULTS_distribution.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{d}/RESULTS_distribution.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
