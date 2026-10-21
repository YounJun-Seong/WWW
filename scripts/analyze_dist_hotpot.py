"""Sealed analysis of W2 on HotpotQA final 1,000 (Experiment W3a), run once. EM against the frozen labels.
Same estimands, bootstrap (10,000, seed 0, one shared index matrix) and reading rule as analyze_distribution.py:
PRIMARY DiD(S_UNION) = [MAS − No-comm]_D − [MAS − No-comm]_S_UNION; secondary DiD(S_5), DiD(S_UNION_T7) and per-condition
P-int, MAS − Pooled, MAS_cond − MAS_D, No-comm_cond − No-comm_D, round-0 agreement, κ, BRR/HRR, tokens, F1 means.
  python analyze_dist_hotpot.py runs/dist_hotpot  ->  RESULTS_dist_hotpot.{txt,json}"""
import json
import sys

import numpy as np

from common import ROOT, f1, normalize

B = 10000
CONDS = ("S_UNION", "S_5", "S_UNION_T7")
ROLES = ("A0", "B0", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM")
tok = lambda r: r["input_tokens"] + r["output_tokens"]


def main():
    d = sys.argv[1]
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    G = {"D": {g["idx"]: g for g in map(json.loads, open(f"{ROOT}/runs/final/hotpot_final_gen.jsonl"))}}
    for c in CONDS:
        G[c] = {g["idx"]: g for g in map(json.loads, open(f"{d}/{c}_final_gen.jsonl"))}
    ids = sorted(G["D"]); n = len(ids)
    assert all(sorted(G[c]) == ids and all(g["error"] is None for g in G[c].values()) for c in G)
    em = lambda a, i: float(normalize(a or "") == normalize(lab[str(i)]))
    V = {c: {r: np.array([em(G[c][i][r]["answer"], i) for i in ids]) for r in ROLES} for c in G}
    F1 = {c: float(np.mean([f1(G[c][i]["MAS"]["answer"] or "", lab[str(i)]) for i in ids])) for c in G}
    pool = np.array([em(G["D"][i]["POOL"]["answer"], i) for i in ids])
    idx = np.random.default_rng(0).integers(0, n, (B, n))

    def ci(x):
        bs = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pint = {c: V[c]["MAS"] - V[c]["NOCOMM"] for c in G}
    res = dict(n=n, primary=dict(DiD_S_UNION=ci(pint["D"] - pint["S_UNION"])),
               secondary={f"DiD_{c}": ci(pint["D"] - pint[c]) for c in ("S_5", "S_UNION_T7")}, conditions={})
    for c in G:
        v, g = V[c], G[c]; a, b = v["A0"], v["B0"]
        disc = [(k, "A" if a[k] == 0 else "B") for k in range(n) if a[k] != b[k]]
        k0 = np.maximum(a, b); km = np.maximum(v["A1m"], v["B1m"])
        res["conditions"][c] = dict(
            em={r: float(v[r].mean()) for r in v}, MAS_f1=F1[c], P_int=ci(pint[c]), MAS_minus_Pooled=ci(v["MAS"] - pool),
            MAS_minus_MAS_D=ci(v["MAS"] - V["D"]["MAS"]) if c != "D" else None,
            NOCOMM_minus_NOCOMM_D=ci(v["NOCOMM"] - V["D"]["NOCOMM"]) if c != "D" else None,
            round0_answer_agreement=float(np.mean([bool(g[i]["A0"]["answer"]) and normalize(g[i]["A0"]["answer"]) == normalize(g[i]["B0"]["answer"] or "") for i in ids])),
            kappa0=float(k0.mean()), kappa_MAS=float(km.mean()), n_discordant=len(disc),
            BRR=float(np.mean([v[f"{w}1m"][k] for k, w in disc])) if disc else None,
            HRR=float(np.mean([1 - v[f"{'B' if w == 'A' else 'A'}1m"][k] for k, w in disc])) if disc else None,
            tokens_MAS=float(np.mean([tok(g[i]["A0"]) + tok(g[i]["B0"]) + tok(g[i]["A1m"]) + tok(g[i]["B1m"]) + sum(tok(x) for x in g[i]["MAS"]["calls"]) for i in ids])))
    lo, hi = res["primary"]["DiD_S_UNION"][1:]
    res["reading"] = "distribution moderates the communication effect" if lo > 0 else ("reversed" if hi < 0 else "not established")
    pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L = [f"=== HotpotQA final evidence-distribution experiment (EM), n={n} ===",
         f"PRIMARY  DiD(S_UNION) = {pp(res['primary']['DiD_S_UNION'])}  -> {res['reading']}",
         *[f"secondary DiD({c}) = {pp(res['secondary'][f'DiD_{c}'])}" for c in ("S_5", "S_UNION_T7")]]
    for c, r in res["conditions"].items():
        L.append(f"[{c}] MAS {r['em']['MAS']:.3f} (F1 {r['MAS_f1']:.3f}) No-comm {r['em']['NOCOMM']:.3f} A0 {r['em']['A0']:.3f} B0 {r['em']['B0']:.3f}; "
                 f"P-int {pp(r['P_int'])}; MAS−Pooled {pp(r['MAS_minus_Pooled'])}"
                 + (f"; MAS−MAS_D {pp(r['MAS_minus_MAS_D'])}; NoComm−NoComm_D {pp(r['NOCOMM_minus_NOCOMM_D'])}" if c != "D" else ""))
        L.append(f"     round-0 agreement {r['round0_answer_agreement']:.3f}; κ0 {r['kappa0']:.3f} → MAS {r['kappa_MAS']:.3f}; "
                 f"discordant {r['n_discordant']} BRR {r['BRR']} HRR {r['HRR']}; MAS tokens/q {r['tokens_MAS']:.0f}")
    open(f"{d}/RESULTS_dist_hotpot.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{d}/RESULTS_dist_hotpot.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
