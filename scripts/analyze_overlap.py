"""Sealed analysis of W4 (evidence overlap), FRAMES final 624, judge-based. Run once.
PRIMARY: the OLS slope of P-int(o) = Acc(MAS) − Acc(No-comm) over o = 0..5 (pp per shared passage). Paired bootstrap
over questions (10,000, seed 0): each resample recomputes the six P-int values and their slope.
SECONDARY: P-int(o) for each o; P-int(0) − P-int(5); the slopes of Acc(MAS) and Acc(No-comm).
DESCRIPTIVE covariates per o: per-agent gold coverage, team gold coverage (A ∪ B shrinks to 10 − o passages), round-0
accuracy and answer agreement, synthesizer shares, tokens.
Reading (fixed in advance): slope CI below 0 -> "communication gain decreases as evidence overlap increases"; above 0 ->
"increases"; otherwise "not established". Because the team's union shrinks with o, a negative slope is not attributed
to overlap alone; the coverage covariates are reported next to it.
  python analyze_overlap.py runs/overlap  ->  RESULTS_overlap.{txt,json}"""
import json
import sys

import numpy as np

from common import ROOT, normalize

B = 10000


def main():
    d = sys.argv[1]
    P = {p["pid"]: p["page"] for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    rd = lambda p: {x["idx"]: x for x in map(json.loads, open(p))}
    G = {0: rd(f"{ROOT}/runs/final/frames_final_gen.jsonl")}; J = {0: rd(f"{ROOT}/runs/final/frames_final_judged.jsonl")}
    for o in range(1, 6):
        G[o] = rd(f"{d}/O{o}_final_gen.jsonl"); J[o] = rd(f"{d}/O{o}_final_judged.jsonl")
    ids = sorted(G[0]); n = len(ids); os_ = np.arange(6)
    assert all(sorted(G[o]) == ids and all(g["error"] is None for g in G[o].values()) for o in G)
    V = {o: {r: np.array([float(J[o][i][r]["verdict"]) for i in ids]) for r in ("MAS", "NOCOMM", "A0", "B0")} for o in G}
    pint = np.stack([V[o]["MAS"] - V[o]["NOCOMM"] for o in os_])           # 6 x n
    mas = np.stack([V[o]["MAS"] for o in os_]); nc = np.stack([V[o]["NOCOMM"] for o in os_])
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    xc = os_ - os_.mean()
    slope = lambda M: (M.mean(1) * xc).sum() / (xc ** 2).sum()
    def sci(M):
        bs = np.array([((M[:, ix].mean(1)) * xc).sum() / (xc ** 2).sum() for ix in idx])
        return [float(slope(M)), *map(float, np.percentile(bs, [2.5, 97.5]))]
    def ci(x):
        return [float(x.mean()), *map(float, np.percentile(x[idx].mean(1), [2.5, 97.5]))]
    res = dict(n=n, primary=dict(slope_P_int=sci(pint)),
               secondary=dict(P_int={int(o): ci(pint[o]) for o in os_}, P_int_0_minus_5=ci(pint[0] - pint[5]),
                              slope_MAS=sci(mas), slope_NOCOMM=sci(nc)), covariates={})
    for o in os_:
        pid = (lambda i, s: R[i][s]) if o == 0 else (lambda i, s, o=o: G[o][i]["pids"][s])
        cov = []
        for i in ids:
            gold = set(R[i]["gold_pages"]); gp = {s: {P[p] for p in pid(i, s)} & gold for s in "AB"}
            g = G[o][i]
            cov.append((np.mean([bool(gp[s]) for s in "AB"]), bool(gp["A"] | gp["B"]), (gp["A"] | gp["B"]) == gold,
                        len(set(pid(i, "A")) & set(pid(i, "B"))),
                        bool(g["A0"]["answer"]) and normalize(g["A0"]["answer"]) == normalize(g["B0"]["answer"] or ""),
                        g["MAS"]["synthesized"], g["NOCOMM"]["synthesized"]))
        c = np.array(cov, float)
        res["covariates"][int(o)] = dict(agent_any_gold=c[:, 0].mean(), team_any_gold=c[:, 1].mean(), team_all_gold=c[:, 2].mean(),
                                         mean_overlap=c[:, 3].mean(), round0_answer_agreement=c[:, 4].mean(),
                                         synth_MAS=c[:, 5].mean(), synth_NOCOMM=c[:, 6].mean(),
                                         acc_round0=float(((V[o]["A0"] + V[o]["B0"]) / 2).mean()),
                                         acc_MAS=float(V[o]["MAS"].mean()), acc_NOCOMM=float(V[o]["NOCOMM"].mean()))
        res["covariates"][int(o)] = {k: float(v) for k, v in res["covariates"][int(o)].items()}
    lo, hi = res["primary"]["slope_P_int"][1:]
    res["reading"] = ("communication gain decreases as evidence overlap increases" if hi < 0 else
                      "communication gain increases as evidence overlap increases" if lo > 0 else "not established")
    pp = lambda v: f"{v[0]*100:+.2f} pp [{v[1]*100:+.2f}, {v[2]*100:+.2f}]"
    L = [f"=== W4 evidence overlap, FRAMES final n={n} (judge-based) ===",
         f"PRIMARY slope of P-int per shared passage = {pp(res['primary']['slope_P_int'])}  -> {res['reading']}",
         f"P-int(0) − P-int(5) = {pp(res['secondary']['P_int_0_minus_5'])}; slope MAS {pp(res['secondary']['slope_MAS'])}; slope No-comm {pp(res['secondary']['slope_NOCOMM'])}",
         f"{'o':>2} {'P-int':>24} {'MAS':>6} {'NoCm':>6} {'r0acc':>6} {'agentGold':>9} {'teamAny':>7} {'teamAll':>7} {'r0agree':>7} {'synM':>5} {'synN':>5}"]
    for o in os_:
        c = res["covariates"][int(o)]
        L.append(f"{o:>2} {pp(res['secondary']['P_int'][int(o)]):>24} {c['acc_MAS']:6.3f} {c['acc_NOCOMM']:6.3f} {c['acc_round0']:6.3f} "
                 f"{c['agent_any_gold']:9.3f} {c['team_any_gold']:7.3f} {c['team_all_gold']:7.3f} {c['round0_answer_agreement']:7.3f} "
                 f"{c['synth_MAS']:5.3f} {c['synth_NOCOMM']:5.3f}")
    open(f"{d}/RESULTS_overlap.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{d}/RESULTS_overlap.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
