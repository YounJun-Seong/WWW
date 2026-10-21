"""HQ analyses (DESIGN_HQ.md, tag hq-freeze). EM (frozen normalisation) primary, token F1 secondary.
  python hq_analyze.py   -> runs/hq/RESULTS_hq.{txt,json}"""
import json

import numpy as np

from common import ROOT, f1, normalize
from rx_analyze import boot, cboot, pp

OUT = f"{ROOT}/runs/hq"


def rd(p):
    return {r["k"]: r for r in map(json.loads, open(p))}


def main():
    S = {s["k"]: s for s in map(json.loads, open(f"{ROOT}/data/hq/sample.jsonl"))}
    G = {m: rd(f"{OUT}/hq_{m}_gen.jsonl") for m in ("BASE", "PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN")}
    for m, g in G.items():
        assert all(r.get("error") is None for r in g.values()), m
    ks = sorted(set.intersection(*[set(g) for g in G.values()])); n = len(ks); ci = boot(n)
    em = lambda k, a: float(bool(a) and normalize(a) == normalize(S[k]["answer"]))
    fs = lambda k, a: float(f1(a or "", S[k]["answer"]))
    team = {"LOCAL_NOCOMM": lambda k: G["BASE"][k]["NOCOMM"]["answer"], "UNI": lambda k: G["BASE"][k]["UNI"]["answer"],
            **{m: (lambda k, m=m: G[m][k]["MAS"]["answer"]) for m in ("PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN")}}
    acc = {c: np.array([em(k, f(k)) for k in ks]) for c, f in team.items()}
    accf = {c: np.array([fs(k, f(k)) for k in ks]) for c, f in team.items()}
    r0 = {s: np.array([em(k, G["BASE"][k][f"{s}0"]["answer"]) for k in ks]) for s in "AB"}
    res = dict(n=n, accuracy={c: ci(v) for c, v in acc.items()}, f1={c: float(v.mean()) for c, v in accf.items()},
               round0=dict(A_tables=float(r0["A"].mean()), B_passages=float(r0["B"].mean())))
    res["P1_comm_gain"] = ci(acc["PEER_EV"] - acc["LOCAL_NOCOMM"])
    res["P2_policy_PEER_EV_minus_REMOTE_GEN"] = ci(acc["PEER_EV"] - acc["REMOTE_GEN"])
    units = [(k, s) for k in ks for s in "AB"]; cb = cboot([k for k, _ in units])
    ag = lambda m: np.array([em(k, G[m][k][f"{s}1m"]["answer"]) for k, s in units])
    d3 = ag("REMOTE_Q") - ag("PEER_EV"); c95, _ = cb(d3)
    rng = np.random.default_rng(0); uq = sorted(set(ks)); pos = {k: i for i, k in enumerate(uq)}
    per_q = np.zeros(len(uq)); cnt = np.zeros(len(uq))
    for (k, _), v in zip(units, d3):
        per_q[pos[k]] += v; cnt[pos[k]] += 1
    idx = rng.integers(0, len(uq), (10000, len(uq))); bs = per_q[idx].sum(1) / cnt[idx].sum(1)
    c90 = [c95[0], float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    res["P3_matched_REMOTE_Q_minus_PEER_EV_agent"] = dict(ci95=c95, ci90=c90, equivalent_95=bool(c95[1] > -0.02 and c95[2] < 0.02),
                                                          equivalent_TOST90=bool(c90[1] > -0.02 and c90[2] < 0.02))
    res["references"] = dict(PEER_FULL_minus_LOCAL=ci(acc["PEER_FULL"] - acc["LOCAL_NOCOMM"]), UNI_minus_PEER_EV=ci(acc["UNI"] - acc["PEER_EV"]),
                             REMOTE_GEN_minus_LOCAL=ci(acc["REMOTE_GEN"] - acc["LOCAL_NOCOMM"]))
    # coverage diagnostics (traced answer nodes = distant supervision)
    A_has = np.array([S[k]["A_has_sup"] for k in ks]); B_has = np.array([S[k]["B_has_sup"] for k in ks])
    any_has = A_has | B_has
    gen_new = np.array([any(i in S[k]["sup_passages"] for i in G["REMOTE_GEN"][k]["meta"]["A"]["received_ids"]) or
                        any(i in S[k]["sup_rows"] for i in G["REMOTE_GEN"][k]["meta"]["B"]["received_ids"]) for k in ks])
    res["coverage"] = dict(A_own_traced_row=float(A_has.mean()), B_own_traced_passage=float(B_has.mean()),
                           team_any_traced=float(any_has.mean()),
                           peer_delivers_traced_to_A=float(B_has.mean()), peer_delivers_traced_to_B=float(A_has.mean()),
                           remote_gen_retrieves_traced=float(gen_new.mean()),
                           comm_gain_by_team_coverage={str(bool(v)): boot(int((any_has == v).sum()))((acc["PEER_EV"] - acc["LOCAL_NOCOMM"])[any_has == v]) if (any_has == v).sum() > 1 else None
                                                       for v in (True, False)})
    # candidate stages per condition
    stages = {}
    for m in ("PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN"):
        cand = np.array([max(em(k, G[m][k]["A1m"]["answer"]), em(k, G[m][k]["B1m"]["answer"])) for k in ks]); fin = acc[m]
        stages[m] = dict(preserved=int(((cand == 1) & (fin == 1)).sum()), lost=int(((cand == 1) & (fin == 0)).sum()),
                         created=int(((cand == 0) & (fin == 1)).sum()), none=int(((cand == 0) & (fin == 0)).sum()))
    cand = np.array([max(em(k, G["BASE"][k]["A1n"]["answer"]), em(k, G["BASE"][k]["B1n"]["answer"])) for k in ks]); fin = acc["LOCAL_NOCOMM"]
    stages["LOCAL_NOCOMM"] = dict(preserved=int(((cand == 1) & (fin == 1)).sum()), lost=int(((cand == 1) & (fin == 0)).sum()),
                                  created=int(((cand == 0) & (fin == 1)).sum()), none=int(((cand == 0) & (fin == 0)).sum()))
    res["stages"] = stages
    tok = lambda r, keys: sum(r[x]["input_tokens"] + r[x]["output_tokens"] for x in keys)
    res["tokens_per_question"] = {
        "LOCAL_NOCOMM": float(np.mean([tok(G["BASE"][k], ["A0", "B0", "A1n", "B1n"]) + sum(c["input_tokens"] + c["output_tokens"] for c in G["BASE"][k]["NOCOMM"]["calls"]) for k in ks])),
        "UNI": float(np.mean([tok(G["BASE"][k], ["UNI"]) for k in ks])),
        **{m: float(np.mean([tok(G[m][k], ["A0", "B0", "A1m", "B1m"]) + sum(c["input_tokens"] + c["output_tokens"] for c in G[m][k]["MAS"]["calls"])
                             + (sum(G[m][k]["qgen"][s]["input_tokens"] + G[m][k]["qgen"][s]["output_tokens"] for s in "AB") if m == "REMOTE_GEN" else 0) for k in ks]))
           for m in ("PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN")}}
    L = [f"=== HQ source-constrained allocation, HybridQA dev sample n={n} (A = table rows, B = linked passages; EM; POST-HOC, pre-specified) ===",
         f"round-0 EM: table agent {res['round0']['A_tables']:.3f}, passage agent {res['round0']['B_passages']:.3f}",
         f"{'condition':14s} {'EM [95% CI]':>24s} {'F1':>6s} {'tokens':>7s}"]
    for c in ("LOCAL_NOCOMM", "PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN", "UNI"):
        a = res["accuracy"][c]; L.append(f"{c:14s} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] {res['f1'][c]:6.3f} {res['tokens_per_question'][c]:7.0f}")
    p3 = res["P3_matched_REMOTE_Q_minus_PEER_EV_agent"]
    L += [f"HQ-P1 communication gain PEER_EV − LOCAL_NOCOMM = {pp(res['P1_comm_gain'])}",
          f"HQ-P2 feasible-policy PEER_EV − REMOTE_GEN = {pp(res['P2_policy_PEER_EV_minus_REMOTE_GEN'])}  (not a causal effect of communication)",
          f"HQ-P3 matched REMOTE_Q − PEER_EV (agent) = {pp(p3['ci95'])}; TOST90 [{100 * p3['ci90'][1]:+.1f}, {100 * p3['ci90'][2]:+.1f}]; equivalent ±2pp: 95% {p3['equivalent_95']}, TOST {p3['equivalent_TOST90']}",
          "references: " + "; ".join(f"{k} {pp(v)}" for k, v in res["references"].items()),
          f"coverage (traced answer nodes, distant supervision): table agent own top-5 {res['coverage']['A_own_traced_row']:.3f}; passage agent own top-5 "
          f"{res['coverage']['B_own_traced_passage']:.3f}; team any {res['coverage']['team_any_traced']:.3f}; REMOTE_GEN retrieves traced {res['coverage']['remote_gen_retrieves_traced']:.3f}",
          "comm gain by team coverage: " + "; ".join(f"covered={k} {pp(v) if v else '-'}" for k, v in res["coverage"]["comm_gain_by_team_coverage"].items()),
          "candidate stages (preserved / lost in synthesis / created / none): " + "; ".join(f"{m} {v['preserved']}/{v['lost']}/{v['created']}/{v['none']}" for m, v in stages.items())]
    json.dump(res, open(f"{OUT}/RESULTS_hq.json", "w"), indent=1); open(f"{OUT}/RESULTS_hq.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
