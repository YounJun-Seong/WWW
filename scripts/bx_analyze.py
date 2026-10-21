"""BX analyses, exactly the contrasts fixed in DESIGN_BX.md (tag bx-freeze). POST-HOC w.r.t. the original confirmatory
reads; pre-specified within BX. Conventions as RX: FRAMES = frozen judge, HotpotQA = EM, paired bootstrap over questions
(10,000, seed 0); agent-level units resampled by question.
  python bx_analyze.py b14 frames|hotpot     access ladder + equal-budget transfer
  python bx_analyze.py b3 hotpot             support recall x placement
  python bx_analyze.py b2 frames|hotpot      deployable policy (dev = pilot, eval = final)
-> runs/bx/RESULTS_<exp>_<ds>.{txt,json}"""
import json
import os
import sys

import numpy as np

from common import ROOT, normalize
from rx_analyze import Score, boot, cboot, pp, rd

BX = f"{ROOT}/runs/bx"
tk = lambda c: c["input_tokens"] + c["output_tokens"]


def save(name, res, L):
    open(f"{BX}/RESULTS_{name}.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{BX}/RESULTS_{name}.json", "w"), indent=1)
    print("\n".join(L))


def arm(ds, path_gen, path_judged):
    g = rd(path_gen)
    assert all(x.get("error") is None for x in g.values()), f"generation errors in {path_gen}"
    return g, Score(ds).fn(path_judged)


def synth(r):
    return r.get("calls", [])


# ----------------------------------------------------------------------------------------------------------- B1 + B4
def b14(ds):
    G0, c0 = arm(ds, f"runs/final/{ds}_final_gen.jsonl", f"runs/final/{ds}_final_judged.jsonl")
    E1 = {m: arm(ds, f"runs/e1/{ds}_{m}_gen.jsonl", f"runs/e1/{ds}_{m}_judged.jsonl") for m in ("ANS", "ANS_RSN", "ANS_EV", "EV")}
    UNI = arm(ds, f"runs/rx/{ds}_UNI_gen.jsonl", f"runs/rx/{ds}_UNI_judged.jsonl")
    BXA = {m: arm(ds, f"{BX}/{ds}_{m}_gen.jsonl", f"{BX}/{ds}_{m}_judged.jsonl")
           for m in [f"{f}_{k}" for f in ("PEER_EV", "CQ_Q", "CQ_GEN") for k in (1, 3, 5)] + ["LOCAL_SC"]}
    ids = sorted(i for i in G0 if all(not g[i].get("skipped") for g, _ in list(BXA.values()) + list(E1.values()) + [UNI]))
    n = len(ids); ci = boot(n)
    S = {}          # system -> dict(acc=array per question, calls, tokens, retr, moved, depth, regime, kind)

    def add(name, regime, kind, acc, calls, tokens, retr, moved, depth):
        S[name] = dict(regime=regime, kind=kind, acc=np.asarray(acc, float), calls=float(np.mean(calls)),
                       tokens=float(np.mean(tokens)), retr=retr, moved=moved, depth=float(np.mean(depth)))

    two = lambda f: [np.mean([f(i, s) for s in "AB"]) for i in ids]          # single-agent: mean of the 2 deployments
    # R0 local
    add("LOCAL_ONE", "R0", "single", two(lambda i, s: c0(i, G0[i], f"{s}0")), [1] * n, two(lambda i, s: tk(G0[i][f"{s}0"])), 1, 0, [1] * n)
    gsc, csc = BXA["LOCAL_SC"]
    add("LOCAL_SC", "R0", "single", two(lambda i, s: csc(i, gsc[i], f"{s}_SC")), [4] * n,
        two(lambda i, s: sum(tk(x) for x in gsc[i][f"{s}_SC"]["samples"])), 1, 0, [1] * n)
    add("LOCAL_REV", "R0", "single", two(lambda i, s: c0(i, G0[i], f"{s}1n")), [2] * n,
        two(lambda i, s: tk(G0[i][f"{s}0"]) + tk(G0[i][f"{s}1n"])), 1, 0, [2] * n)

    def team(g, c, role, rev, moved, regime, name, pre=()):
        cl = lambda i: [g[i]["A0"] if "A0" in g[i] else G0[i]["A0"], g[i]["B0"] if "B0" in g[i] else G0[i]["B0"],
                        *[g[i][p][s] for p in pre for s in "AB"], g[i][f"A1{rev}"], g[i][f"B1{rev}"], *synth(g[i][role])]
        add(name, regime, "team", [c(i, g[i], role) for i in ids], [len(cl(i)) for i in ids],
            [sum(tk(x) for x in cl(i)) for i in ids], 2 + (2 if pre else 0), moved,
            [2 + len(pre) + bool(synth(g[i][role])) for i in ids])
    team(G0, c0, "NOCOMM", "n", 0, "R0", "NOCOMM")
    for m, reg, mv in (("ANS", "R1", 0), ("ANS_RSN", "R1", 0), ("ANS_EV", "R2", 10), ("EV", "R2", 10)):
        team(*E1[m], "MAS", "m", mv, reg, f"E1_{m}")
    team(G0, c0, "MAS", "m", 10, "R2", "FULL(MAS)")
    for k in (1, 3, 5):
        team(*BXA[f"PEER_EV_{k}"], "MAS", "m", 2 * k, "R2", f"PEER_EV_{k}")
    for fam in ("CQ_Q", "CQ_GEN"):
        for k in (1, 3, 5):
            g, c = BXA[f"{fam}_{k}"]; pre = ("qgen",) if fam == "CQ_GEN" else ()
            add(f"{fam}_{k}", "R3", "single", two(lambda i, s: c(i, g[i], f"{s}1m")), [2 + len(pre)] * n,
                two(lambda i, s: tk(G0[i][f"{s}0"]) + sum(tk(g[i][p][s]) for p in pre) + tk(g[i][f"{s}1m"])),
                2, k, [2 + len(pre)] * n)
            team(g, c, "MAS", "m", 2 * k, "R3", f"{fam}_{k}_team", pre)
    gu, cu = UNI
    add("UNI_ONE", "R4", "single", [cu(i, gu[i], "UNI_ONE") for i in ids], [1] * n, [tk(gu[i]["UNI_ONE"]) for i in ids], 2, 5, [1] * n)
    add("UNI_REV", "R4", "single", [cu(i, gu[i], "UNI_REV") for i in ids], [2] * n,
        [tk(gu[i]["UNI_ONE"]) + tk(gu[i]["UNI_REV"]) for i in ids], 2, 5, [2] * n)
    add("UNI_SC", "R4", "single", [cu(i, gu[i], "UNI_SC") for i in ids], [gu[ids[0]]["UNI_SC"]["k"]] * n,
        [sum(tk(x) for x in gu[i]["UNI_SC"]["samples"]) for i in ids], 2, 5, [1] * n)
    add("POOL", "R5", "single", [c0(i, G0[i], "POOL") for i in ids], [1] * n, [tk(G0[i]["POOL"]) for i in ids], 1, None, [1] * n)
    add("SC", "R5", "single", [c0(i, G0[i], "SC") for i in ids], [len(G0[ids[0]]["SC"]["samples"])] * n,
        [sum(tk(x) for x in G0[i]["SC"]["samples"]) for i in ids], 1, None, [1] * n)

    # agent-level units for B4-P1 / P2
    units = [(i, s) for i in ids for s in "AB"]; cb = cboot([i for i, _ in units])
    ag = lambda m: np.array([BXA[m][1](i, BXA[m][0][i], f"{s}1m") for i, s in units], float)
    res = dict(dataset=ds, n=n, systems={k: {**{kk: vv for kk, vv in v.items() if kk != "acc"}, "accuracy": ci(v["acc"])} for k, v in S.items()})
    p1, _ = cb(ag("CQ_Q_5") - ag("PEER_EV_5"))
    res["B4_P1_framing_CQ_Q5_minus_PEER_EV5_agent"] = dict(diff=p1, within_pm2pp=bool(p1[1] > -0.02 and p1[2] < 0.02))
    res["B4_P2_targeting_CQ_GEN_minus_PEER_EV_agent"] = {k: cb(ag(f"CQ_GEN_{k}") - ag(f"PEER_EV_{k}"))[0] for k in (1, 3, 5)}
    res["B4_agent_acc"] = {m: cb(ag(m))[0] for m in BXA if m != "LOCAL_SC"}
    res["B4_P3_team_curve"] = {f"{f}_{k}": float(S[f"{f}_{k}" if f == "PEER_EV" else f"{f}_{k}_team"]["acc"].mean())
                               for f in ("PEER_EV", "CQ_Q", "CQ_GEN") for k in (1, 3, 5)}
    qg = BXA["CQ_GEN_5"][0]
    res["query_stats"] = dict(parsed=float(np.mean([qg[i]["meta"][s]["query_parsed"] for i in ids for s in "AB"])),
                              overlap_with_peer_top5_k5=float(np.mean([qg[i]["meta"][s]["overlap_with_peer_top5"] for i in ids for s in "AB"])),
                              gold_sent={m: float(np.mean([BXA[m][0][i]["meta"][s]["gold_sent"] for i in ids for s in "AB"]))
                                         for m in ("PEER_EV_1", "PEER_EV_3", "PEER_EV_5", "CQ_GEN_1", "CQ_GEN_3", "CQ_GEN_5")})
    # B1: best admissible per cumulative regime, recovery ratio, frontiers
    order = ["R0", "R1", "R2", "R3", "R4", "R5"]
    best = {}
    for j, r in enumerate(order):
        adm = {k: v for k, v in S.items() if v["regime"] in order[:j + 1]}
        b = max(adm, key=lambda k: adm[k]["acc"].mean()); best[r] = b

        def front(cost):
            return sorted(k for k, v in adm.items() if not any(
                w["acc"].mean() >= v["acc"].mean() and w[cost] <= v[cost] and (w["acc"].mean() > v["acc"].mean() or w[cost] < v[cost])
                for w in adm.values()))
        res.setdefault("B1_P2_frontiers", {})[r] = dict(best=b, tokens=front("tokens"), calls=front("calls"), retr=front("retr"))
    res["B1_P1_best_per_regime"] = {r: dict(system=best[r], accuracy=ci(S[best[r]]["acc"])) for r in order}
    r0, r2, r4 = (S[best[r]]["acc"] for r in ("R0", "R2", "R4"))
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (10000, n))
    num, den = r2[idx].mean(1) - r0[idx].mean(1), r4[idx].mean(1) - r0[idx].mean(1)
    rr = num / np.where(np.abs(den) < 1e-9, np.nan, den)
    res["B1_P1_recovery_ratio"] = [float((r2.mean() - r0.mean()) / (r4.mean() - r0.mean())) if r4.mean() != r0.mean() else None,
                                   *map(float, np.nanpercentile(rr, [2.5, 97.5]))]
    L = [f"=== B1+B4 access regimes and equal-budget transfer, {ds} final n={n} (BX, pre-specified; POST-HOC overall) ===",
         f"{'system':22s} {'reg':4s} {'kind':6s} {'acc':>6s} {'95% CI':>16s} {'calls':>6s} {'tokens':>7s} {'retr':>4s} {'moved':>5s} {'depth':>5s}"]
    for k, v in S.items():
        a = ci(v["acc"])
        L.append(f"{k:22s} {v['regime']:4s} {v['kind']:6s} {a[0]:6.3f} [{a[1]:.3f}, {a[2]:.3f}] {v['calls']:6.2f} {v['tokens']:7.0f} "
                 f"{v['retr']:>4} {str(v['moved']) if v['moved'] is not None else '-':>5s} {v['depth']:5.2f}")
    L += [f"B4-P1 framing (agent level) CQ_Q_5 − PEER_EV_5 = {pp(p1)}; within ±2 pp: {res['B4_P1_framing_CQ_Q5_minus_PEER_EV5_agent']['within_pm2pp']}",
          "B4-P2 targeting (agent level) CQ_GEN_k − PEER_EV_k: " + "; ".join(f"k={k} {pp(v)}" for k, v in res["B4_P2_targeting_CQ_GEN_minus_PEER_EV_agent"].items()),
          "agent-level accuracy: " + ", ".join(f"{m} {v[0]:.3f}" for m, v in res["B4_agent_acc"].items()),
          "B4-P3 team accuracy by k: " + ", ".join(f"{m} {v:.3f}" for m, v in res["B4_P3_team_curve"].items()),
          f"query stats: parsed {res['query_stats']['parsed']:.3f}; CQ_GEN_5 overlap with peer top-5 {res['query_stats']['overlap_with_peer_top5_k5']:.2f}/5; "
          "gold passages delivered per agent: " + ", ".join(f"{m} {v:.2f}" for m, v in res["query_stats"]["gold_sent"].items()),
          "B1-P1 best admissible system per cumulative regime: " + "; ".join(f"{r} {v['system']} {v['accuracy'][0]:.3f}" for r, v in res["B1_P1_best_per_regime"].items()),
          f"B1-P1 recovery ratio (best R0–R2 − best R0) / (best R0–R4 − best R0) = {res['B1_P1_recovery_ratio'][0]} "
          f"[{res['B1_P1_recovery_ratio'][1]:.2f}, {res['B1_P1_recovery_ratio'][2]:.2f}] (systems selected by point estimate)"]
    for r, f in res["B1_P2_frontiers"].items():
        L.append(f"B1-P2 frontier R0–{r}: tokens {f['tokens']} | calls {f['calls']} | retrieval calls {f['retr']}")
    return res, L


# ---------------------------------------------------------------------------------------------------------------- B3
def b3(ds="hotpot"):
    cells = ["NONE", "P_EXCL", "P_SHAR", "F_SPLIT", "F_CONC", "F_SHAR"]
    C = {c: arm(ds, f"{BX}/hotpot_B3_{c}_gen.jsonl", "") for c in cells}
    ids = sorted(i for i in C["NONE"][0] if all(not C[c][0][i].get("skipped") for c in cells)); n = len(ids); ci = boot(n)
    acc = {c: {r: np.array([C[c][1](i, C[c][0][i], r) for i in ids]) for r in ("MAS", "NOCOMM", "A0", "B0")} for c in cells}
    pint = {c: acc[c]["MAS"] - acc[c]["NOCOMM"] for c in cells}
    hold = lambda c, which: np.array([C[c][1](i, C[c][0][i], ("A0" if (C[c][0][i]["holder"] == "A") == (which == "H") else "B0")) for i in ids])
    res = dict(dataset=ds, n=n, cells={c: dict(MAS=ci(acc[c]["MAS"]), NOCOMM=ci(acc[c]["NOCOMM"]), P_int=ci(pint[c]),
                                             round0_holder=float(hold(c, "H").mean()), round0_other=float(hold(c, "O").mean()))
                                     for c in cells})
    res["B3_P1_F_SPLIT_minus_F_SHAR"] = ci(pint["F_SPLIT"] - pint["F_SHAR"])
    res["B3_P2_P_EXCL_minus_P_SHAR"] = ci(pint["P_EXCL"] - pint["P_SHAR"])
    res["B3_P3_F_SPLIT_minus_F_CONC"] = ci(pint["F_SPLIT"] - pint["F_CONC"])
    res["B3_P4"] = dict(retrieval_floor_MAS_NONE=ci(acc["NONE"]["MAS"]), utilisation_failure_F_SHAR=ci(1 - acc["F_SHAR"]["MAS"]),
                        by_recall={"0": dict(MAS=float(acc["NONE"]["MAS"].mean()), NOCOMM=float(acc["NONE"]["NOCOMM"].mean())),
                                   "1": {r: float(np.mean([acc[c][r].mean() for c in ("P_EXCL", "P_SHAR")])) for r in ("MAS", "NOCOMM")},
                                   "2": {r: float(np.mean([acc[c][r].mean() for c in ("F_SPLIT", "F_CONC", "F_SHAR")])) for r in ("MAS", "NOCOMM")}})
    L = [f"=== B3 support recall x placement, HotpotQA final n={n} (EM; BX pre-specified; POST-HOC overall) ===",
         f"{'cell':8s} {'MAS':>6s} {'NoComm':>7s}  P-int                      round-0 holder / other"]
    for c in cells:
        v = res["cells"][c]
        L.append(f"{c:8s} {v['MAS'][0]:6.3f} {v['NOCOMM'][0]:7.3f}  {pp(v['P_int']):26s} {v['round0_holder']:.3f} / {v['round0_other']:.3f}")
    L += [f"B3-P1 P-int(F_SPLIT) − P-int(F_SHAR) = {pp(res['B3_P1_F_SPLIT_minus_F_SHAR'])}",
          f"B3-P2 P-int(P_EXCL) − P-int(P_SHAR) = {pp(res['B3_P2_P_EXCL_minus_P_SHAR'])}",
          f"B3-P3 P-int(F_SPLIT) − P-int(F_CONC) = {pp(res['B3_P3_F_SPLIT_minus_F_CONC'])}",
          f"B3-P4 retrieval floor (MAS acc, NONE) {res['B3_P4']['retrieval_floor_MAS_NONE'][0]:.3f}; "
          f"utilisation failure (1 − MAS acc, F_SHAR) {res['B3_P4']['utilisation_failure_F_SHAR'][0]:.3f} "
          f"[{res['B3_P4']['utilisation_failure_F_SHAR'][1]:.3f}, {res['B3_P4']['utilisation_failure_F_SHAR'][2]:.3f}]",
          "team accuracy by recall (MAS / NoComm): " + "; ".join(f"{k}: {v['MAS']:.3f} / {v['NOCOMM']:.3f}" for k, v in res["B3_P4"]["by_recall"].items())]
    return res, L


# ---------------------------------------------------------------------------------------------------------------- B2
def b2(ds):
    from sklearn.linear_model import LogisticRegression
    from rx_offline import HEDGE, calls, content_novelty, load_run, score_gap
    feats, Sx = ["disagree", "hedge", "gap", "nov", "missing", "alen"], {}
    for split in ("pilot", "final"):
        g, c = load_run(ds, split)
        ids = sorted(i for i in g if g[i].get("error") is None)
        gap, nov = score_gap(ds, split), content_novelty(ds, split)
        ans = lambda i, s: g[i][f"{s}0"]["answer"] or ""
        X = np.array([[float(not (ans(i, "A") and ans(i, "B") and normalize(ans(i, "A")) == normalize(ans(i, "B")))),
                       len(HEDGE.findall(g[i]["A0"]["text"] or "")) + len(HEDGE.findall(g[i]["B0"]["text"] or "")),
                       gap[i], nov[i], float(not ans(i, "A") or not ans(i, "B")),
                       (len(ans(i, "A").split()) + len(ans(i, "B").split())) / 2] for i in ids])
        Sx[split] = dict(ids=ids, g=g, X=X, mas=np.array([c(i, g[i], "MAS") for i in ids]), nc=np.array([c(i, g[i], "NOCOMM") for i in ids]))
    P, F = Sx["pilot"], Sx["final"]
    mu, sd = P["X"].mean(0), P["X"].std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((P["X"] - mu) / sd, (P["mas"] - P["nc"] > 0).astype(int))
    score = {"DISAGREE-RANK": lambda S: S["X"][:, 0] * 10 + S["X"][:, 3], "NOVELTY-RANK": lambda S: S["X"][:, 3],
             "EVIDENCE-AWARE": lambda S: lr.predict_proba((S["X"] - mu) / sd)[:, 1]}
    tok_of = lambda S, comm: np.array([sum(tk(x) for x in calls(S["g"][i], "MAS" if m else "NC")) for i, m in zip(S["ids"], comm)], float)
    ci = boot(len(F["ids"])); res = dict(dataset=ds, n_pilot=len(P["ids"]), n_final=len(F["ids"]), features=feats,
                                         coef=dict(zip(feats, map(float, lr.coef_[0]))), budgets={})
    L = [f"=== B2 deployable communication policy, {ds}: fitted / thresholded on pilot n={len(P['ids'])}, read on final n={len(F['ids'])} (BX pre-specified) ===",
         "logistic coefficients (standardised): " + ", ".join(f"{k} {v:+.2f}" for k, v in res["coef"].items()),
         f"AlwaysComm {F['mas'].mean():.3f}  NeverComm {F['nc'].mean():.3f}"]
    for b in (0.2, 0.4, 0.6, 0.8):
        row = {}
        for name, f in score.items():
            thr = np.quantile(f(P), 1 - b); comm = f(F) >= thr
            acc = np.where(comm, F["mas"], F["nc"]); a = comm.mean()
            rnd_b = b * F["mas"] + (1 - b) * F["nc"]; rnd_a = a * F["mas"] + (1 - a) * F["nc"]
            row[name] = dict(activation=float(a), accuracy=float(acc.mean()), tokens=float(tok_of(F, comm).mean()),
                             minus_RANDOM_at_b=ci(acc - rnd_b), minus_RANDOM_at_realised=ci(acc - rnd_a), _acc=acc)
        for name in row:
            row[name]["minus_DISAGREE_RANK"] = ci(row[name]["_acc"] - row["DISAGREE-RANK"]["_acc"])
        res["budgets"][str(b)] = {k: {kk: vv for kk, vv in v.items() if kk != "_acc"} for k, v in row.items()}
        L.append(f"budget b={b}: RANDOM acc {float((b * F['mas'] + (1 - b) * F['nc']).mean()):.3f}")
        for name, v in row.items():
            L.append(f"  {name:15s} act {v['activation']:.3f} acc {v['accuracy']:.3f} tok {v['tokens']:6.0f} | − RANDOM(b) {pp(v['minus_RANDOM_at_b'])}"
                     f" | − RANDOM(realised) {pp(v['minus_RANDOM_at_realised'])} | − DISAGREE-RANK {pp(v['minus_DISAGREE_RANK'])}")
    # secondary: fixed routing rules
    gq = rd(f"{BX}/{ds}_CQ_GEN_5_gen.jsonl"); cq = Score(ds).fn(f"{BX}/{ds}_CQ_GEN_5_judged.jsonl")
    dis = F["X"][:, 0] > 0
    cqa = np.array([cq(i, gq[i], "MAS") for i in F["ids"]])
    cq_tok = np.array([sum(tk(x) for x in [gq[i]["A0"], gq[i]["B0"], gq[i]["qgen"]["A"], gq[i]["qgen"]["B"], gq[i]["A1m"], gq[i]["B1m"], *gq[i]["MAS"].get("calls", [])]) for i in F["ids"]], float)
    rp = np.where(dis, F["mas"], F["nc"]); rr = np.where(dis, cqa, F["nc"])
    nc_tok = tok_of(F, np.zeros(len(dis), bool)); mas_tok = tok_of(F, np.ones(len(dis), bool))
    res["secondary"] = dict(activation=float(dis.mean()), ROUTE_PEER=dict(accuracy=float(rp.mean()), tokens=float(np.where(dis, mas_tok, nc_tok).mean())),
                            ROUTE_RETRIEVE=dict(accuracy=float(rr.mean()), tokens=float(np.where(dis, cq_tok, nc_tok).mean())),
                            RETRIEVE_minus_PEER=ci(rr - rp))
    s = res["secondary"]
    L.append(f"secondary (fixed rules, activation {s['activation']:.3f}): ROUTE-PEER acc {s['ROUTE_PEER']['accuracy']:.3f} tok {s['ROUTE_PEER']['tokens']:.0f}; "
             f"ROUTE-RETRIEVE acc {s['ROUTE_RETRIEVE']['accuracy']:.3f} tok {s['ROUTE_RETRIEVE']['tokens']:.0f}; RETRIEVE − PEER {pp(s['RETRIEVE_minus_PEER'])}")
    wins = [b for b, r in res["budgets"].items() if r["EVIDENCE-AWARE"]["minus_RANDOM_at_realised"][1] > 0 and r["EVIDENCE-AWARE"]["minus_DISAGREE_RANK"][1] > 0]
    res["promotion_budgets"] = wins
    L.append(f"budgets where EVIDENCE-AWARE beats both RANDOM (realised) and DISAGREE-RANK (CI excludes 0): {wins} (promotion needs ≥ 2 on both datasets)")
    return res, L


if __name__ == "__main__":
    exp, ds = sys.argv[1], sys.argv[2]
    res, L = dict(b14=b14, b3=b3, b2=b2)[exp](ds)
    save(f"{exp}_{ds}", res, L)
