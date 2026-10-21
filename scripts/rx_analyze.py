"""RX analyses (WWW-final round). Same conventions as analyze_e123.py: FRAMES = frozen judge verdicts, HotpotQA = EM;
paired bootstrap over QUESTIONS, 10,000 resamples, seed 0 (agent-level quantities are resampled by question, so the two
agents of a question stay together). Every RX result is POST-HOC with respect to the frozen confirmatory reads.
  python rx_analyze.py <r1|r2|r3|r4|r5|r6|r10|r11|r12> <frames|hotpot>   ->  runs/rx/RESULTS_<exp>_<ds>.{txt,json}"""
import json
import os
import sys

import numpy as np

from common import ROOT, normalize

B = 10000
RX = os.environ.get("RX_DIR", f"{ROOT}/runs/rx")            # RX_DIR=runs/rx32 for the 32B replication
# A7 judge sensitivity (FRAMES only): RX_SCORE=contain | f1 replaces the frozen judge's verdict with a judge-free score of
# the SAME response string the judge saw: contain = normalized gold is a substring of the normalized response;
# f1 = SQuAD token F1 >= 0.5. Outputs get the suffix _<RX_SCORE>.
SCORE = os.environ.get("RX_SCORE", "judge")


def rd(p):
    p = p if p.startswith("/") else f"{ROOT}/{p}"
    return {x["idx"]: x for x in map(json.loads, open(p))}


class Score:
    """correct(ds, idx, gen_row, role, judged_rows) -> 0/1"""
    def __init__(self, ds):
        self.ds = ds
        if ds == "hotpot":
            self.lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))

    def fn(self, judged):
        if self.ds == "frames":
            J = rd(judged)
            if SCORE == "judge":
                return lambda i, g, r: float(J[i][r]["verdict"])
            import pandas as pd
            from common import f1
            gold = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")["Answer"].astype(str)
            if SCORE == "contain":
                return lambda i, g, r: float(bool(normalize(gold[i])) and normalize(gold[i]) in normalize(J[i][r]["response"] or ""))
            assert SCORE == "f1", SCORE
            return lambda i, g, r: float(f1(J[i][r]["response"] or "", gold[i]) >= 0.5)
        return lambda i, g, r: float(bool(g[r].get("answer")) and normalize(g[r]["answer"]) == normalize(self.lab[str(i)]))


def gen_and_score(ds, mode):
    g = rd(f"{RX}/{ds}_{mode}_gen.jsonl")
    assert all(x["error"] is None for x in g.values()), f"{ds} {mode}: generation errors"
    return g, Score(ds).fn(f"{RX}/{ds}_{mode}_judged.jsonl")


def frozen(ds):
    base = os.environ.get("RX_FROZEN", f"runs/final/{ds}_final")   # 32B: runs/backbones/qwen2.5-32b/frames_final
    return rd(f"{base}_gen.jsonl"), Score(ds).fn(f"{base}_judged.jsonl")


def boot(n):
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    return lambda x: [float(np.mean(x)), *map(float, np.percentile(np.asarray(x)[idx].mean(1), [2.5, 97.5]))]


def cboot(qs):
    """cluster bootstrap for agent-level units: qs = list of question ids per unit."""
    uq = sorted(set(qs)); pos = {q: k for k, q in enumerate(uq)}
    members = [[] for _ in uq]
    for u, q in enumerate(qs):
        members[pos[q]].append(u)
    idx = np.random.default_rng(0).integers(0, len(uq), (B, len(uq)))

    def f(x, mask=None):
        x = np.asarray(x, float); m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
        s = np.array([x[[u for u in members[k] if m[u]]].sum() for k in range(len(uq))])
        c = np.array([sum(m[u] for u in members[k]) for k in range(len(uq))], float)
        bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
        return [float(s.sum() / max(c.sum(), 1)), *map(float, np.percentile(bs, [2.5, 97.5]))], int(c.sum())
    return f


pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
pr = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def save(name, res, L):
    if SCORE != "judge":
        name += f"_{SCORE}"; L = [f"[A7 judge-free scoring: {SCORE}]"] + L; res = dict(res, scoring=SCORE)
    open(f"{RX}/RESULTS_{name}.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{RX}/RESULTS_{name}.json", "w"), indent=1)
    print("\n".join(L))


def agent_units(ids, G0, c0, Garm, carm):
    """per (question, receiver): initial correct, revised correct under the arm, meta."""
    U = []
    for i in ids:
        for me, slot in (("A", "A1m"), ("B", "B1m")):
            U.append(dict(q=i, me=me, init=c0(i, G0[i], f"{me}0"), rev=carm(i, Garm[i], slot),
                          meta=Garm[i].get("meta", {}).get(me, {})))
    return U


# ---------------------------------------------------------------- R1 counterfactual removal / replacement
def r1(ds):
    G0, c0 = frozen(ds)
    arms = ("R1_FULL", "R1_GOLDRM", "R1_GOLDREP", "R1_NONGOLDRM")
    GA = {m: gen_and_score(ds, m) for m in arms}
    ids = sorted(i for i in G0 if all(not GA[m][0][i].get("skipped") for m in arms))
    ci = boot(len(ids))
    fin = {m: np.array([GA[m][1](i, GA[m][0][i], "MAS") for i in ids]) for m in arms}
    fin["NC"] = np.array([c0(i, G0[i], "NOCOMM") for i in ids])
    res = dict(dataset=ds, n=len(ids), skipped=len(G0) - len(ids),
               accuracy={m: float(v.mean()) for m, v in fin.items()},
               final=dict(FULL_minus_GOLDREP=ci(fin["R1_FULL"] - fin["R1_GOLDREP"]),
                          FULL_minus_GOLDRM=ci(fin["R1_FULL"] - fin["R1_GOLDRM"]),
                          FULL_minus_NONGOLDRM=ci(fin["R1_FULL"] - fin["R1_NONGOLDRM"]),
                          GOLDREP_minus_NC=ci(fin["R1_GOLDREP"] - fin["NC"]),
                          NONGOLDRM_minus_NC=ci(fin["R1_NONGOLDRM"] - fin["NC"]), FULL_minus_NC=ci(fin["R1_FULL"] - fin["NC"])))
    # agent level: receivers whose partner holds >= 1 gold passage (only there can a removal differ from FULL)
    U = {m: agent_units(ids, G0, c0, *GA[m]) for m in arms}
    qs = [u["q"] for u in U["R1_FULL"]]; cb = cboot(qs)
    pg = [u["meta"].get("partner_gold", 0) > 0 for u in U["R1_FULL"]]
    own_has = []
    for u in U["R1_FULL"]:
        q = G0[u["q"]]
        own_has.append(None)
    res["agent"] = {}
    for sub, mask in (("partner_holds_gold", pg), ("all", [True] * len(qs))):
        d = {}
        for m in arms:
            wrong0 = [u["init"] == 0 for u in U[m]]; right0 = [u["init"] == 1 for u in U[m]]
            corr = [u["rev"] for u in U[m]]; harm = [1 - u["rev"] for u in U[m]]
            mc = [a and b for a, b in zip(mask, wrong0)]; mh = [a and b for a, b in zip(mask, right0)]
            (cv, nc), (hv, nh) = cb(corr, mc), cb(harm, mh)
            d[m] = dict(correction=cv, n_wrong=nc, harm=hv, n_right=nh)
        diff = lambda a, b, key, mk: cb([x["rev"] - y["rev"] for x, y in zip(U[a], U[b])],
                                         [m_ and (x["init"] == (0 if key == "c" else 1)) for m_, x in zip(mk, U[a])])[0]
        d["contrast_correction_FULL_minus_GOLDREP"] = diff("R1_FULL", "R1_GOLDREP", "c", mask)
        d["contrast_correction_FULL_minus_GOLDRM"] = diff("R1_FULL", "R1_GOLDRM", "c", mask)
        d["contrast_correction_FULL_minus_NONGOLDRM"] = diff("R1_FULL", "R1_NONGOLDRM", "c", mask)
        res["agent"][sub] = d
    L = [f"=== R1 counterfactual gold removal / replacement, {ds} final n={len(ids)} ({'judge' if ds == 'frames' else 'EM'}; POST-HOC) ===",
         "gold = " + ("passage from a gold page (page-level, not fact-level)" if ds == "frames" else "supporting-fact paragraph"),
         "final accuracy: " + ", ".join(f"{m} {v:.3f}" for m, v in res["accuracy"].items()),
         *[f"  final {k}: {pp(v)}" for k, v in res["final"].items()]]
    for sub, d in res["agent"].items():
        L.append(f"agent level, {sub}:")
        for m in arms:
            L.append(f"  {m:13s} correction {pr(d[m]['correction'])} (n wrong {d[m]['n_wrong']}); harm {pr(d[m]['harm'])} (n right {d[m]['n_right']})")
        for k in ("FULL_minus_GOLDREP", "FULL_minus_GOLDRM", "FULL_minus_NONGOLDRM"):
            L.append(f"  correction {k}: {pp(d['contrast_correction_' + k])}")
    return res, L


# ---------------------------------------------------------------- R2 injection x sufficiency
def r2(ds):
    G0, c0 = frozen(ds)
    arms = ("R2_I1a", "R2_I1b", "R2_I2", "R2_I3", "R2_I4")
    GA = {m: gen_and_score(ds, m) for m in arms}
    ids = sorted(i for i in G0 if all(not GA[m][0][i].get("skipped") for m in arms))
    U = {m: agent_units(ids, G0, c0, *GA[m]) for m in arms}
    nc = []
    for i in ids:
        for me in "AB":
            nc.append(c0(i, G0[i], f"{me}1n"))
    qs = [u["q"] for u in U["R2_I2"]]; cb = cboot(qs)
    own = [u["meta"].get("own_gold", 0) for u in U["R2_I2"]]
    res = dict(dataset=ds, n_questions=len(ids), n_receivers=len(qs), strata={})
    L = [f"=== R2 evidence injection x sufficiency, {ds} final, questions with |G_inj| = 2: n={len(ids)} (POST-HOC) ===",
         "I0 = the receiver's frozen No-comm revision; I1a/I1b = one gold passage; I2 = both; I3 = both + 3 hard negatives; "
         "I4 = both + partner's answer",
         "gold = " + ("gold-page passage (page-level)" if ds == "frames" else "supporting-fact paragraph (complete set = 2)")]
    I1 = [(a["rev"] + b["rev"]) / 2 for a, b in zip(U["R2_I1a"], U["R2_I1b"])]
    series = {"I0": nc, "I1": I1, **{m[3:]: [u["rev"] for u in U[m]] for m in ("R2_I2", "R2_I3", "R2_I4")}}
    for og in sorted(set(own)) + ["all"]:
        for init in (0, 1):
            mask = [(og == "all" or o == og) and u["init"] == init for o, u in zip(own, U["R2_I2"])]
            if not any(mask):
                continue
            row = {}
            for s, v in series.items():
                row[s], n = cb(v, mask)
            key = f"own_gold={og}|init={'wrong' if init == 0 else 'right'}"
            res["strata"][key] = dict(n=n, **row,
                                      I2_minus_I1=cb([a - b for a, b in zip(series["I2"], series["I1"])], mask)[0],
                                      I1_minus_I0=cb([a - b for a, b in zip(series["I1"], series["I0"])], mask)[0],
                                      I3_minus_I2=cb([a - b for a, b in zip(series["I3"], series["I2"])], mask)[0],
                                      I4_minus_I2=cb([a - b for a, b in zip(series["I4"], series["I2"])], mask)[0])
            r = res["strata"][key]
            L.append(f"{key:30s} n={n:4d}  P(correct after): " + "  ".join(f"{s} {r[s][0]:.3f}" for s in series)
                     + f" | I1−I0 {pp(r['I1_minus_I0'])}; I2−I1 {pp(r['I2_minus_I1'])}; I3−I2 {pp(r['I3_minus_I2'])}; I4−I2 {pp(r['I4_minus_I2'])}")
    m = [u["init"] == 0 for u in U["R2_I2"]]
    res["integration_failure_complete_support"] = cb([1 - x for x in series["I2"]], m)[0]
    L.append(f"integration failure: P(still wrong | initially wrong, full gold set injected) = {pr(res['integration_failure_complete_support'])}")
    return res, L


# ---------------------------------------------------------------- R3 HotpotQA E2 replication
def r3(ds="hotpot"):
    cells = ("EXCL_LOW", "EXCL_HIGH", "SHAR_LOW", "SHAR_HIGH")
    GA = {c: gen_and_score(ds, f"E2H_{c}") for c in cells}
    ids = sorted(i for i in GA[cells[0]][0] if not GA[cells[0]][0][i].get("skipped"))
    ci = boot(len(ids))
    V = {c: {r: np.array([GA[c][1](i, GA[c][0][i], r) for i in ids]) for r in ("MAS", "NOCOMM", "A0", "B0")} for c in cells}
    P = {c: V[c]["MAS"] - V[c]["NOCOMM"] for c in cells}
    place = ((P["EXCL_LOW"] - P["SHAR_LOW"]) + (P["EXCL_HIGH"] - P["SHAR_HIGH"])) / 2
    over = ((P["EXCL_LOW"] - P["EXCL_HIGH"]) + (P["SHAR_LOW"] - P["SHAR_HIGH"])) / 2
    inter = (P["EXCL_LOW"] - P["EXCL_HIGH"]) - (P["SHAR_LOW"] - P["SHAR_HIGH"])
    X = rd("data/rx/assign_hotpot.jsonl")
    res = dict(n=len(ids), n_total=len(X), excluded={str(k): v for k, v in
               __import__("collections").Counter(f"g={x['e2h'].get('g')},fill={x['e2h'].get('n_fill')}" for x in X.values() if not x["e2h"]["eligible"]).items()},
               placement=ci(place), overlap=ci(over), interaction=ci(inter), cells={})
    for c in cells:
        G = GA[c][0]
        agree = np.mean([bool(G[i]["A0"]["answer"]) and normalize(G[i]["A0"]["answer"]) == normalize(G[i]["B0"]["answer"] or "") for i in ids])
        res["cells"][c] = dict(P_int=ci(P[c]), MAS=float(V[c]["MAS"].mean()), NOCOMM=float(V[c]["NOCOMM"].mean()),
                               round0_acc=float(((V[c]["A0"] + V[c]["B0"]) / 2).mean()), round0_agree=float(agree),
                               shared=float(np.mean([len({t for t in G[i]["ids"]["A"]} & {t for t in G[i]["ids"]["B"]}) for i in ids])),
                               gold_A=float(np.mean([G[i]["gold"]["A"] for i in ids])), gold_B=float(np.mean([G[i]["gold"]["B"] for i in ids])))
    L = [f"=== R3 HotpotQA E2 replication (overlap x gold placement, complete 2-paragraph gold set), eligible n={len(ids)}/{len(X)} (EM; POST-HOC) ===",
         f"placement effect (EXCL − SHAR) = {pp(res['placement'])}", f"overlap effect (LOW − HIGH) = {pp(res['overlap'])}",
         f"interaction = {pp(res['interaction'])}", f"excluded: {res['excluded']}",
         *[f"  {c:9s} P-int {pp(v['P_int'])}; MAS {v['MAS']:.3f} No-comm {v['NOCOMM']:.3f}; round-0 acc {v['round0_acc']:.3f} agree {v['round0_agree']:.3f}; shared {v['shared']:.1f}; gold A {v['gold_A']:.2f} B {v['gold_B']:.2f}"
           for c, v in res["cells"].items()]]
    return res, L


# ---------------------------------------------------------------- R4 matched context
def r4(ds):
    G0, c0 = frozen(ds)
    dist = f"runs/dist{'_hotpot' if ds == 'hotpot' else ''}/S_UNION_final"
    GS, cS = rd(f"{dist}_gen.jsonl"), Score(ds).fn(f"{dist}_judged.jsonl")
    GP, cP = gen_and_score(ds, "DPAD")
    ids = sorted(i for i in G0 if not GP[i].get("skipped") and i in GS)
    ci = boot(len(ids))
    pint = lambda G, c: np.array([c(i, G[i], "MAS") - c(i, G[i], "NOCOMM") for i in ids])
    acc = lambda G, c, r: float(np.mean([c(i, G[i], r) for i in ids]))
    pD, pS, pP = pint(G0, c0), pint(GS, cS), pint(GP, cP)
    res = dict(dataset=ds, n=len(ids), P_int=dict(D=ci(pD), S_UNION=ci(pS), DPAD=ci(pP)),
               DiD_original=ci(pD - pS), DiD_padded=ci(pP - pS), DPAD_minus_D=ci(pP - pD),
               accuracy={k: dict(MAS=acc(G, c, "MAS"), NOCOMM=acc(G, c, "NOCOMM")) for k, (G, c) in
                         dict(D=(G0, c0), S_UNION=(GS, cS), DPAD=(GP, cP)).items()})
    L = [f"=== R4 matched-context control, {ds} final n={len(ids)} (POST-HOC) ===",
         "DPAD = each agent keeps its own 5 passages + 5 exclusive hard-negative pads (10 per agent, as in S_UNION)",
         *[f"  P-int {k}: {pp(v)}" for k, v in res["P_int"].items()],
         f"  DiD original (D − S_UNION) = {pp(res['DiD_original'])}", f"  DiD length-matched (DPAD − S_UNION) = {pp(res['DiD_padded'])}",
         f"  DPAD − D = {pp(res['DPAD_minus_D'])}",
         *[f"  {k}: MAS {v['MAS']:.3f} No-comm {v['NOCOMM']:.3f}" for k, v in res["accuracy"].items()]]
    if ds == "hotpot":
        rows = []
        for o in range(0, 6):
            G, c = (G0, c0) if o == 0 else gen_and_score(ds, f"OVL{o}")
            rows.append((o, pint(G, c)))
        x = np.repeat([o for o, _ in rows], len(ids)); y = np.concatenate([v for _, v in rows])
        idx = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
        Y = np.stack([v for _, v in rows])
        os_ = np.arange(6); xm = os_.mean()
        slope = lambda Ym: float(((os_ - xm) * (Ym.mean(1) - Ym.mean())).sum() / ((os_ - xm) ** 2).sum())
        bs = [slope(Y[:, ix]) for ix in idx]
        res["overlap"] = dict(P_int={o: ci(v) for o, v in rows}, slope_pp_per_passage=[slope(Y) * 100, *(np.percentile(bs, [2.5, 97.5]) * 100).tolist()])
        L += [f"  HotpotQA overlap o=0..5 (W4 rule) P-int: " + "; ".join(f"o={o} {v[0]*100:+.1f}" for o, v in res["overlap"]["P_int"].items()),
              f"  slope {res['overlap']['slope_pp_per_passage'][0]:+.2f} pp/passage [{res['overlap']['slope_pp_per_passage'][1]:+.2f}, {res['overlap']['slope_pp_per_passage'][2]:+.2f}]"]
    return res, L


# ---------------------------------------------------------------- R5 partition seeds
def r5(ds):
    G0, c0 = frozen(ds)
    dist = f"runs/dist{'_hotpot' if ds == 'hotpot' else ''}/S_UNION_final"
    GS, cS = rd(f"{dist}_gen.jsonl"), Score(ds).fn(f"{dist}_judged.jsonl")
    seeds = [2027] + [s for s in (2028, 2029, 2030) if os.path.exists(f"{RX}/{ds}_SEED{s}_D_gen.jsonl")]
    ids = sorted(G0); ci = boot(len(ids)); per = {}
    pint = lambda G, c: np.array([c(i, G[i], "MAS") - c(i, G[i], "NOCOMM") for i in ids])
    for s in seeds:
        if s == 2027:
            (GD, cD), (GU, cU) = (G0, c0), (GS, cS)
        else:
            (GD, cD), (GU, cU) = gen_and_score(ds, f"SEED{s}_D"), gen_and_score(ds, f"SEED{s}_SU")
        pd_, pu = pint(GD, cD), pint(GU, cU)
        a0 = np.array([cD(i, GD[i], "A0") for i in ids]); b0 = np.array([cD(i, GD[i], "B0") for i in ids])
        per[s] = dict(P_int_D=ci(pd_), P_int_SU=ci(pu), DiD=ci(pd_ - pu),
                      MAS_D=float(np.mean([cD(i, GD[i], "MAS") for i in ids])), NOCOMM_D=float(np.mean([cD(i, GD[i], "NOCOMM") for i in ids])),
                      MAS_SU=float(np.mean([cU(i, GU[i], "MAS") for i in ids])), NOCOMM_SU=float(np.mean([cU(i, GU[i], "NOCOMM") for i in ids])),
                      complementarity=float(np.mean(a0 != b0)), one_correct=float(np.mean((a0 + b0) == 1)))
        if s != 2027:
            per[s]["gold_A"] = float(np.mean([GD[i]["gold"]["A"] > 0 for i in ids])); per[s]["gold_B"] = float(np.mean([GD[i]["gold"]["B"] > 0 for i in ids]))
    vals = lambda k: np.array([per[s][k][0] for s in seeds])
    res = dict(dataset=ds, n=len(ids), seeds=seeds, per_seed=per,
               across_seeds={k: dict(mean=float(vals(k).mean()), sd=float(vals(k).std(ddof=1)) if len(seeds) > 1 else None,
                                     min=float(vals(k).min()), max=float(vals(k).max())) for k in ("P_int_D", "P_int_SU", "DiD")})
    L = [f"=== R5 partition-seed robustness, {ds} final n={len(ids)}, seeds {seeds} (2027 = frozen) (POST-HOC) ==="]
    for s in seeds:
        v = per[s]
        L.append(f"  seed {s}: P-int D {pp(v['P_int_D'])}; P-int S_UNION {pp(v['P_int_SU'])}; DiD {pp(v['DiD'])}; "
                 f"MAS/NC D {v['MAS_D']:.3f}/{v['NOCOMM_D']:.3f} SU {v['MAS_SU']:.3f}/{v['NOCOMM_SU']:.3f}; one-correct {v['one_correct']:.3f}")
    for k, v in res["across_seeds"].items():
        L.append(f"  across seeds {k}: mean {v['mean']*100:+.1f} pp, sd {(v['sd'] or 0)*100:.1f}, range [{v['min']*100:+.1f}, {v['max']*100:+.1f}]")
    return res, L


# ---------------------------------------------------------------- R6 single-agent union baselines
def r6(ds):
    G0, c0 = frozen(ds)
    GU, cU = gen_and_score(ds, "UNI")
    ids = sorted(G0); ci = boot(len(ids))
    v = {r: np.array([c0(i, G0[i], r) for i in ids]) for r in ("MAS", "NOCOMM", "POOL", "SC")}
    v |= {r: np.array([cU(i, GU[i], r) for i in ids]) for r in ("UNI_ONE", "UNI_REV", "UNI_SC")}
    tok = lambda r: r["input_tokens"] + r["output_tokens"]
    t_in = lambda r: r["input_tokens"]; t_out = lambda r: r["output_tokens"]
    cost = {}
    for name, f in (("MAS", lambda g: [g["A0"], g["B0"], g["A1m"], g["B1m"], *g["MAS"]["calls"]]),
                    ("NOCOMM", lambda g: [g["A0"], g["B0"], g["A1n"], g["B1n"], *g["NOCOMM"]["calls"]]),
                    ("POOL", lambda g: [g["POOL"]]), ("SC", lambda g: g["SC"]["samples"])):
        cs = [f(G0[i]) for i in ids]
        cost[name] = dict(calls=float(np.mean([len(c) for c in cs])), input=float(np.mean([sum(map(t_in, c)) for c in cs])),
                          output=float(np.mean([sum(map(t_out, c)) for c in cs])))
    for name, f in (("UNI_ONE", lambda g: [g["UNI_ONE"]]), ("UNI_REV", lambda g: [g["UNI_ONE"], g["UNI_REV"]]),
                    ("UNI_SC", lambda g: g["UNI_SC"]["samples"])):
        cs = [f(GU[i]) for i in ids]
        cost[name] = dict(calls=float(np.mean([len(c) for c in cs])), input=float(np.mean([sum(map(t_in, c)) for c in cs])),
                          output=float(np.mean([sum(map(t_out, c)) for c in cs])))
    res = dict(dataset=ds, n=len(ids), accuracy={k: float(x.mean()) for k, x in v.items()}, cost=cost,
               contrasts={f"MAS_minus_{k}": ci(v["MAS"] - v[k]) for k in ("UNI_ONE", "UNI_REV", "UNI_SC", "POOL", "SC")})
    L = [f"=== R6 single-agent union baselines (same evidence union A5∪B5 as the MAS team), {ds} final n={len(ids)} (POST-HOC) ===",
         *[f"  {k:8s} acc {res['accuracy'][k]:.3f}  calls {cost[k]['calls']:.1f}  in {cost[k]['input']:.0f}  out {cost[k]['output']:.0f}" for k in v],
         *[f"  {k}: {pp(x)}" for k, x in res["contrasts"].items()]]
    return res, L


# ---------------------------------------------------------------- R10 / R12 message controls
def rmsg(ds, arms, title):
    G0, c0 = frozen(ds)
    full = gen_and_score(ds, "R1_FULL")
    GA = {"R1_FULL": full, **{m: gen_and_score(ds, m) for m in arms}}
    ids = sorted(i for i in G0 if all(not GA[m][0][i].get("skipped") for m in GA))
    ci = boot(len(ids))
    fin = {m: np.array([GA[m][1](i, GA[m][0][i], "MAS") for i in ids]) for m in GA}
    fin["NC"] = np.array([c0(i, G0[i], "NOCOMM") for i in ids])
    U = {m: agent_units(ids, G0, c0, *GA[m]) for m in GA}
    cb = cboot([u["q"] for u in U["R1_FULL"]])
    words = {m: float(np.mean([u["meta"].get("sent_words", 0) for u in U[m]])) for m in GA}
    rows = {}
    for m in GA:
        corr, nw = cb([u["rev"] for u in U[m]], [u["init"] == 0 for u in U[m]])
        harm, nr = cb([1 - u["rev"] for u in U[m]], [u["init"] == 1 for u in U[m]])
        rows[m] = dict(final=float(fin[m].mean()), minus_FULL=ci(fin[m] - fin["R1_FULL"]), minus_NC=ci(fin[m] - fin["NC"]),
                       correction=corr, harm=harm, words_sent=words[m])
    res = dict(dataset=ds, n=len(ids), arms=rows, NC=float(fin["NC"].mean()))
    L = [f"=== {title}, {ds} final n={len(ids)} (POST-HOC) ===", f"  No-comm final {res['NC']:.3f}"]
    for m, r in rows.items():
        L.append(f"  {m:13s} final {r['final']:.3f}; −FULL {pp(r['minus_FULL'])}; −NC {pp(r['minus_NC'])}; "
                 f"correction {r['correction'][0]:.3f} harm {r['harm'][0]:.3f}; words sent {r['words_sent']:.0f}")
    return res, L


# ---------------------------------------------------------------- R11 aggregation
def r11(ds):
    G0, c0 = frozen(ds)
    GA, cA = gen_and_score(ds, "AGG")
    GE, cE = rd(f"runs/e1/{ds}_EV_gen.jsonl"), Score(ds).fn(f"runs/e1/{ds}_EV_judged.jsonl" if ds == "frames" else "")
    ids = sorted(G0); ci = boot(len(ids))
    src = {"MAS": (G0, c0, "A1m", "B1m", "MAS"), "NOCOMM": (G0, c0, "A1n", "B1n", "NOCOMM"), "EV": (GE, cE, "A1m", "B1m", "MAS")}
    V = {}
    for s, (G, c, ra, rb, fin) in src.items():
        agree = np.array([bool(G[i][ra]["answer"]) and normalize(G[i][ra]["answer"]) == normalize(G[i][rb]["answer"] or "") for i in ids])
        V[s] = dict(existing=np.array([c(i, G[i], fin) for i in ids]),
                    ALWAYS=np.array([cA(i, GA[i], f"ALWAYS_{s}") for i in ids]),
                    EVSYN=np.array([cA(i, GA[i], f"EVSYN_{s}") for i in ids]),
                    NOSYN_A=np.array([c(i, G[i], ra) for i in ids]), NOSYN_B=np.array([c(i, G[i], rb) for i in ids]),
                    synth_rate=float(1 - agree.mean()))
    res = dict(dataset=ds, n=len(ids), rules={}, P_int={})
    for rule in ("existing", "ALWAYS", "EVSYN", "NOSYN_A", "NOSYN_B"):
        res["rules"][rule] = {s: float(V[s][rule].mean()) for s in src}
        res["P_int"][rule] = ci(V["MAS"][rule] - V["NOCOMM"][rule])
    res["P_int"]["NOSYN_mean"] = ci(((V["MAS"]["NOSYN_A"] + V["MAS"]["NOSYN_B"]) - (V["NOCOMM"]["NOSYN_A"] + V["NOCOMM"]["NOSYN_B"])) / 2)
    res["EV_minus_FULL"] = {rule: ci(V["EV"][rule] - V["MAS"][rule]) for rule in ("existing", "ALWAYS", "EVSYN")}
    res["synth_rate"] = {s: V[s]["synth_rate"] for s in src}
    L = [f"=== R11 aggregation intervention on frozen revisions, {ds} final n={len(ids)} (POST-HOC) ===",
         "  synthesizer invocation under the existing rule: " + ", ".join(f"{s} {v:.3f}" for s, v in res["synth_rate"].items()),
         *[f"  {rule:9s} acc " + ", ".join(f"{s} {v:.3f}" for s, v in res['rules'][rule].items()) + f";  P-int (MAS−NoComm) {pp(res['P_int'][rule])}" for rule in res["rules"]],
         f"  NOSYN (mean of the two fixed-agent picks) P-int {pp(res['P_int']['NOSYN_mean'])}",
         *[f"  EV − FULL under {rule}: {pp(v)}" for rule, v in res["EV_minus_FULL"].items()]]
    return res, L


if __name__ == "__main__":
    exp, ds = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "hotpot")
    fn = dict(r1=r1, r2=r2, r3=lambda d: r3(), r4=r4, r5=r5, r6=r6, r11=r11,
              r10=lambda d: rmsg(d, [m for m in ("R10_SHORT", "R10_HARDNEG", "R10_RANDNEG") if d == "hotpot" or m != "R10_SHORT"],
                                 "R10 message length / relevance controls"),
              r12=lambda d: rmsg(d, ["R12_N3", "R12_N6", "R12_GOLDLAST"], "R12 retrieval-noise stress"))[exp]
    res, L = fn(ds)
    save(f"{exp}_{ds}", res, L)
