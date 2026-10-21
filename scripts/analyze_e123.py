"""Sealed analyses of E1, E2 and E3 (DESIGN_DRAFT.md), each run once after generation (+ judging for FRAMES).
Correctness: FRAMES = frozen judge verdicts (Qwen3-32B-FP8); HotpotQA = EM. Paired bootstrap over questions,
10,000 resamples, seed 0, one shared index matrix per experiment.
E1 (per dataset)  conditions NC (frozen No-comm), ANS, ANS_RSN, ANS_EV, ANS_IRR, EV, FULL (frozen MAS); outcome = final
   accuracy. PRIMARY: C−B = ANS_EV − ANS and C−D = ANS_EV − ANS_IRR. Secondary: each mode − NC; ANS_RSN − ANS;
   FULL − ANS_EV; EV − ANS_EV; on round-0 discordant pairs (identical across modes): BRR, HRR, and the share of the
   wrong agent's revisions that equal the partner's round-0 answer. Reading: C−B CI > 0 -> "genuine evidence adds beyond
   the partner's answer"; C−D CI > 0 -> "the gain is sensitive to evidence relevance"; both -> "evidence-transfer
   sensitive"; otherwise "not established" for that contrast.
E2  cells EXCL_LOW, EXCL_HIGH, SHAR_LOW, SHAR_HIGH on the eligible questions; outcome P-int = MAS − No-comm.
   PRIMARY: placement effect = mean_overlap[P-int(EXCL) − P-int(SHAR)]; overlap effect = mean_placement[P-int(LOW) −
   P-int(HIGH)]. Secondary: interaction; per-cell P-int, MAS, No-comm, round-0 agreement. Reading: placement CI > 0 ->
   "partner-exclusive gold evidence drives the communication gain"; overlap CI > 0 -> "overlap reduces the gain with the
   team's gold content held fixed"; otherwise "not established".
E3  topic partition (PEOPLE vs WORLD). PRIMARY: P-int; P-util = MAS − Pooled (frozen Pooled; same pooled index).
   Secondary: MAS − SC-4 (frozen SC); DiD vs the random partition = P-int(topic) − P-int(random D); share of questions
   whose gold pages span both domains; held / retrieved availability on discordant pairs.
  python analyze_e123.py e1 frames|hotpot | e2 | e3   ->  runs/<e1|e2|e3>/RESULTS_*.{txt,json}"""
import json
import sys

import numpy as np

from common import ROOT, normalize

B = 10000
rd = lambda p: {x["idx"]: x for x in map(json.loads, open(f"{ROOT}/{p}"))}


def scorer(ds, judged):
    if ds == "frames":
        J = rd(judged)
        return lambda i, g, r: float(J[i][r]["verdict"])
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    return lambda i, g, r: float(normalize(g[r]["answer"] or "") == normalize(lab[str(i)]))


def boot(n):
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    return lambda x: [float(x.mean()), *map(float, np.percentile(x[idx].mean(1), [2.5, 97.5]))]


pp = lambda v: f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"


def e1(ds):
    base = "runs/final/frames_final" if ds == "frames" else "runs/final/hotpot_final"
    G0 = rd(f"{base}_gen.jsonl"); c0 = scorer(ds, f"{base}_judged.jsonl")
    ids = sorted(G0); ci = boot(len(ids))
    modes = ("ANS", "ANS_RSN", "ANS_EV", "ANS_IRR", "EV")
    G = {m: rd(f"runs/e1/{ds}_{m}_gen.jsonl") for m in modes}
    C = {m: scorer(ds, f"runs/e1/{ds}_{m}_judged.jsonl") for m in modes}
    assert all(sorted(G[m]) == ids and all(x["error"] is None for x in G[m].values()) for m in modes)
    fin = {"NC": np.array([c0(i, G0[i], "NOCOMM") for i in ids]), "FULL": np.array([c0(i, G0[i], "MAS") for i in ids])}
    rev = {"NC": ("A1n", "B1n", G0, c0), "FULL": ("A1m", "B1m", G0, c0)}
    for m in modes:
        fin[m] = np.array([C[m](i, G[m][i], "MAS") for i in ids]); rev[m] = ("A1m", "B1m", G[m], C[m])
    a0 = np.array([c0(i, G0[i], "A0") for i in ids]); b0 = np.array([c0(i, G0[i], "B0") for i in ids])
    disc = [(k, i, "A" if a0[k] == 0 else "B") for k, i in enumerate(ids) if a0[k] != b0[k]]
    res = dict(dataset=ds, n=len(ids), accuracy={m: float(v.mean()) for m, v in fin.items()},
               primary=dict(C_minus_B=ci(fin["ANS_EV"] - fin["ANS"]), C_minus_D=ci(fin["ANS_EV"] - fin["ANS_IRR"])),
               secondary={f"{m}_minus_NC": ci(fin[m] - fin["NC"]) for m in ("ANS", "ANS_RSN", "ANS_EV", "ANS_IRR", "EV", "FULL")}
               | dict(ANS_RSN_minus_ANS=ci(fin["ANS_RSN"] - fin["ANS"]), FULL_minus_ANS_EV=ci(fin["FULL"] - fin["ANS_EV"]),
                      EV_minus_ANS_EV=ci(fin["EV"] - fin["ANS_EV"])), revision={})
    for m, (ra, rb, g, c) in rev.items():
        r = {"A": ra, "B": rb}
        brr = [c(i, g[i], r[w]) for _, i, w in disc]
        hrr = [1 - c(i, g[i], r["B" if w == "A" else "A"]) for _, i, w in disc]
        adopt = [bool(g[i][r[w]]["answer"]) and normalize(g[i][r[w]]["answer"]) == normalize(G0[i][("B" if w == "A" else "A") + "0"]["answer"] or "")
                 for _, i, w in disc]
        res["revision"][m] = dict(n=len(disc), BRR=float(np.mean(brr)), HRR=float(np.mean(hrr)), adopt_partner=float(np.mean(adopt)))
    cb, cd = (res["primary"][k][1] > 0 for k in ("C_minus_B", "C_minus_D"))
    res["reading"] = dict(C_minus_B="genuine evidence adds beyond the partner's answer" if cb else "not established",
                          C_minus_D="the gain is sensitive to evidence relevance" if cd else "not established",
                          overall="evidence-transfer sensitive" if cb and cd else "not established")
    L = [f"=== E1 evidence-transfer intervention, {ds} final n={len(ids)} ({'judge' if ds == 'frames' else 'EM'}) ===",
         f"PRIMARY  C−B (answer+evidence − answer only)       = {pp(res['primary']['C_minus_B'])} -> {res['reading']['C_minus_B']}",
         f"PRIMARY  C−D (answer+evidence − answer+irrelevant) = {pp(res['primary']['C_minus_D'])} -> {res['reading']['C_minus_D']}",
         "final accuracy: " + ", ".join(f"{m} {v:.3f}" for m, v in res["accuracy"].items()),
         *[f"  {k}: {pp(v)}" for k, v in res["secondary"].items()],
         f"round-0 discordant pairs n={len(disc)}:",
         *[f"  {m:8s} BRR {v['BRR']:.3f} HRR {v['HRR']:.3f} adopt-partner {v['adopt_partner']:.3f}" for m, v in res["revision"].items()]]
    return res, L, f"runs/e1/RESULTS_e1_{ds}"


def e2():
    cells = ("EXCL_LOW", "EXCL_HIGH", "SHAR_LOW", "SHAR_HIGH")
    G = {c: rd(f"runs/e2/{c}_gen.jsonl") for c in cells}; C = {c: scorer("frames", f"runs/e2/{c}_judged.jsonl") for c in cells}
    ids = sorted(G[cells[0]]); ci = boot(len(ids))
    assert all(sorted(G[c]) == ids and all(x["error"] is None for x in G[c].values()) for c in cells)
    V = {c: {r: np.array([C[c](i, G[c][i], r) for i in ids]) for r in ("MAS", "NOCOMM", "A0", "B0")} for c in cells}
    P = {c: V[c]["MAS"] - V[c]["NOCOMM"] for c in cells}
    place = ((P["EXCL_LOW"] - P["SHAR_LOW"]) + (P["EXCL_HIGH"] - P["SHAR_HIGH"])) / 2
    over = ((P["EXCL_LOW"] - P["EXCL_HIGH"]) + (P["SHAR_LOW"] - P["SHAR_HIGH"])) / 2
    inter = (P["EXCL_LOW"] - P["EXCL_HIGH"]) - (P["SHAR_LOW"] - P["SHAR_HIGH"])
    res = dict(n=len(ids), primary=dict(placement=ci(place), overlap=ci(over)), secondary=dict(interaction=ci(inter)), cells={})
    for c in cells:
        agree = np.mean([bool(G[c][i]["A0"]["answer"]) and normalize(G[c][i]["A0"]["answer"]) == normalize(G[c][i]["B0"]["answer"] or "") for i in ids])
        res["cells"][c] = dict(P_int=ci(P[c]), MAS=float(V[c]["MAS"].mean()), NOCOMM=float(V[c]["NOCOMM"].mean()),
                               round0_acc=float(((V[c]["A0"] + V[c]["B0"]) / 2).mean()), round0_agree=float(agree),
                               shared=float(np.mean([len(set(G[c][i]["pids"]["A"]) & set(G[c][i]["pids"]["B"])) for i in ids])))
    res["reading"] = dict(placement="partner-exclusive gold evidence drives the communication gain" if res["primary"]["placement"][1] > 0 else "not established",
                          overlap="overlap reduces the gain with the team's gold content held fixed" if res["primary"]["overlap"][1] > 0 else "not established")
    L = [f"=== E2 overlap × gold placement, FRAMES final eligible n={len(ids)} (judge) ===",
         f"PRIMARY placement effect (EXCL − SHAR) = {pp(res['primary']['placement'])} -> {res['reading']['placement']}",
         f"PRIMARY overlap effect (LOW − HIGH)    = {pp(res['primary']['overlap'])} -> {res['reading']['overlap']}",
         f"interaction = {pp(res['secondary']['interaction'])}",
         *[f"  {c:9s} P-int {pp(v['P_int'])}; MAS {v['MAS']:.3f} No-comm {v['NOCOMM']:.3f}; round-0 acc {v['round0_acc']:.3f} agree {v['round0_agree']:.3f}; shared {v['shared']:.1f}"
           for c, v in res["cells"].items()]]
    return res, L, "runs/e2/RESULTS_e2"


def e3():
    G = rd("runs/e3/TOPIC_gen.jsonl"); c = scorer("frames", "runs/e3/TOPIC_judged.jsonl")
    G0 = rd("runs/final/frames_final_gen.jsonl"); c0 = scorer("frames", "runs/final/frames_final_judged.jsonl")
    R = rd("data/retrieval_topic.jsonl")
    ids = sorted(G); ci = boot(len(ids))
    assert sorted(G0) == ids and all(x["error"] is None for x in G.values())
    v = {r: np.array([c(i, G[i], r) for i in ids]) for r in ("MAS", "NOCOMM", "A0", "B0", "A1m", "B1m")}
    pool = np.array([c0(i, G0[i], "POOL") for i in ids]); sc = np.array([c0(i, G0[i], "SC") for i in ids])
    pint_d = np.array([c0(i, G0[i], "MAS") - c0(i, G0[i], "NOCOMM") for i in ids])
    tok = lambda r: r["input_tokens"] + r["output_tokens"]
    span = np.mean([len(set(R[i]["gold_owner"].values())) == 2 for i in ids])
    disc = [(k, i, "A" if v["A0"][k] == 0 else "B") for k, i in enumerate(ids) if v["A0"][k] != v["B0"][k]]
    held = np.mean([any(o == ("B" if w == "A" else "A") for o in R[i]["gold_owner"].values()) for _, i, w in disc])
    ret = np.mean([bool(R[i][("B" if w == "A" else "A") + "_gold_retrieved"]) for _, i, w in disc])
    res = dict(n=len(ids), primary=dict(P_int=ci(v["MAS"] - v["NOCOMM"]), P_util=ci(v["MAS"] - pool)),
               secondary=dict(MAS_minus_SC=ci(v["MAS"] - sc), DiD_topic_minus_random=ci((v["MAS"] - v["NOCOMM"]) - pint_d)),
               accuracy=dict(MAS=float(v["MAS"].mean()), NOCOMM=float(v["NOCOMM"].mean()), A0=float(v["A0"].mean()), B0=float(v["B0"].mean()),
                             POOL=float(pool.mean()), SC=float(sc.mean())),
               gold_spans_both_domains=float(span), discordant=len(disc), held=float(held), retrieved=float(ret),
               tokens_MAS=float(np.mean([tok(G[i]["A0"]) + tok(G[i]["B0"]) + tok(G[i]["A1m"]) + tok(G[i]["B1m"]) + sum(tok(x) for x in G[i]["MAS"]["calls"]) for i in ids])))
    L = [f"=== E3 topic-partitioned retrieval (PEOPLE vs WORLD), FRAMES final n={len(ids)} (judge) ===",
         f"PRIMARY P-int MAS − No-comm = {pp(res['primary']['P_int'])}",
         f"PRIMARY P-util MAS − Pooled = {pp(res['primary']['P_util'])}",
         f"MAS − SC-4 = {pp(res['secondary']['MAS_minus_SC'])}; DiD (topic − random partition) = {pp(res['secondary']['DiD_topic_minus_random'])}",
         "accuracy: " + ", ".join(f"{k} {x:.3f}" for k, x in res["accuracy"].items()) + f"; MAS tokens/q {res['tokens_MAS']:.0f}",
         f"gold pages span both domains: {span:.3f}; discordant {len(disc)}: partner held {held:.3f}, partner retrieved {ret:.3f}"]
    return res, L, "runs/e3/RESULTS_e3"


if __name__ == "__main__":
    res, L, out = e1(sys.argv[2]) if sys.argv[1] == "e1" else (e2() if sys.argv[1] == "e2" else e3())
    open(f"{ROOT}/{out}.txt", "w").write("\n".join(L) + "\n"); json.dump(res, open(f"{ROOT}/{out}.json", "w"), indent=1)
    print("\n".join(L))
