"""RX offline experiments and analyses — no new inference; frozen generations and judgments only. POST-HOC.
  python rx_offline.py <x7|x8|a1|a2|a3|a4|a5|a6> <frames|hotpot>   ->  runs/rx/RESULTS_<name>_<ds>.{txt,json}
x7  conditional communication: per question, communicate (MAS) or not (No-comm) from signals observable after round 0.
    Policies NeverComm, AlwaysComm, Disagreement, LowConfidence (hedging in either round-0 rationale), ScoreGap
    (metadata only: the partner's best BM25 score vs one's own — a number an agent can share without its passages),
    Disagreement-or-ScoreGap, Oracle (uses the outcome: an upper bound, not a policy). Thresholds are chosen on the PILOT
    cohort only (max pilot accuracy, ties -> lower activation) and applied unchanged to the final cohort.
    Cost = the calls each question actually runs (input / output tokens, calls).
x8  adaptive message content on the E1 cached arms; rules fixed here with no tuning:
    Adaptive-ANS/EV (agree -> answer-only, else evidence-only), Adaptive-NC/EV (agree -> no message, else EV),
    Adaptive-NC/FULL (agree -> no message, else full message).
a1  evidence availability x initial correctness (agent level); a2 E2 eligibility / selection (FRAMES);
a3  correctness transitions W->C, C->W, ... by condition; a4 evidence-transfer funnel; a5 RQ1 DiD decomposition;
a6  accuracy-cost Pareto over every system with full-pipeline cost (R6 baselines, X7 policies; X8 on its own
    revision-stage axis). Reads the RESULTS files only; descriptive (no CI on the frontier itself)."""
import json
import re
import sys
from collections import Counter

import numpy as np

from common import ROOT, normalize
from rx_analyze import Score, boot, cboot, pp, rd, save

HEDGE = re.compile(r"(not (explicitly )?(mention|state|specif|provide|include|contain)|does not (say|give|provide|mention)|"
                   r"insufficient|cannot (be )?determin|not enough information|no information|unclear|unknown|best guess|"
                   r"not possible to|unable to|lack of|missing)", re.I)
tok_re = lambda s: re.findall(r"\w+", s.lower())


def load_run(ds, split):
    if split == "final":
        g, c = rd(f"runs/final/{ds}_final_gen.jsonl"), Score(ds).fn(f"runs/final/{ds}_final_judged.jsonl")
        return g, c
    g = rd(f"runs/{ds}_pilot_gen.jsonl")
    if ds == "frames":
        return g, Score(ds).fn("runs/frames_pilot_judged.jsonl")
    lab = json.load(open(f"{ROOT}/data/hotpot_pilot_labels.json"))
    return g, lambda i, gg, r: float(bool(gg[r].get("answer")) and normalize(gg[r]["answer"]) == normalize(lab[str(i)]))


def calls(g, arm):
    if arm == "MAS":
        return [g["A0"], g["B0"], g["A1m"], g["B1m"], *g["MAS"]["calls"]]
    return [g["A0"], g["B0"], g["A1n"], g["B1n"], *g["NOCOMM"]["calls"]]


def score_gap(ds, split):
    """metadata signal: partner's best BM25 score minus own best (both agents' views), per question."""
    from rx_build import BM25Okapi
    from run_webqa import load
    Q = {q["idx"]: q for q in load(ds, split)}
    out = {}
    for i, q in Q.items():
        qt = tok_re(q["question"]); best = {}
        for s in "AB":
            docs = [tok_re(p["title"] + " " + p["text"]) for p in q[s]]
            best[s] = float(np.max(BM25Okapi(docs).get_scores(qt))) if docs and any(docs) else 0.0
        out[i] = abs(best["A"] - best["B"]) / (abs(best["A"]) + abs(best["B"]) + 1e-9)
    return out


def content_novelty(ds, split):
    """content-based signal: how much of the partner's evidence is NEW to the receiver. Per receiver: mean over partner
    passages of (1 - max TF-IDF cosine to any own passage); question-level = max over the two receivers. Computing it
    requires the partner's passages, so its cost is charged even when the policy then decides not to communicate."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from run_webqa import load
    out = {}
    for q in load(ds, split):
        docs = [p["title"] + " " + p["text"] for p in q["A"] + q["B"]]
        if not q["A"] or not q["B"]:
            out[q["idx"]] = 0.0; continue
        X = TfidfVectorizer().fit_transform(docs); a, b = X[:len(q["A"])], X[len(q["A"]):]
        sim = (a @ b.T).toarray()                       # own A x partner B
        nov_A = float(np.mean(1 - sim.max(0))); nov_B = float(np.mean(1 - sim.max(1)))
        out[q["idx"]] = max(nov_A, nov_B)
    return out


def x7(ds):
    sig = {}
    for split in ("pilot", "final"):
        g, c = load_run(ds, split)
        ids = sorted(i for i in g if g[i].get("error") is None)
        gap = score_gap(ds, split); nov = content_novelty(ds, split)
        sig[split] = dict(ids=ids, g=g,
                          mas=np.array([c(i, g[i], "MAS") for i in ids]), nc=np.array([c(i, g[i], "NOCOMM") for i in ids]),
                          disagree=np.array([not (g[i]["A0"]["answer"] and g[i]["B0"]["answer"] and
                                                  normalize(g[i]["A0"]["answer"]) == normalize(g[i]["B0"]["answer"])) for i in ids]),
                          hedge=np.array([len(HEDGE.findall(g[i]["A0"]["text"] or "")) + len(HEDGE.findall(g[i]["B0"]["text"] or "")) for i in ids]),
                          gap=np.array([gap[i] for i in ids]), nov=np.array([nov[i] for i in ids]),
                          # tokens of the partner message (what a content-based check must receive): MAS - No-comm revision input
                          partner_tok=np.array([max(0, g[i]["A1m"]["input_tokens"] - g[i]["A1n"]["input_tokens"]) +
                                                max(0, g[i]["B1m"]["input_tokens"] - g[i]["B1n"]["input_tokens"]) for i in ids]))

    def evaluate(S, comm, content=False):
        acc = np.where(comm, S["mas"], S["nc"])
        cs = [calls(S["g"][i], "MAS" if m else "NC") for i, m in zip(S["ids"], comm)]
        surcharge = np.where(comm, 0, S["partner_tok"]) if content else np.zeros(len(comm))
        return acc, dict(activation=float(np.mean(comm)), accuracy=float(acc.mean()),
                         calls=float(np.mean([len(c) for c in cs])),
                         input=float(np.mean([sum(x["input_tokens"] for x in c) for c in cs]) + surcharge.mean()),
                         output=float(np.mean([sum(x["output_tokens"] for x in c) for c in cs])))

    def tune(name, fn, grid):
        P = sig["pilot"]; best = None
        for t in grid:
            acc, m = evaluate(P, fn(P, t), name.startswith("Content"))
            key = (m["accuracy"], -m["activation"])
            if best is None or key > best[0]:
                best = (key, t)
        return best[1]

    pol = {"NeverComm": (lambda S, t: np.zeros(len(S["ids"]), bool), [None]),
           "AlwaysComm": (lambda S, t: np.ones(len(S["ids"]), bool), [None]),
           "Disagreement": (lambda S, t: S["disagree"], [None]),
           "LowConfidence": (lambda S, t: S["hedge"] >= t, list(range(0, 7))),
           "ScoreGap": (lambda S, t: S["gap"] >= t, [float(x) for x in np.round(np.linspace(0, 0.6, 13), 2)]),
           "Disagree_or_ScoreGap": (lambda S, t: S["disagree"] | (S["gap"] >= t), [float(x) for x in np.round(np.linspace(0, 0.6, 13), 2)]),
           "ContentNovelty": (lambda S, t: S["nov"] >= t, [float(x) for x in np.round(np.linspace(0, 1, 21), 2)]),
           "ContentNovelty_or_Disagree": (lambda S, t: S["disagree"] | (S["nov"] >= t), [float(x) for x in np.round(np.linspace(0, 1, 21), 2)])}
    F = sig["final"]; ci = boot(len(F["ids"]))
    always, _ = evaluate(F, np.ones(len(F["ids"]), bool))
    res = dict(dataset=ds, n_final=len(F["ids"]), n_pilot=len(sig["pilot"]["ids"]), policies={})
    for name, (fn, grid) in pol.items():
        t = tune(name, fn, grid)
        acc, m = evaluate(F, fn(F, t), name.startswith("Content"))
        res["policies"][name] = dict(threshold=t, **m, minus_AlwaysComm=ci(acc - always),
                                     cost_note=("content check charged: partner passages received on every question" if name.startswith("Content") else None))
    oracle = np.maximum(F["mas"], F["nc"]); comm_o = (F["mas"] > F["nc"])
    _, mo = evaluate(F, comm_o)
    res["policies"]["Oracle (upper bound)"] = dict(threshold=None, **{**mo, "accuracy": float(oracle.mean())})
    L = [f"=== X7 conditional communication, {ds}: thresholds chosen on pilot n={res['n_pilot']}, read on final n={res['n_final']} (POST-HOC) ===",
         f"{'policy':24s} {'thr':>6s} {'act':>6s} {'acc':>6s} {'calls':>6s} {'in_tok':>7s} {'out_tok':>7s}  acc − AlwaysComm"]
    for k, v in res["policies"].items():
        L.append(f"{k:24s} {str(v['threshold']):>6s} {v['activation']:6.3f} {v['accuracy']:6.3f} {v['calls']:6.2f} {v['input']:7.0f} {v['output']:7.0f}  "
                 + (pp(v["minus_AlwaysComm"]) if "minus_AlwaysComm" in v else "(uses outcomes)"))
    return res, L


def x8(ds):
    G0, c0 = load_run(ds, "final")
    arms = {"ANS": "ANS", "ANS_RSN": "ANS_RSN", "ANS_EV": "ANS_EV", "EV": "EV"}
    GA = {m: (rd(f"runs/e1/{ds}_{m}_gen.jsonl"), Score(ds).fn(f"runs/e1/{ds}_{m}_judged.jsonl")) for m in arms}
    ids = sorted(G0); ci = boot(len(ids))
    agree = np.array([bool(G0[i]["A0"]["answer"] and G0[i]["B0"]["answer"]) and
                      normalize(G0[i]["A0"]["answer"]) == normalize(G0[i]["B0"]["answer"]) for i in ids])
    acc = {"NC": np.array([c0(i, G0[i], "NOCOMM") for i in ids]), "FULL": np.array([c0(i, G0[i], "MAS") for i in ids])}
    for m, (g, c) in GA.items():
        acc[m] = np.array([c(i, g[i], "MAS") for i in ids])
    rev_calls = lambda g: [g["A1m"], g["B1m"], *g["MAS"]["calls"]]
    cost = {"NC": [sum(x["input_tokens"] + x["output_tokens"] for x in [G0[i]["A1n"], G0[i]["B1n"], *G0[i]["NOCOMM"]["calls"]]) for i in ids],
            "FULL": [sum(x["input_tokens"] + x["output_tokens"] for x in rev_calls(G0[i])) for i in ids]}
    for m, (g, _) in GA.items():
        cost[m] = [sum(x["input_tokens"] + x["output_tokens"] for x in rev_calls(g[i])) for i in ids]
    cost = {k: np.array(v, float) for k, v in cost.items()}
    rules = {"Always-ANS": ("ANS", "ANS"), "Always-EV": ("EV", "EV"), "Always-FULL": ("FULL", "FULL"), "No-comm": ("NC", "NC"),
             "Adaptive-ANS/EV": ("ANS", "EV"), "Adaptive-NC/EV": ("NC", "EV"), "Adaptive-NC/FULL": ("NC", "FULL")}
    res = dict(dataset=ds, n=len(ids), agree_rate=float(agree.mean()), rules={})
    for name, (on_agree, on_dis) in rules.items():
        a = np.where(agree, acc[on_agree], acc[on_dis]); t = np.where(agree, cost[on_agree], cost[on_dis])
        res["rules"][name] = dict(accuracy=float(a.mean()), revision_tokens=float(t.mean()),
                                  minus_AlwaysEV=ci(a - acc["EV"]), minus_AlwaysFULL=ci(a - acc["FULL"]))
    L = [f"=== X8 adaptive message content on the E1 cached arms, {ds} final n={len(ids)}; round-0 agreement {agree.mean():.3f} (rules fixed, no tuning; POST-HOC) ==="]
    for k, v in res["rules"].items():
        L.append(f"  {k:18s} acc {v['accuracy']:.3f}  revision-stage tokens {v['revision_tokens']:7.0f}  − Always-EV {pp(v['minus_AlwaysEV'])}  − Always-FULL {pp(v['minus_AlwaysFULL'])}")
    return res, L


def agent_gold(ds):
    """per question and agent: number of gold pages / paragraphs held (D layout) and the question's gold total."""
    if ds == "frames":
        R = rd("data/retrieval_frames.jsonl")
        return {i: dict(A=len(r["A_gold_retrieved"]), B=len(r["B_gold_retrieved"]), total=len(r["gold_pages"]),
                        owner=r["gold_owner"]) for i, r in R.items()}
    H = rd("data/hotpot_final.jsonl")
    return {i: dict(A=len(r["A_gold"]), B=len(r["B_gold"]), total=len(r["gold_titles"])) for i, r in H.items()}


def a1(ds):
    G0, c0 = load_run(ds, "final"); gold = agent_gold(ds)
    ids = sorted(G0)
    st = lambda k, t: "none" if k == 0 else ("complete" if k >= t else "partial")
    rows = Counter(); out = {}
    units = []
    for i in ids:
        a0, b0 = c0(i, G0[i], "A0"), c0(i, G0[i], "B0")
        for me, oth in (("A", "B"), ("B", "A")):
            units.append(dict(q=i, own=st(gold[i][me], gold[i]["total"]), partner=st(gold[i][oth], gold[i]["total"]),
                              init_me=c0(i, G0[i], f"{me}0"), init_partner=c0(i, G0[i], f"{oth}0"),
                              mas=c0(i, G0[i], f"{me}1m"), nc=c0(i, G0[i], f"{me}1n")))
    cb = cboot([u["q"] for u in units])
    L = [f"=== A1 evidence availability x initial correctness (agent level), {ds} final n={len(ids)} questions (POST-HOC) ===",
         "availability = gold " + ("pages retrieved (page-level)" if ds == "frames" else "paragraphs held (complete = both)"),
         f"{'own':9s} {'partner':9s} {'me0':>4s} {'pt0':>4s} {'n':>5s}  P(correct): MAS-rev  NoComm-rev   MAS−NC"]
    for own in ("none", "partial", "complete"):
        for par in ("none", "partial", "complete"):
            for im in (0, 1):
                for ip in (0, 1):
                    m = [u["own"] == own and u["partner"] == par and u["init_me"] == im and u["init_partner"] == ip for u in units]
                    if sum(m) < 10:
                        continue
                    (vm, n), (vn, _) = cb([u["mas"] for u in units], m), cb([u["nc"] for u in units], m)
                    vd, _ = cb([u["mas"] - u["nc"] for u in units], m)
                    key = f"{own}|{par}|{im}|{ip}"; out[key] = dict(n=n, MAS_rev=vm, NC_rev=vn, diff=vd)
                    L.append(f"{own:9s} {par:9s} {im:4d} {ip:4d} {n:5d}  {vm[0]:7.3f}  {vn[0]:9.3f}   {pp(vd)}")
    return dict(dataset=ds, cells=out), L


def a2(ds="frames"):
    G0, c0 = load_run("frames", "final"); E = rd("data/e2_assign.jsonl"); R = rd("data/retrieval_frames.jsonl")
    P = {p["pid"]: p for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    ids = sorted(G0); el = {i: E[i]["eligible"] for i in ids}
    feat = lambda i: dict(NC=c0(i, G0[i], "NOCOMM"), MAS=c0(i, G0[i], "MAS"), P_int=c0(i, G0[i], "MAS") - c0(i, G0[i], "NOCOMM"),
                          round0_acc=(c0(i, G0[i], "A0") + c0(i, G0[i], "B0")) / 2,
                          agree=float(bool(G0[i]["A0"]["answer"] and G0[i]["B0"]["answer"]) and normalize(G0[i]["A0"]["answer"]) == normalize(G0[i]["B0"]["answer"])),
                          n_gold_pages=len(R[i]["gold_pages"]), team_gold_retrieved=float(bool(R[i]["A_gold_retrieved"] or R[i]["B_gold_retrieved"])),
                          n_gold_in_pool=E[i]["n_gold_pool"], evidence_words=sum(len(P[j]["text"].split()) for j in R[i]["A"] + R[i]["B"]),
                          question_words=len(str(G0[i]["A0"].get("text", "")).split()) * 0)
    F = {i: feat(i) for i in ids}; keys = [k for k in F[ids[0]] if k != "question_words"]
    res = dict(n=len(ids), eligible=sum(el.values()), excluded=len(ids) - sum(el.values()),
               reasons=Counter("no gold passage in pool" if not E[i]["G"] else "fewer than 13 fillers" for i in ids if not el[i]), compare={})
    rng = np.random.default_rng(0)
    for k in keys:
        a = np.array([F[i][k] for i in ids if el[i]]); b = np.array([F[i][k] for i in ids if not el[i]])
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
        res["compare"][k] = dict(eligible=float(a.mean()), excluded=float(b.mean()), diff=[float(a.mean() - b.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))])
    L = [f"=== A2 E2 eligibility and selection, FRAMES final: eligible {res['eligible']} / excluded {res['excluded']} (POST-HOC) ===",
         f"exclusion reasons: {dict(res['reasons'])}",
         *[f"  {k:20s} eligible {v['eligible']:8.3f}  excluded {v['excluded']:8.3f}  diff {v['diff'][0]:+.3f} [{v['diff'][1]:+.3f}, {v['diff'][2]:+.3f}]" for k, v in res["compare"].items()]]
    res["reasons"] = dict(res["reasons"])
    return res, L


def a3(ds):
    G0, c0 = load_run(ds, "final")
    conds = {"FULL(MAS)": (G0, c0, "m"), "No-comm": (G0, c0, "n")}
    for m in ("ANS", "EV"):
        conds[m] = (rd(f"runs/e1/{ds}_{m}_gen.jsonl"), Score(ds).fn(f"runs/e1/{ds}_{m}_judged.jsonl"), "m")
    dist = f"runs/dist{'_hotpot' if ds == 'hotpot' else ''}/S_UNION_final"
    conds["S_UNION MAS"] = (rd(f"{dist}_gen.jsonl"), Score(ds).fn(f"{dist}_judged.jsonl"), "m")
    conds["S_UNION No-comm"] = (conds["S_UNION MAS"][0], conds["S_UNION MAS"][1], "n")
    ids = sorted(G0); res = {}; L = [f"=== A3 agent-level correctness transitions (round 0 -> revision), {ds} final n={len(ids)} (POST-HOC) ==="]
    for name, (G, c, r) in conds.items():
        t = Counter()
        for i in ids:
            for me in "AB":
                t[("C" if c(i, G[i], f"{me}0") else "W") + "->" + ("C" if c(i, G[i], f"{me}1{r}") else "W")] += 1
        n = sum(t.values()); res[name] = dict(t)
        L.append(f"  {name:16s} " + "  ".join(f"{k} {t[k]:4d} ({t[k]/n:.3f})" for k in ("W->C", "C->W", "W->W", "C->C"))
                 + f"   net {(t['W->C'] - t['C->W'])/n*100:+.1f} pp")
    return dict(dataset=ds, transitions=res), L


def a4(ds):
    G0, c0 = load_run(ds, "final"); gold = agent_gold(ds); ids = sorted(G0)
    stages = Counter()
    for i in ids:
        for me, oth in (("A", "B"), ("B", "A")):
            if c0(i, G0[i], f"{me}0"):
                continue
            stages["0 receiver initially wrong"] += 1
            if ds == "frames":
                avail = any(o == oth for o in gold[i]["owner"].values())
            else:
                avail = gold[i][oth] > 0
            if not avail:
                continue
            stages["1 available (partner side holds gold)"] += 1
            if gold[i][oth] == 0:
                continue
            stages["2 retrieved by partner"] += 1
            stages["3 transmitted (partner passages are always sent in full)"] += 1
            if not c0(i, G0[i], f"{me}1m"):
                continue
            stages["4 corrected (receiver right after revision)"] += 1
            if c0(i, G0[i], "MAS"):
                stages["5 preserved (team final answer right)"] += 1
    keys = sorted(stages); L = [f"=== A4 evidence-transfer funnel, {ds} final, receivers initially wrong (POST-HOC; descriptive, not causal) ==="]
    prev = None
    for k in keys:
        L.append(f"  {k:58s} {stages[k]:5d}" + (f"   ({stages[k]/stages[prev]:.3f} of previous)" if prev else ""))
        prev = k
    return dict(dataset=ds, funnel=dict(stages)), L


def a5(ds):
    G0, c0 = load_run(ds, "final")
    dist = f"runs/dist{'_hotpot' if ds == 'hotpot' else ''}/S_UNION_final"
    GS, cS = rd(f"{dist}_gen.jsonl"), Score(ds).fn(f"{dist}_judged.jsonl")
    ids = sorted(G0); ci = boot(len(ids)); gold = agent_gold(ds)
    v = lambda G, c, r: np.array([c(i, G[i], r) for i in ids])
    D = {r: v(G0, c0, r) for r in ("MAS", "NOCOMM", "A0", "B0")}; S = {r: v(GS, cS, r) for r in ("MAS", "NOCOMM", "A0", "B0")}
    agree = lambda G: np.mean([bool(G[i]["MAS"]["answer"]) for i in ids])
    fin_agree = lambda G, a, b: float(np.mean([bool(G[i][a]["answer"] and G[i][b]["answer"]) and normalize(G[i][a]["answer"]) == normalize(G[i][b]["answer"]) for i in ids]))
    ptok = lambda G: float(np.mean([G[i]["A0"]["input_tokens"] + G[i]["B0"]["input_tokens"] for i in ids]) / 2)
    cov_d = float(np.mean([(gold[i]["A"] > 0) + (gold[i]["B"] > 0) for i in ids]) / 2)
    cov_s = float(np.mean([(gold[i]["A"] + gold[i]["B"]) > 0 for i in ids]))
    res = dict(dataset=ds, n=len(ids), DiD=ci((D["MAS"] - D["NOCOMM"]) - (S["MAS"] - S["NOCOMM"])),
               delta_shared_minus_dist=dict(MAS=ci(S["MAS"] - D["MAS"]), NOCOMM=ci(S["NOCOMM"] - D["NOCOMM"]),
                                            round0=ci((S["A0"] + S["B0"]) / 2 - (D["A0"] + D["B0"]) / 2)),
               complementarity=dict(D=float(np.mean(D["A0"] != D["B0"])), S_UNION=float(np.mean(S["A0"] != S["B0"]))),
               round0_answer_agreement=dict(D=fin_agree(G0, "A0", "B0"), S_UNION=fin_agree(GS, "A0", "B0")),
               revised_agreement_MAS=dict(D=fin_agree(G0, "A1m", "B1m"), S_UNION=fin_agree(GS, "A1m", "B1m")),
               per_agent_gold_coverage=dict(D=cov_d, S_UNION=cov_s),
               round0_prompt_tokens=dict(D=ptok(G0), S_UNION=ptok(GS)))
    L = [f"=== A5 decomposition of the RQ1 DiD, {ds} final n={len(ids)} (POST-HOC) ===",
         f"DiD = {pp(res['DiD'])}",
         *[f"  Shared-union − Distributed, {k}: {pp(x)}" for k, x in res["delta_shared_minus_dist"].items()],
         f"  one-correct complementarity: D {res['complementarity']['D']:.3f}  S_UNION {res['complementarity']['S_UNION']:.3f}",
         f"  round-0 answer agreement: D {res['round0_answer_agreement']['D']:.3f}  S_UNION {res['round0_answer_agreement']['S_UNION']:.3f}",
         f"  MAS revised agreement: D {res['revised_agreement_MAS']['D']:.3f}  S_UNION {res['revised_agreement_MAS']['S_UNION']:.3f}",
         f"  per-agent gold coverage: D {cov_d:.3f}  S_UNION {cov_s:.3f}",
         f"  round-0 prompt tokens per agent: D {res['round0_prompt_tokens']['D']:.0f}  S_UNION {res['round0_prompt_tokens']['S_UNION']:.0f}"]
    return res, L


def a6(ds):
    import os
    R = f"{ROOT}/runs/rx"; pts = {}
    if os.path.exists(f"{R}/RESULTS_r6_{ds}.json"):
        r6 = json.load(open(f"{R}/RESULTS_r6_{ds}.json"))
        for k, a in r6["accuracy"].items():
            pts[f"R6:{k}"] = dict(accuracy=a, **r6["cost"][k])
    x7r = json.load(open(f"{R}/RESULTS_x7_{ds}.json"))
    for k, v in x7r["policies"].items():
        if k.startswith("Oracle") or (k in ("NeverComm", "AlwaysComm") and "R6:MAS" in pts):
            continue
        pts[f"X7:{k}"] = dict(accuracy=v["accuracy"], calls=v["calls"], input=v["input"], output=v["output"])
    for v in pts.values():
        v["tokens"] = v["input"] + v["output"]

    def frontier(P, cost):
        return {k for k, v in P.items() if not any(w["accuracy"] >= v["accuracy"] and w[cost] <= v[cost] and
                                                   (w["accuracy"] > v["accuracy"] or w[cost] < v[cost]) for w in P.values())}
    fr_tok, fr_calls = frontier(pts, "tokens"), frontier(pts, "calls")
    res = dict(dataset=ds, points=pts, pareto_tokens=sorted(fr_tok), pareto_calls=sorted(fr_calls))
    L = [f"=== A6 accuracy-cost Pareto, {ds} final (POST-HOC; descriptive; * = on the frontier) ===",
         f"{'system':34s} {'acc':>6s} {'calls':>6s} {'tokens':>7s}  tok  calls"]
    for k, v in sorted(pts.items(), key=lambda kv: kv[1]["tokens"]):
        L.append(f"{k:34s} {v['accuracy']:6.3f} {v['calls']:6.2f} {v['tokens']:7.0f}  {'*' if k in fr_tok else ' ':>3s}  {'*' if k in fr_calls else ' ':>4s}")
    if os.path.exists(f"{R}/RESULTS_x8_{ds}.json"):
        x8r = json.load(open(f"{R}/RESULTS_x8_{ds}.json"))["rules"]
        P8 = {k: dict(accuracy=v["accuracy"], tokens=v["revision_tokens"]) for k, v in x8r.items()}
        f8 = frontier(P8, "tokens"); res["x8_revision_stage"] = dict(points=P8, pareto=sorted(f8))
        L.append("X8 message rules on the revision-stage token axis (round 0 and synthesis excluded):")
        for k, v in sorted(P8.items(), key=lambda kv: kv[1]["tokens"]):
            L.append(f"  {k:32s} {v['accuracy']:6.3f} {v['tokens']:7.0f}  {'*' if k in f8 else ''}")
    if "R6:MAS" not in pts:
        L.append("(R6 baselines not yet available for this dataset: X7 points only)")
    return res, L


if __name__ == "__main__":
    exp, ds = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "frames")
    res, L = dict(x7=x7, x8=x8, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6)[exp](ds)
    save(f"{exp}_{ds}", res, L)
