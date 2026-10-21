"""BX2 analyses (DESIGN_BX2.md, tag bx2-freeze). POST-HOC overall, pre-specified within BX2.
  python bx2_analyze.py a     B3 evidence-integration audit (A1 primary, A2 secondary, A3 mechanism; citation = PROXY)
  python bx2_analyze.py b     ownership-framing control on F_SPLIT
  python bx2_analyze.py c     latency / cost audit (+ modeled latency sensitivity)
  python bx2_analyze.py d     statistics and leakage audit
-> runs/bx2/RESULTS_<part>.{txt,json}"""
import ast
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

from common import ROOT, normalize
from rx_analyze import B, boot, pp, rd

OUT = f"{ROOT}/runs/bx2"
CELLS = ["NONE", "P_EXCL", "P_SHAR", "F_SPLIT", "F_CONC", "F_SHAR"]


def save(name, res, L):
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/RESULTS_{name}.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{OUT}/RESULTS_{name}.json", "w"), indent=1)
    print("\n".join(L))


def hotpot_gold():
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    gold = {r["idx"]: set(r["gold_titles"]) for r in map(json.loads, open(f"{ROOT}/data/rx/assign_hotpot.jsonl"))}
    em = lambda i, a: float(bool(a) and normalize(a) == normalize(lab[str(i)]))
    return gold, em


def cited(text):
    out = set()
    for grp in re.findall(r"\[([^\]]*)\]", text or ""):
        out |= {int(x) for x in re.findall(r"P\s*(\d+)", grp)}
    return out


def agent_view(g, me, gold):
    other = "B" if me == "A" else "A"
    own = {t for t in g["ids"][me] if t in gold}; par = {t for t in g["ids"][other] if t in gold}
    if own >= gold:
        cfg = "own+own"
    elif not own and par >= gold:
        cfg = "partner+partner"
    elif own and (own | par) >= gold:
        cfg = "own+partner"
    else:
        cfg = "incomplete"
    pos = {t: {j + 1 for j, x in enumerate(g["ids"][me]) if x == t} | {6 + j for j, x in enumerate(g["ids"][other]) if x == t} for t in gold}
    c = cited(g[f"{me}1m"]["text"])
    return cfg, all(pos[t] & c for t in gold), (own | par) >= gold


def joint_rate_diff(qids, units_a, units_b):
    """bootstrap over questions of rate(a) − rate(b); units = list of (question, success) pairs."""
    uq = sorted(set(qids)); pos = {q: k for k, q in enumerate(uq)}
    def agg(units):
        s = np.zeros(len(uq)); c = np.zeros(len(uq))
        for q, y in units:
            s[pos[q]] += y; c[pos[q]] += 1
        return s, c
    sa, ca = agg(units_a); sb, cb = agg(units_b)
    idx = np.random.default_rng(0).integers(0, len(uq), (B, len(uq)))
    ra = sa[idx].sum(1) / np.maximum(ca[idx].sum(1), 1); rb = sb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1)
    pt = sa.sum() / max(ca.sum(), 1) - sb.sum() / max(cb.sum(), 1)
    return [float(pt), *map(float, np.percentile(ra - rb, [2.5, 97.5]))], int(ca.sum()), int(cb.sum())


# ------------------------------------------------------------------------------------------------------------ A
def part_a():
    gold, em = hotpot_gold()
    G = {c: rd(f"runs/bx/hotpot_B3_{c}_gen.jsonl") for c in CELLS}
    ids = sorted(i for i in G["F_SPLIT"] if all(not G[c][i].get("skipped") for c in CELLS)); n = len(ids); ci = boot(n)
    hold = lambda c, i: G[c][i]["holder"]; oth = lambda c, i: "B" if hold(c, i) == "A" else "A"
    rev = lambda c, i, s: em(i, G[c][i][f"{s}1m"]["answer"]); r0 = lambda c, i, s: em(i, G[c][i][f"{s}0"]["answer"])
    fin = lambda c, i: em(i, G[c][i]["MAS"]["answer"])
    res = dict(n=n, A1={}, A2={}, A3={})
    # A1 PRIMARY — all agents
    agent = dict(F_SPLIT=np.array([(rev("F_SPLIT", i, "A") + rev("F_SPLIT", i, "B")) / 2 for i in ids]),
                 F_SHAR=np.array([(rev("F_SHAR", i, "A") + rev("F_SHAR", i, "B")) / 2 for i in ids]),
                 F_CONC_holder=np.array([rev("F_CONC", i, hold("F_CONC", i)) for i in ids]),
                 F_CONC_other=np.array([rev("F_CONC", i, oth("F_CONC", i)) for i in ids]))
    team = {c: np.array([fin(c, i) for i in ids]) for c in ("F_SPLIT", "F_CONC", "F_SHAR")}
    res["A1"]["agent_revised_acc"] = {k: ci(v) for k, v in agent.items()}
    res["A1"]["agent_contrasts"] = {f"F_SPLIT − {k}": ci(agent["F_SPLIT"] - agent[k]) for k in ("F_SHAR", "F_CONC_holder", "F_CONC_other")}
    res["A1"]["team_final"] = {k: ci(v) for k, v in team.items()}
    res["A1"]["team_contrasts"] = {"F_SPLIT − F_CONC": ci(team["F_SPLIT"] - team["F_CONC"]), "F_SPLIT − F_SHAR": ci(team["F_SPLIT"] - team["F_SHAR"]),
                                   "F_CONC − F_SHAR": ci(team["F_CONC"] - team["F_SHAR"])}
    # A2 SECONDARY — agents wrong at round 0
    units = {k: [] for k in agent}
    for i in ids:
        for s in "AB":
            if not r0("F_SPLIT", i, s):
                units["F_SPLIT"].append((i, rev("F_SPLIT", i, s)))
            if not r0("F_SHAR", i, s):
                units["F_SHAR"].append((i, rev("F_SHAR", i, s)))
        h, o = hold("F_CONC", i), oth("F_CONC", i)
        if not r0("F_CONC", i, h):
            units["F_CONC_holder"].append((i, rev("F_CONC", i, h)))
        if not r0("F_CONC", i, o):
            units["F_CONC_other"].append((i, rev("F_CONC", i, o)))
    for k in units:
        d, na, _ = joint_rate_diff(ids, units[k], [])
        res["A2"][k] = dict(n_agents=na, repair_rate=d)
    res["A2"]["contrasts"] = {f"F_SPLIT − {k}": joint_rate_diff(ids, units["F_SPLIT"], units[k])[0] for k in ("F_SHAR", "F_CONC_holder", "F_CONC_other")}
    # A3 MECHANISM AUDIT
    stage = {}
    for c in CELLS:
        rows = []
        for i in ids:
            g = G[c][i]; va, vb = agent_view(g, "A", gold[i]), agent_view(g, "B", gold[i])
            cand0 = r0(c, i, "A") or r0(c, i, "B"); candr = rev(c, i, "A") or rev(c, i, "B"); f = fin(c, i)
            rows.append(dict(received=float(va[2] or vb[2]), cited_agent=(va[1] + vb[1]) / 2, cited_any=float(va[1] or vb[1]),
                             cand_round0=float(cand0), cand_revision=float(candr), final=f,
                             no_candidate=float(not candr), lost_in_aggregation=float(candr and not f)))
        stage[c] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        pres = [stage[c]["final"][j] for j in range(n) if stage[c]["cand_revision"][j]]
        res["A3"][c] = {k: ci(v) for k, v in stage[c].items()}
        res["A3"][c]["preserved_given_candidate"] = float(np.mean(pres)) if pres else None
    res["A3"]["paired"] = {f"{a} − {b}": {k: ci(stage[a][k] - stage[b][k]) for k in stage[a]} for a, b in (("F_SPLIT", "F_CONC"), ("F_SPLIT", "F_SHAR"))}
    L = [f"=== BX2-A B3 evidence-integration audit, HotpotQA final n={n} (POST-HOC; pre-specified in DESIGN_BX2.md) ===",
         "A1 PRIMARY (all agents, same questions; total effect of placement)"]
    for k, v in res["A1"]["agent_revised_acc"].items():
        L.append(f"  agent revised acc  {k:15s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    L += [f"  {k:30s} {pp(v)}" for k, v in res["A1"]["agent_contrasts"].items()]
    L += [f"  team final {k:8s} {v[0]:.3f}" for k, v in res["A1"]["team_final"].items()]
    L += [f"  team {k:22s} {pp(v)}" for k, v in res["A1"]["team_contrasts"].items()]
    L.append("A2 SECONDARY (agents wrong at round 0; conditional repair rate — NOT a causal total effect)")
    L += [f"  {k:15s} n={v['n_agents']:4d}  repair {v['repair_rate'][0]:.3f} [{v['repair_rate'][1]:.3f}, {v['repair_rate'][2]:.3f}]" for k, v in res["A2"].items() if k != "contrasts"]
    L += [f"  {k:30s} {pp(v)}" for k, v in res["A2"]["contrasts"].items()]
    L.append("A3 MECHANISM AUDIT (question level; cited = citation-based PROXY of evidence use)")
    L.append(f"  {'cell':8s} {'recv':>5s} {'cite/ag':>7s} {'cite_any':>8s} {'cand0':>6s} {'cand_rev':>8s} {'final':>6s} {'no_cand':>7s} {'lost_agg':>8s} {'pres|cand':>9s}")
    for c in CELLS:
        v = res["A3"][c]
        L.append(f"  {c:8s} {v['received'][0]:5.3f} {v['cited_agent'][0]:7.3f} {v['cited_any'][0]:8.3f} {v['cand_round0'][0]:6.3f} "
                 f"{v['cand_revision'][0]:8.3f} {v['final'][0]:6.3f} {v['no_candidate'][0]:7.3f} {v['lost_in_aggregation'][0]:8.3f} {v['preserved_given_candidate']:9.3f}")
    for k, d in res["A3"]["paired"].items():
        L.append(f"  paired {k}: " + "; ".join(f"{s} {pp(x)}" for s, x in d.items() if s != "received"))
    return res, L


# ------------------------------------------------------------------------------------------------------------ B
def part_b():
    _, em = hotpot_gold()
    C = rd("runs/bx/hotpot_B3_F_SPLIT_gen.jsonl"); T = rd(f"{OUT}/hotpot_F_SPLIT_LOCAL_gen.jsonl")
    assert all(r.get("error") is None for r in T.values())
    ids = sorted(set(C) & set(T)); n = len(ids); ci = boot(n)
    assert all(T[i]["A0"]["text"] == C[i]["A0"]["text"] and T[i]["B0"]["text"] == C[i]["B0"]["text"] for i in ids), "round-0 differs"
    ag = lambda G, i: (em(i, G[i]["A1m"]["answer"]) + em(i, G[i]["B1m"]["answer"])) / 2
    ac, at = np.array([ag(C, i) for i in ids]), np.array([ag(T, i) for i in ids])
    fc, ft = np.array([em(i, C[i]["MAS"]["answer"]) for i in ids]), np.array([em(i, T[i]["MAS"]["answer"]) for i in ids])
    cand = lambda G, i: em(i, G[i]["A1m"]["answer"]) or em(i, G[i]["B1m"]["answer"])
    pc = joint_rate_diff(ids, [(i, em(i, T[i]["MAS"]["answer"])) for i in ids if cand(T, i)], [(i, em(i, C[i]["MAS"]["answer"])) for i in ids if cand(C, i)])
    same = np.mean([T[i]["A1m"]["answer"] == C[i]["A1m"]["answer"] for i in ids] + [T[i]["B1m"]["answer"] == C[i]["B1m"]["answer"] for i in ids])
    res = dict(n=n, prompt_diff=["-Their evidence passages:", "+More of your own evidence passages:"],
               PRIMARY_agent_revised=dict(control=ci(ac), treatment=ci(at), diff=ci(at - ac)),
               SECONDARY_team_final=dict(control=ci(fc), treatment=ci(ft), diff=ci(ft - fc)),
               SECONDARY_preservation=dict(treatment_minus_control=pc[0], n_t=pc[1], n_c=pc[2]),
               identical_revised_answer_share=float(same))
    L = [f"=== BX2-B ownership-framing control, HotpotQA B3_F_SPLIT, n={n} (POST-HOC; pre-specified) ===",
         "only prompt difference: 'Their evidence passages:' -> 'More of your own evidence passages:'",
         f"PRIMARY agent revised acc: control {ac.mean():.3f}, treatment {at.mean():.3f}; treatment − control {pp(res['PRIMARY_agent_revised']['diff'])}",
         f"SECONDARY team final: control {fc.mean():.3f}, treatment {ft.mean():.3f}; diff {pp(res['SECONDARY_team_final']['diff'])}",
         f"SECONDARY preservation (final | revision candidate): treatment − control {pp(pc[0])} (n {pc[1]} / {pc[2]})",
         f"identical revised answers between conditions: {same:.3f}"]
    return res, L


# ------------------------------------------------------------------------------------------------------------ C
CRIT = dict(LOCAL_ONE=[["A0"]], PEER_EV_5=[["A0", "B0"], ["A1m", "B1m"], ["synth"]], FULL=[["A0", "B0"], ["A1m", "B1m"], ["synth"]],
            CQ_Q_5=[["A0"], ["A1m"]], CQ_GEN_5=[["A0"], ["qgen"], ["A1m"]], UNI_ONE=[["UNI_ONE"]], POOL=[["POOL"]])
RCRIT = dict(LOCAL_ONE=[["retrieve_A"]], PEER_EV_5=[["retrieve_A", "retrieve_B"]], FULL=[["retrieve_A", "retrieve_B"]],
             CQ_Q_5=[["retrieve_A"], ["retrieve_remote"]], CQ_GEN_5=[["retrieve_A"], ["retrieve_remote"]],
             UNI_ONE=[["retrieve_A", "retrieve_B"]], POOL=[["retrieve_central"]])


def crit(recs, plan):
    w = {}
    for r in recs:
        w[r["stage"]] = w.get(r["stage"], 0) + r["wall"]
    return sum(max([w.get(s, 0) for s in grp] or [0]) for grp in plan)


def part_c():
    rows = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/lat_*.jsonl")) for l in open(f)]
    systems = list(CRIT); res = dict(n_runs=len(rows), conditions={})
    calls = [c for r in rows for c in r["calls"]]
    fz = [c for c in calls if c["frozen_exists"]]
    res["cache_hits"] = int(sum(c["hit"] for c in calls)); res["n_calls"] = len(calls)
    res["reproduction"] = dict(n_compared=len(fz), same_text=float(np.mean([c["same_text"] for c in fz])) if fz else None,
                               same_answer=float(np.mean([c["same_answer"] for c in fz])) if fz else None)
    L = [f"=== BX2-C latency / cost audit, FRAMES 100 pre-drawn questions (measured; local BM25 is NOT Web-retrieval latency) ===",
         f"calls {len(calls)}, cache hits {res['cache_hits']}; reproduction vs frozen cache: same text {res['reproduction']['same_text']}, "
         f"same parsed answer {res['reproduction']['same_answer']} (n {len(fz)})"]
    for cond in ("isolated", "loaded"):
        R = [r for r in rows if r["cond"] == cond]
        if not R:
            continue
        qids = sorted({r["idx"] for r in R}); ci = boot(len(qids)); out = {}
        L.append(f"[{cond}] reps {sorted({r['rep'] for r in R})}, questions {len(qids)}, max in-flight {max(r['inflight_at_start'] for r in R)}")
        L.append(f"  {'system':10s} {'e2e mean [95% CI]':>24s} {'median':>7s} {'LLM crit':>8s} {'retr crit ms':>12s} {'calls':>6s} {'tokens':>7s}")
        for S in systems:
            per = {i: [r for r in R if r["idx"] == i and r["system"] == S] for i in qids}
            e2e = np.array([np.mean([r["e2e"] for r in per[i]]) for i in qids])
            llm = np.array([np.mean([crit(r["calls"], CRIT[S]) for r in per[i]]) for i in qids])
            ret = np.array([np.mean([crit(r["retr"], RCRIT[S]) for r in per[i]]) for i in qids])
            ncall = np.mean([len(r["calls"]) for i in qids for r in per[i]])
            tok = np.mean([sum(c["input_tokens"] + c["output_tokens"] for c in r["calls"]) for i in qids for r in per[i]])
            out[S] = dict(e2e=ci(e2e), e2e_median=float(np.median(e2e)), llm_crit=float(llm.mean()), retr_crit=float(ret.mean()),
                          calls=float(ncall), tokens=float(tok), serial_retrieval_rounds=per[qids[0]][0]["serial_retrieval_rounds"], _llm=llm)
            L.append(f"  {S:10s} {out[S]['e2e'][0]:7.2f} s [{out[S]['e2e'][1]:.2f}, {out[S]['e2e'][2]:.2f}] {out[S]['e2e_median']:7.2f} "
                     f"{out[S]['llm_crit']:8.2f} {1000 * out[S]['retr_crit']:12.1f} {ncall:6.2f} {tok:7.0f}")
        if cond == "isolated":
            L.append("  MODELED latency sensitivity (not measured): mean(LLM critical path) + r × serial retrieval rounds")
            for r_ in (0.05, 0.2, 0.5, 1, 2, 5):
                m = {S: float(out[S]["_llm"].mean() + r_ * out[S]["serial_retrieval_rounds"]) for S in systems}
                L.append(f"    r={r_:>4}s: " + ", ".join(f"{S} {v:.1f}" for S, v in sorted(m.items(), key=lambda kv: kv[1])))
                out.setdefault("_modeled", {})[str(r_)] = m
        res["conditions"][cond] = {k: ({kk: vv for kk, vv in v.items() if kk != "_llm"} if isinstance(v, dict) and "_llm" in v else v) for k, v in out.items()}
    return res, L


# ------------------------------------------------------------------------------------------------------------ D
def part_d():
    import pandas as pd
    res = dict(denominators={}, tost={}, recompute={}, altscore={}, leakage={})
    L = ["=== BX2-D statistics and leakage audit (POST-HOC; pre-specified) ===", "denominators (gen files: rows / skipped / errors):"]
    for f in sorted(glob.glob(f"{ROOT}/runs/rx/*_gen.jsonl") + glob.glob(f"{ROOT}/runs/rx32/*_gen.jsonl") + glob.glob(f"{ROOT}/runs/bx/*_gen.jsonl")
                    + glob.glob(f"{OUT}/*_gen.jsonl")):
        R = [json.loads(l) for l in open(f)]
        d = dict(rows=len(R), skipped=sum(bool(r.get("skipped")) for r in R), errors=sum(r.get("error") is not None for r in R))
        res["denominators"][os.path.relpath(f, ROOT)] = d
    bad = {k: v for k, v in res["denominators"].items() if v["errors"]}
    L.append(f"  {len(res['denominators'])} generation files; files with errors: {bad or 'none'}; with skips: "
             + ", ".join(f"{os.path.basename(k)} {v['skipped']}" for k, v in res["denominators"].items() if v["skipped"]))
    # TOST + independent recomputation of B4-P1 / P2 (per-question means, own bootstrap code)
    from rx_analyze import Score
    for ds in ("frames", "hotpot"):
        def acc(m):
            g = rd(f"runs/bx/{ds}_{m}_gen.jsonl"); c = Score(ds).fn(f"runs/bx/{ds}_{m}_judged.jsonl")
            return {i: (c(i, g[i], "A1m") + c(i, g[i], "B1m")) / 2 for i in g if not g[i].get("skipped")}
        A = {m: acc(m) for m in ("PEER_EV_1", "PEER_EV_3", "PEER_EV_5", "CQ_Q_5", "CQ_GEN_1", "CQ_GEN_3", "CQ_GEN_5")}
        ids = sorted(set.intersection(*[set(v) for v in A.values()]))
        rng = np.random.default_rng(12345); idx = rng.integers(0, len(ids), (10000, len(ids)))
        def ci2(x, lo, hi):
            x = np.asarray(x); bs = x[idx].mean(1)
            return [float(x.mean()), float(np.percentile(bs, lo)), float(np.percentile(bs, hi))]
        d1 = [A["CQ_Q_5"][i] - A["PEER_EV_5"][i] for i in ids]
        c95, c90 = ci2(d1, 2.5, 97.5), ci2(d1, 5, 95)
        res["tost"][ds] = dict(diff=c95[0], ci95=c95[1:], ci90=c90[1:], margin=0.02, margin_prespecified="DESIGN_BX.md @ bx-freeze 97fcc5c",
                               equivalent_95=bool(c95[1] > -0.02 and c95[2] < 0.02), equivalent_TOST90=bool(c90[1] > -0.02 and c90[2] < 0.02))
        orig = json.load(open(f"{ROOT}/runs/bx/RESULTS_b14_{ds}.json"))
        res["recompute"][ds] = dict(P1_orig=orig["B4_P1_framing_CQ_Q5_minus_PEER_EV5_agent"]["diff"][0], P1_new=c95[0],
                                    P2_orig={k: v[0] for k, v in orig["B4_P2_targeting_CQ_GEN_minus_PEER_EV_agent"].items()},
                                    P2_new={k: float(np.mean([A[f"CQ_GEN_{k}"][i] - A[f"PEER_EV_{k}"][i] for i in ids])) for k in (1, 3, 5)})
        t = res["tost"][ds]
        L.append(f"B4-P1 {ds}: diff {100 * t['diff']:+.2f} pp; 95% CI [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}] (pre-specified reading) "
                 f"equivalent {t['equivalent_95']}; TOST 90% CI [{100 * t['ci90'][0]:+.2f}, {100 * t['ci90'][1]:+.2f}] equivalent {t['equivalent_TOST90']} (margin ±2 pp, pre-specified)")
        r = res["recompute"][ds]
        L.append(f"  independent recomputation {ds}: P1 {r['P1_orig']:+.4f} vs {r['P1_new']:+.4f}; P2 " +
                 ", ".join(f"k={k} {r['P2_orig'][str(k)]:+.4f} vs {r['P2_new'][k]:+.4f}" for k in (1, 3, 5)))
    # alternative scoring (FRAMES), run in a subprocess so RX_SCORE is read at import time; nothing is overwritten
    for sc in ("contain", "f1"):
        code = ("import json,bx_analyze as b; r,_=b.b14('frames'); "
                "print(json.dumps(dict(P1=r['B4_P1_framing_CQ_Q5_minus_PEER_EV5_agent']['diff'], P2=r['B4_P2_targeting_CQ_GEN_minus_PEER_EV_agent'],"
                "ladder={k:v['accuracy'][0] for k,v in r['systems'].items()})))")
        env = dict(os.environ, RX_SCORE=sc)
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=f"{ROOT}/scripts")
        assert o.returncode == 0, o.stderr[-2000:]
        res["altscore"][sc] = json.loads(o.stdout.strip().split("\n")[-1])
        a = res["altscore"][sc]
        L.append(f"alt scoring FRAMES [{sc}]: P1 {pp(a['P1'])}; P2 " + ", ".join(f"k={k} {pp(v)}" for k, v in a["P2"].items())
                 + f"; CQ_GEN_5 {a['ladder']['CQ_GEN_5']:.3f} vs FULL(MAS) {a['ladder']['FULL(MAS)']:.3f} vs UNI_SC {a['ladder']['UNI_SC']:.3f}")
    # leakage (i) static: no gold-like name inside any prompt(...) call
    # gold-bearing identifiers only (case-sensitive): gold fields, label dicts, the FRAMES gold column, RX gold sets.
    # (first version was case-insensitive and flagged the prompt name 'answer' / fields answer_1 — false positives)
    sus = re.compile(r"gold|label|\blab\b|['\"]Answer['\"]|sup_sents|G_inj"); hits = []
    for f in ("run_webqa.py", "rx_run.py", "bx_run.py", "bx2_run.py", "bx2_latency.py"):
        tree = ast.parse(open(f"{ROOT}/scripts/{f}").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "prompt":
                src = ast.unparse(node)
                if sus.search(src):
                    hits.append(f"{f}: {src[:160]}")
    res["leakage"]["static_hits"] = hits
    L.append(f"leakage (static): prompt(...) calls referencing gold-like names: {hits or 'none'}"
             " (rx_run R1/R2/R10 build gold-informed MESSAGES from data/rx assignments by design; those calls pass passages, not gold labels)")
    # leakage (ii) dynamic: generated queries vs round-0 answers containing the gold answer
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t"); lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    gold = dict(frames=lambda i: str(df.loc[i, "Answer"]), hotpot=lambda i: lab[str(i)])
    for ds in ("frames", "hotpot"):
        g = rd(f"runs/bx/{ds}_CQ_GEN_5_gen.jsonl")
        has = lambda i, t: (lambda a: len(a) >= 2 and f" {a} " in f" {normalize(t or '')} ")(normalize(gold[ds](i)))
        qy = np.mean([has(i, g[i]["meta"][s]["query"]) for i in g for s in "AB"])
        a0 = np.mean([has(i, g[i][f"{s}0"]["text"]) for i in g for s in "AB"])
        res["leakage"][ds] = dict(query_contains_gold=float(qy), round0_text_contains_gold=float(a0))
        L.append(f"leakage (dynamic) {ds}: generated query contains gold answer {qy:.3f}; round-0 reasoning contains it {a0:.3f}")
    L.append("bootstrap type: question-level paired (rx_analyze.boot) for question outcomes; agent outcomes averaged per question "
             "or cluster-resampled (rx_analyze.cboot / bx2 joint_rate_diff) — the two agents of a question always move together")
    return res, L


if __name__ == "__main__":
    part = sys.argv[1]
    res, L = dict(a=part_a, b=part_b, c=part_c, d=part_d)[part]()
    save(part, res, L)
