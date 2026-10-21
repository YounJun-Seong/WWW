"""BX4 analyses (DESIGN_BX4.md, tag bx4-freeze). POST-HOC overall; pre-specified within BX4.
  python bx4_analyze.py a   sufficiency audit (model proxies; human labels if the workbooks are filled)
  python bx4_analyze.py b   matched-question placement transitions (F_SPLIT vs F_CONC)
  python bx4_analyze.py c   evidence-aware aggregation on fixed revised candidates
-> runs/bx4/RESULTS_<part>.{txt,json}"""
import json
import math
import os
import sys

import numpy as np

from common import ROOT, normalize
from rx_analyze import Score, boot, cboot, pp, rd

OUT = f"{ROOT}/runs/bx4"


def save(name, res, L):
    open(f"{OUT}/RESULTS_{name}.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{OUT}/RESULTS_{name}.json", "w"), indent=1); print("\n".join(L))


def human_labels():
    """{(ds, idx): {annotator: label}} from the filled workbooks; empty if not yet annotated."""
    path = f"{OUT}/human"; out = {}
    if not os.path.exists(f"{path}/KEY.json"):
        return out
    import openpyxl
    key = json.load(open(f"{path}/KEY.json"))
    for who in ("A", "B"):
        ws = openpyxl.load_workbook(f"{path}/annotator_{who}.xlsx")["Items"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and row[5]:
                k = key[row[0]]; out.setdefault((k["dataset"], k["idx"]), {})[who] = str(row[5]).strip().upper()
    return out


# ------------------------------------------------------------------------------------------------------------ A
def part_a():
    res, L = {}, ["=== BX4-A supporting-evidence sufficiency audit (R2 I2; receivers wrong at round 0) — MODEL PROXIES ==="]
    H = human_labels()
    for ds in ("frames", "hotpot"):
        g0, c0 = rd(f"runs/final/{ds}_final_gen.jsonl"), Score(ds).fn(f"runs/final/{ds}_final_judged.jsonl")
        g2, c2 = rd(f"runs/rx/{ds}_R2_I2_gen.jsonl"), Score(ds).fn(f"runs/rx/{ds}_R2_I2_judged.jsonl")
        gr, cr = rd(f"{OUT}/{ds}_READER_gen.jsonl"), Score(ds).fn(f"{OUT}/{ds}_READER_judged.jsonl")
        sf = rd(f"{OUT}/suff_{ds}.jsonl")
        ids = sorted(i for i in sf if i in g2 and not g2[i].get("skipped") and i in gr)
        units = [(i, s) for i in ids for s in "AB" if not c0(i, g0[i], f"{s}0")]
        fail = np.array([1 - c2(i, g2[i], f"{s}1m") for i, s in units], float)
        verd = [sf[i]["verdict"] for i, _ in units]; rdr = np.array([cr(i, gr[i], "READER") for i, _ in units], bool)
        cb = cboot([i for i, _ in units])
        masks = {"all": np.ones(len(units), bool), "SUFFICIENT": np.array([v == "SUFFICIENT" for v in verd]),
                 "INSUFFICIENT or AMBIGUOUS": np.array([v in ("INSUFFICIENT", "AMBIGUOUS") for v in verd]),
                 "reader correct": rdr, "reader wrong": ~rdr,
                 "doubly sufficient (SUFFICIENT and reader correct)": np.array([v == "SUFFICIENT" for v in verd]) & rdr}
        r = dict(n_questions=len(ids), n_receivers_initially_wrong=len(units),
                 verdict_distribution_questions={v: sum(sf[i]["verdict"] == v for i in ids) for v in ("SUFFICIENT", "INSUFFICIENT", "AMBIGUOUS", "UNPARSED")},
                 reader_accuracy_questions=float(np.mean([cr(i, gr[i], "READER") for i in ids])), failure={})
        for k, m in masks.items():
            v, n = cb(fail, m); r["failure"][k] = dict(n=n, rate=v)
        tot_fail = fail.sum(); r["share_of_failures_on_SUFFICIENT"] = float(fail[masks["SUFFICIENT"]].sum() / tot_fail) if tot_fail else None
        hl = {i: H[(ds, i)] for i in ids if (ds, i) in H}
        r["human_annotated_questions"] = len(hl)
        res[ds] = r
        L.append(f"[{ds}] questions {len(ids)}, initially-wrong receivers {len(units)}; SUFF-JUDGE verdicts {r['verdict_distribution_questions']}; "
                 f"reader-only accuracy {r['reader_accuracy_questions']:.3f}")
        for k, v in r["failure"].items():
            L.append(f"   still wrong after I2 | {k:50s} n={v['n']:4d}  {v['rate'][0]:.3f} [{v['rate'][1]:.3f}, {v['rate'][2]:.3f}]")
        L.append(f"   share of all I2 failures that occur on SUFFICIENT items: {r['share_of_failures_on_SUFFICIENT']:.3f}")
        L.append(f"   human labels available: {len(hl)} questions" + (" (analysis pending until both workbooks are filled)" if len(hl) == 0 else ""))
    L.append("SUFF-JUDGE = frozen judge model with reference answer + injected passages only; READER = agent model with injected "
             "passages only. Both are model proxies, not human verification.")
    return res, L


# ------------------------------------------------------------------------------------------------------------ B
def part_b():
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    em = lambda i, a: bool(a) and normalize(a) == normalize(lab[str(i)])
    S, C = rd("runs/bx/hotpot_B3_F_SPLIT_gen.jsonl"), rd("runs/bx/hotpot_B3_F_CONC_gen.jsonl")
    ids = sorted(set(S) & set(C)); n = len(ids)
    f = lambda G, i: em(i, G[i]["MAS"]["answer"]); cand = lambda G, i: em(i, G[i]["A1m"]["answer"]) or em(i, G[i]["B1m"]["answer"])
    r0 = lambda G, i: em(i, G[i]["A0"]["answer"]) or em(i, G[i]["B0"]["answer"])
    tab = {(a, b): sum(f(S, i) == a and f(C, i) == b for i in ids) for a in (True, False) for b in (True, False)}
    b_, c_ = tab[(True, False)], tab[(False, True)]
    p = min(1.0, 2 * sum(math.comb(b_ + c_, k) for k in range(0, min(b_, c_) + 1)) / 2 ** (b_ + c_)) if b_ + c_ else 1.0
    stage = lambda G, i: "lost in synthesis" if cand(G, i) else "no correct revised candidate"
    disc_c = [stage(S, i) for i in ids if f(C, i) and not f(S, i)]; disc_s = [stage(C, i) for i in ids if f(S, i) and not f(C, i)]
    strata = {}
    for name, sel in (("both had a round-0 candidate", lambda i: r0(S, i) and r0(C, i)), ("split only", lambda i: r0(S, i) and not r0(C, i)),
                      ("concentrated only", lambda i: r0(C, i) and not r0(S, i)), ("neither", lambda i: not r0(S, i) and not r0(C, i))):
        q = [i for i in ids if sel(i)]
        d = np.array([f(S, i) - f(C, i) for i in q], float)
        strata[name] = dict(n=len(q), split_final=float(np.mean([f(S, i) for i in q])) if q else None,
                            conc_final=float(np.mean([f(C, i) for i in q])) if q else None, diff=boot(len(q))(d) if len(q) > 1 else None)
    res = dict(n=n, table={f"split={a},conc={b}": v for (a, b), v in tab.items()}, mcnemar_exact_p=p,
               conc_right_split_wrong_stage={k: disc_c.count(k) for k in set(disc_c)},
               split_right_conc_wrong_stage={k: disc_s.count(k) for k in set(disc_s)}, round0_strata=strata)
    L = [f"=== BX4-B matched-question transitions, HotpotQA F_SPLIT vs F_CONC, n={n} (offline; POST-HOC) ===",
         f"final correct: both {tab[(True, True)]}, split only {b_}, concentrated only {c_}, neither {tab[(False, False)]}; exact McNemar p = {p:.2g}",
         f"concentrated right / split wrong ({c_}): split failed at " + ", ".join(f"{k} {v}" for k, v in res["conc_right_split_wrong_stage"].items()),
         f"split right / concentrated wrong ({b_}): concentrated failed at " + ", ".join(f"{k} {v}" for k, v in res["split_right_conc_wrong_stage"].items()),
         "stratified by round-0 correct-candidate presence (split cell / concentrated cell):"]
    for k, v in strata.items():
        L.append(f"   {k:30s} n={v['n']:4d}  final split {v['split_final']:.3f}  concentrated {v['conc_final']:.3f}  split − conc "
                 + (pp(v["diff"]) if v["diff"] else "-"))
    L.append("Limitation: allocation changes round-0 states; strata describe, they do not remove, initial-state differences.")
    return res, L


# ------------------------------------------------------------------------------------------------------------ C
def part_c():
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    em = lambda i, a: float(bool(a) and normalize(a) == normalize(lab[str(i)]))
    cells = ("F_SPLIT", "F_CONC", "F_SHAR"); G, E = {}, {}
    for c in cells:
        G[c] = rd(f"runs/bx/hotpot_B3_{c}_gen.jsonl"); E[c] = rd(f"{OUT}/hotpot_EVAGG_{c}_gen.jsonl")
        assert all(r.get("error") is None for r in E[c].values())
    ids = sorted(set.intersection(*[set(E[c]) for c in cells])); ci = boot(len(ids))
    acc = {c: dict(EXISTING=np.array([em(i, G[c][i]["MAS"]["answer"]) for i in ids]),
                   EVIDENCE_AWARE=np.array([em(i, E[c][i]["EVAGG"]["answer"]) for i in ids]),
                   ORACLE=np.array([max(em(i, G[c][i]["A1m"]["answer"]), em(i, G[c][i]["B1m"]["answer"])) for i in ids])) for c in cells}
    lost = {c: {k: int(sum(o == 1 and a == 0 for o, a in zip(acc[c]["ORACLE"], v))) for k, v in acc[c].items() if k != "ORACLE"} for c in cells}
    res = dict(n=len(ids), accuracy={c: {k: ci(v) for k, v in d.items()} for c, d in acc.items()}, lost_in_synthesis=lost,
               gap_split_minus_conc={k: ci(acc["F_SPLIT"][k] - acc["F_CONC"][k]) for k in ("EXISTING", "EVIDENCE_AWARE", "ORACLE")},
               evidence_aware_minus_existing={c: ci(acc[c]["EVIDENCE_AWARE"] - acc[c]["EXISTING"]) for c in cells})
    L = [f"=== BX4-C evidence-aware aggregation on the SAME revised candidates, HotpotQA B3, n={len(ids)} (POST-HOC) ===",
         f"{'cell':8s} {'existing':>9s} {'evidence-aware':>15s} {'oracle (bound)':>15s}   lost in synthesis (existing / evidence-aware)"]
    for c in cells:
        a = res["accuracy"][c]
        L.append(f"{c:8s} {a['EXISTING'][0]:9.3f} {a['EVIDENCE_AWARE'][0]:15.3f} {a['ORACLE'][0]:15.3f}   {lost[c]['EXISTING']} / {lost[c]['EVIDENCE_AWARE']}")
    L += [f"F_SPLIT − F_CONC gap: " + "; ".join(f"{k} {pp(v)}" for k, v in res["gap_split_minus_conc"].items()),
          "evidence-aware − existing: " + "; ".join(f"{c} {pp(v)}" for c, v in res["evidence_aware_minus_existing"].items()),
          "ORACLE = correct iff any revised answer is correct: a diagnostic upper bound for aggregation, not a system."]
    return res, L


if __name__ == "__main__":
    part = sys.argv[1]; res, L = dict(a=part_a, b=part_b, c=part_c)[part](); save(part, res, L)
