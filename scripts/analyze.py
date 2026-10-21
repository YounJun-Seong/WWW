"""Sealed analysis (DESIGN_DRAFT.md §4). Run once per (dataset, split) after generation (+ judging for FRAMES).

  python analyze.py frames pilot runs/frames_pilot_gen.jsonl runs/frames_pilot_judged.jsonl
  python analyze.py hotpot pilot runs/hotpot_pilot_gen.jsonl
-> runs/RESULTS_<dataset>_<split>.{txt,json}
"""
import json
import os
import sys

import numpy as np

from common import ROOT, f1, normalize

B, SEED = 10000, 0
ROLES = ("A0", "B0", "POOL", "A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM", "SC", "PLACEBO_A")


def boot(d, rng):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def boot_stat(fn, n, rng):
    idx = rng.integers(0, n, (B // 5, n))
    vals = np.array([fn(i) for i in idx])
    return [float(fn(np.arange(n))), *map(float, np.percentile(vals, [2.5, 97.5]))]


def tok(r):
    return r["input_tokens"] + r["output_tokens"]


def main():
    ds, split, gen = sys.argv[1], sys.argv[2], sys.argv[3]
    rng = np.random.default_rng(SEED)
    G = [json.loads(l) for l in open(gen)]
    assert all(g["error"] is None for g in G), "generation errors present"
    G = sorted(G, key=lambda g: g["idx"])
    ids = [g["idx"] for g in G]; Q = len(G)
    C, F1 = {}, {}
    if ds == "frames":
        J = {j["idx"]: j for j in map(json.loads, open(sys.argv[4]))}
        for r in ROLES:
            C[r] = np.array([float(J[i][r]["verdict"]) for i in ids])
        R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
        own = {i: dict(A_held=any(o == "A" for o in R[i]["gold_owner"].values()), B_held=any(o == "B" for o in R[i]["gold_owner"].values()),
                       A_ret=bool(R[i]["A_gold_retrieved"]), B_ret=bool(R[i]["B_gold_retrieved"]),
                       n_gold=len(R[i]["gold_pages"]), n_ret=len(set(R[i]["A_gold_retrieved"]) | set(R[i]["B_gold_retrieved"])))
               for i in ids}
    else:
        lab = json.load(open(f"{ROOT}/data/hotpot_{split}_labels.json"))
        H = {h["idx"]: h for h in map(json.loads, open(f"{ROOT}/data/hotpot_{split}.jsonl"))}
        for r in ROLES:
            C[r] = np.array([float(normalize(g[r].get("answer") or "") == normalize(lab[str(g["idx"])])) for g in G])
            F1[r] = np.array([f1(g[r].get("answer") or "", lab[str(g["idx"])]) for g in G])
        own = {i: dict(A_held=bool(H[i]["A_gold"]), B_held=bool(H[i]["B_gold"]), A_ret=bool(H[i]["A_gold"]), B_ret=bool(H[i]["B_gold"]),
                       n_gold=len(H[i]["gold_titles"]), n_ret=len(set(H[i]["A_gold"]) | set(H[i]["B_gold"]))) for i in ids}
    T = dict(A0=[tok(g["A0"]) for g in G], B0=[tok(g["B0"]) for g in G], POOL=[tok(g["POOL"]) for g in G],
             MAS=[tok(g["A0"]) + tok(g["B0"]) + tok(g["A1m"]) + tok(g["B1m"]) + sum(tok(c) for c in g["MAS"]["calls"]) for g in G],
             NOCOMM=[tok(g["A0"]) + tok(g["B0"]) + tok(g["A1n"]) + tok(g["B1n"]) + sum(tok(c) for c in g["NOCOMM"]["calls"]) for g in G],
             SC=[sum(tok(s) for s in g["SC"]["samples"]) for g in G])
    T = {k: float(np.mean(v)) for k, v in T.items()}
    res = dict(dataset=ds, split=split, n=Q, accuracy={r: float(C[r].mean()) for r in ROLES}, tokens=T,
               synthesized=dict(MAS=float(np.mean([g["MAS"]["synthesized"] for g in G])),
                                NOCOMM=float(np.mean([g["NOCOMM"]["synthesized"] for g in G]))))
    if F1:
        res["f1"] = {r: float(F1[r].mean()) for r in ROLES}
    # primary
    res["P_int"] = boot(C["MAS"] - C["NOCOMM"], rng)
    res["P_util"] = boot(C["MAS"] - C["POOL"], rng)
    # secondary G1
    res["G1"] = dict(MAS_minus_SC=boot(C["MAS"] - C["SC"], rng), token_ratio_MAS_SC=T["MAS"] / T["SC"],
                     token_ratio_MAS_NOCOMM=T["MAS"] / T["NOCOMM"],
                     MAS_minus_A=boot(C["MAS"] - C["A0"], rng), MAS_minus_B=boot(C["MAS"] - C["B0"], rng))
    # G2 complementarity (round 0) and placebo
    a, b, p = C["A0"], C["B0"], C["PLACEBO_A"]
    res["G2"] = dict(
        complementarity=boot_stat(lambda i: np.maximum(a[i], b[i]).mean() - max(a[i].mean(), b[i].mean()), Q, rng),
        placebo=boot_stat(lambda i: np.maximum(a[i], p[i]).mean() - a[i].mean(), Q, rng),
        corrected=boot_stat(lambda i: (np.maximum(a[i], b[i]).mean() - max(a[i].mean(), b[i].mean()))
                            - (np.maximum(a[i], p[i]).mean() - a[i].mean()), Q, rng))
    # G4 coverage
    k0 = np.maximum(a, b); km = np.maximum(C["A1m"], C["B1m"]); kn = np.maximum(C["A1n"], C["B1n"])
    res["G4"] = dict(kappa0=float(k0.mean()), kappa_MAS=float(km.mean()), kappa_NOCOMM=float(kn.mean()),
                     gain_MAS=float(((k0 == 0) & (km == 1)).mean()), loss_MAS=float(((k0 == 1) & (km == 0)).mean()),
                     gain_NOCOMM=float(((k0 == 0) & (kn == 1)).mean()), loss_NOCOMM=float(((k0 == 1) & (kn == 0)).mean()),
                     delta_MAS_minus_NOCOMM=boot(km - kn, rng))
    # G5 revision + evidence transfer on discordant round-0 pairs
    disc = [(qi, "A" if a[qi] == 0 else "B") for qi in range(Q) if a[qi] != b[qi]]     # (question, wrong agent)
    def rev(sfx):
        brr = [C[f"{w}1{sfx}"][qi] for qi, w in disc]
        hrr = [1 - C[f"{'B' if w == 'A' else 'A'}1{sfx}"][qi] for qi, w in disc]
        return dict(BRR=float(np.mean(brr)) if brr else None, HRR=float(np.mean(hrr)) if hrr else None,
                    DI=boot(np.array(brr) - np.array(hrr), rng) if disc else None)
    res["G5"] = dict(n_discordant=len(disc), MAS=rev("m"), NOCOMM=rev("n"))
    et = []
    for qi, w in disc:
        partner = "B" if w == "A" else "A"
        o = own[ids[qi]]
        et.append(dict(held=o[f"{partner}_held"], ret=o[f"{partner}_ret"], own_ret=o[f"{w}_ret"],
                       util_m=C[f"{w}1m"][qi], util_n=C[f"{w}1n"][qi]))
    sel = [e for e in et if e["ret"]]
    sel_x = [e for e in et if e["ret"] and not e["own_ret"]]
    res["evidence_transfer"] = dict(
        availability_held=float(np.mean([e["held"] for e in et])) if et else None,
        availability_retrieved=float(np.mean([e["ret"] for e in et])) if et else None,
        utilization_MAS=float(np.mean([e["util_m"] for e in sel])) if sel else None,
        utilization_NOCOMM=float(np.mean([e["util_n"] for e in sel])) if sel else None,
        n_available=len(sel),
        utilization_MAS_partner_only=float(np.mean([e["util_m"] for e in sel_x])) if sel_x else None,
        utilization_NOCOMM_partner_only=float(np.mean([e["util_n"] for e in sel_x])) if sel_x else None,
        n_partner_only=len(sel_x))
    et_ = res["evidence_transfer"]
    if et_["availability_retrieved"] is not None and et_["utilization_MAS"] is not None:
        et_["transfer_success_MAS"] = et_["availability_retrieved"] * et_["utilization_MAS"]
    # failure taxonomy for MAS-wrong questions
    tax = dict(retrieval_miss=0, insufficient_evidence=0, integration_failure=0)
    for qi, i in enumerate(ids):
        if C["MAS"][qi] == 0:
            o = own[i]
            if o["n_ret"] == 0:
                tax["retrieval_miss"] += 1
            elif o["n_ret"] < o["n_gold"]:
                tax["insufficient_evidence"] += 1
            else:
                tax["integration_failure"] += 1
    res["failure_taxonomy"] = tax
    pp = lambda v: "n/a" if v is None else f"{v[0]*100:+.1f} pp [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    L = [f"=== {ds} {split}: n={Q} ===",
         "accuracy: " + ", ".join(f"{r} {v:.3f}" for r, v in res["accuracy"].items()),
         *( ["F1: " + ", ".join(f"{r} {v:.3f}" for r, v in res["f1"].items())] if F1 else []),
         "tokens/q: " + ", ".join(f"{k} {v:.0f}" for k, v in T.items()),
         f"PRIMARY  P-int  MAS − No-comm  = {pp(res['P_int'])}",
         f"PRIMARY  P-util MAS − Pooled   = {pp(res['P_util'])}",
         f"G1  MAS − SC-k {pp(res['G1']['MAS_minus_SC'])} (token ratio MAS/SC {res['G1']['token_ratio_MAS_SC']:.2f}); "
         f"MAS − A {pp(res['G1']['MAS_minus_A'])}; MAS − B {pp(res['G1']['MAS_minus_B'])}; MAS/No-comm tokens {res['G1']['token_ratio_MAS_NOCOMM']:.2f}",
         f"G2  complementarity {pp(res['G2']['complementarity'])}; placebo {pp(res['G2']['placebo'])}; corrected {pp(res['G2']['corrected'])}",
         f"G4  κ0 {res['G4']['kappa0']:.3f} → MAS {res['G4']['kappa_MAS']:.3f} (gain {res['G4']['gain_MAS']:.3f}, loss {res['G4']['loss_MAS']:.3f}) | "
         f"No-comm {res['G4']['kappa_NOCOMM']:.3f}; MAS − No-comm {pp(res['G4']['delta_MAS_minus_NOCOMM'])}",
         f"G5  discordant n={len(disc)}: MAS BRR {res['G5']['MAS']['BRR']} HRR {res['G5']['MAS']['HRR']} DI {pp(res['G5']['MAS']['DI'])}; "
         f"No-comm BRR {res['G5']['NOCOMM']['BRR']} HRR {res['G5']['NOCOMM']['HRR']} DI {pp(res['G5']['NOCOMM']['DI'])}",
         "evidence transfer: " + json.dumps(res["evidence_transfer"]),
         "failure taxonomy (MAS-wrong): " + json.dumps(tax),
         f"synthesizer used: {res['synthesized']}"]
    os.makedirs(os.environ.get("WQ_OUT", f"{ROOT}/runs"), exist_ok=True)
    open(f"{os.environ.get('WQ_OUT', ROOT + '/runs')}/RESULTS_{ds}_{split}.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{os.environ.get('WQ_OUT', ROOT + '/runs')}/RESULTS_{ds}_{split}.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
