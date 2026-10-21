"""POST-HOC EXPLORATORY aggregation decomposition (not pre-registered; user request after the W2 results). Logs only, no
model calls. For each condition, the communication gain P-int = MAS − No-comm is split into
  revision component    = mean(A1m, B1m) − mean(A1n, B1n)            (individual revisions after the round)
  aggregation component = [MAS − mean(A1m, B1m)] − [No-comm − mean(A1n, B1n)]   (what the shared final combiner adds)
The two add up to P-int exactly. Paired bootstrap over questions (10,000, seed 0), descriptive.
Also: post-revision answer agreement, synthesizer share, accuracy when the two revisions agree vs when synthesized,
and, when exactly one revision is correct, how often the final answer is correct (the combiner's pick).
Correctness: FRAMES = frozen judge verdicts; HotpotQA = EM.
  python explore_aggregation.py  ->  runs/posthoc/EXPLORE_AGGREGATION.{txt,json}"""
import json
import os

import numpy as np

from common import ROOT, normalize


def load_frames(gen, judged):
    G = {g["idx"]: g for g in map(json.loads, open(f"{ROOT}/{gen}"))}
    J = {j["idx"]: j for j in map(json.loads, open(f"{ROOT}/{judged}"))}
    return G, lambda i, r: float(J[i][r]["verdict"])


def load_hotpot(gen):
    G = {g["idx"]: g for g in map(json.loads, open(f"{ROOT}/{gen}"))}
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    return G, lambda i, r: float(normalize(G[i][r]["answer"] or "") == normalize(lab[str(i)]))


def decompose(G, c):
    ids = sorted(G); n = len(ids)
    v = {r: np.array([c(i, r) for i in ids]) for r in ("A1m", "B1m", "A1n", "B1n", "MAS", "NOCOMM")}
    rm, rn = (v["A1m"] + v["B1m"]) / 2, (v["A1n"] + v["B1n"]) / 2
    comp = dict(P_int=v["MAS"] - v["NOCOMM"], revision=rm - rn, aggregation=(v["MAS"] - rm) - (v["NOCOMM"] - rn))
    idx = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x: [float(x.mean()), *map(float, np.percentile(x[idx].mean(1), [2.5, 97.5]))]
    out = {k: ci(x) for k, x in comp.items()}
    for path, a, b in (("MAS", "A1m", "B1m"), ("NOCOMM", "A1n", "B1n")):
        agree = np.array([bool(G[i][a]["answer"]) and normalize(G[i][a]["answer"]) == normalize(G[i][b]["answer"] or "") for i in ids])
        syn = np.array([G[i][path]["synthesized"] for i in ids])
        one = (v[a] + v[b]) == 1
        out[path] = dict(rev_acc=float(((v[a] + v[b]) / 2).mean()), final_acc=float(v[path].mean()),
                         agree_after_revision=float(agree.mean()), synth_share=float(syn.mean()),
                         acc_if_agree=float(v[path][~syn].mean()) if (~syn).any() else None,
                         acc_if_synth=float(v[path][syn].mean()) if syn.any() else None,
                         both_correct=float(((v[a] + v[b]) == 2).mean()), exactly_one_correct=float(one.mean()),
                         final_correct_given_one=float(v[path][one].mean()) if one.any() else None)
    return out


def main():
    S = {"FRAMES D (final)": load_frames("runs/final/frames_final_gen.jsonl", "runs/final/frames_final_judged.jsonl")}
    for c in ("S_UNION", "S_5", "S_UNION_T7"):
        S[f"FRAMES {c}"] = load_frames(f"runs/dist/{c}_final_gen.jsonl", f"runs/dist/{c}_final_judged.jsonl")
    S["HotpotQA D (final)"] = load_hotpot("runs/final/hotpot_final_gen.jsonl")
    for c in ("S_UNION", "S_5", "S_UNION_T7"):
        p = f"runs/dist_hotpot/{c}_final_gen.jsonl"
        if os.path.exists(f"{ROOT}/{p}"):
            S[f"HotpotQA {c}"] = load_hotpot(p)
    res, L = {}, ["=== POST-HOC EXPLORATORY (not pre-registered): aggregation decomposition of P-int ==="]
    pp = lambda v: f"{v[0]*100:+.1f} [{v[1]*100:+.1f}, {v[2]*100:+.1f}]"
    for name, (G, c) in S.items():
        r = res[name] = decompose(G, c)
        L.append(f"[{name}] P-int {pp(r['P_int'])} = revision {pp(r['revision'])} + aggregation {pp(r['aggregation'])}")
        for path in ("MAS", "NOCOMM"):
            x = r[path]
            L.append(f"    {path:6s} rev-acc {x['rev_acc']:.3f} → final {x['final_acc']:.3f}; agree-after-rev {x['agree_after_revision']:.3f}; "
                     f"synth {x['synth_share']:.3f}; acc|agree {x['acc_if_agree']:.3f} acc|synth {x['acc_if_synth']:.3f}; "
                     f"both-correct {x['both_correct']:.3f}, one-correct {x['exactly_one_correct']:.3f} → final correct {x['final_correct_given_one']:.3f}")
    open(f"{ROOT}/runs/posthoc/EXPLORE_AGGREGATION.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/runs/posthoc/EXPLORE_AGGREGATION.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
