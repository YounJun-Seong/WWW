"""POST-HOC EXPLORATORY analysis of W2 (not pre-registered; written after the W2 results were seen, at the user's request).
It reads only the frozen W2 / FRAMES final outputs, and asks why No-comm rises from D to the Shared conditions:
more evidence per agent, more gold evidence per agent, different round-0 behaviour, or a different combiner path.
Correctness = the frozen judge verdicts. No bootstrap inference: descriptive only, with denominators.
  python explore_w2.py  ->  runs/posthoc/EXPLORE_W2.{txt,json}"""
import json
from collections import defaultdict

import numpy as np

from common import ROOT, normalize

CONDS = ("D", "S_UNION", "S_5", "S_UNION_T7")


def main():
    P = {p["pid"]: p["page"] for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    src = dict(D=("runs/final/frames_final_gen.jsonl", "runs/final/frames_final_judged.jsonl"),
               **{c: (f"runs/dist/{c}_final_gen.jsonl", f"runs/dist/{c}_final_judged.jsonl") for c in CONDS[1:]})
    G = {c: {g["idx"]: g for g in map(json.loads, open(f"{ROOT}/{a}"))} for c, (a, _) in src.items()}
    J = {c: {j["idx"]: j for j in map(json.loads, open(f"{ROOT}/{b}"))} for c, (_, b) in src.items()}
    ids = sorted(G["D"])
    pids = lambda c, i, s: (R[i][s] if c == "D" else G[c][i]["pids"][s])
    out, L = {}, ["=== POST-HOC EXPLORATORY (not pre-registered): W2 mechanism, FRAMES final n=%d, judge-based ===" % len(ids)]
    per_q = defaultdict(dict)
    for c in CONDS:
        rows = []
        for i in ids:
            g, j, gold = G[c][i], J[c][i], set(R[i]["gold_pages"])
            ev = {s: pids(c, i, s) for s in "AB"}
            gp = {s: {P[p] for p in ev[s]} & gold for s in "AB"}
            ans = {s: g[f"{s}0"]["answer"] for s in "AB"}
            rows.append(dict(
                n_pass=np.mean([len(ev["A"]), len(ev["B"])]),
                in_tok_r0=np.mean([g["A0"]["input_tokens"], g["B0"]["input_tokens"]]),
                in_tok_rev_nc=np.mean([g["A1n"]["input_tokens"], g["B1n"]["input_tokens"]]),
                in_tok_rev_mas=np.mean([g["A1m"]["input_tokens"], g["B1m"]["input_tokens"]]),
                agent_any_gold=np.mean([bool(gp[s]) for s in "AB"]),
                agent_all_gold=np.mean([gp[s] == gold for s in "AB"]) if gold else 0.0,
                team_any_gold=bool(gp["A"] | gp["B"]), team_all_gold=(gp["A"] | gp["B"]) == gold,
                acc_r0=np.mean([j["A0"]["verdict"], j["B0"]["verdict"]]),
                both_r0_correct=j["A0"]["verdict"] and j["B0"]["verdict"],
                disc_r0=j["A0"]["verdict"] != j["B0"]["verdict"],
                ans_agree_r0=bool(ans["A"]) and normalize(ans["A"]) == normalize(ans["B"] or ""),
                acc_rev_nc=np.mean([j["A1n"]["verdict"], j["B1n"]["verdict"]]),
                acc_rev_mas=np.mean([j["A1m"]["verdict"], j["B1m"]["verdict"]]),
                syn_nc=g["NOCOMM"]["synthesized"], syn_mas=g["MAS"]["synthesized"],
                nc=j["NOCOMM"]["verdict"], mas=j["MAS"]["verdict"]))
            per_q[i][c] = rows[-1]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for path in ("nc", "mas"):
            s = [r for r in rows if r[f"syn_{path}"]]; a = [r for r in rows if not r[f"syn_{path}"]]
            m[f"{path}_acc_when_synth"] = float(np.mean([r[path] for r in s])) if s else None
            m[f"{path}_acc_when_agree"] = float(np.mean([r[path] for r in a])) if a else None
        out[c] = m
    keys = [("evidence passages per agent", "n_pass", "{:.1f}"), ("input tokens, round 0 (per agent)", "in_tok_r0", "{:.0f}"),
            ("input tokens, No-comm revision", "in_tok_rev_nc", "{:.0f}"), ("input tokens, MAS revision", "in_tok_rev_mas", "{:.0f}"),
            ("agent holds >=1 gold page (per agent)", "agent_any_gold", "{:.3f}"), ("agent holds all gold pages", "agent_all_gold", "{:.3f}"),
            ("team (A ∪ B) holds >=1 gold page", "team_any_gold", "{:.3f}"), ("team holds all gold pages", "team_all_gold", "{:.3f}"),
            ("round-0 accuracy (mean of A0, B0)", "acc_r0", "{:.3f}"), ("round-0 answer agreement (normalized string)", "ans_agree_r0", "{:.3f}"),
            ("round-0 correctness-discordant share", "disc_r0", "{:.3f}"),
            ("No-comm revision accuracy (mean A1n, B1n)", "acc_rev_nc", "{:.3f}"), ("MAS revision accuracy (mean A1m, B1m)", "acc_rev_mas", "{:.3f}"),
            ("No-comm synthesizer share", "syn_nc", "{:.3f}"), ("MAS synthesizer share", "syn_mas", "{:.3f}"),
            ("No-comm final accuracy", "nc", "{:.3f}"), ("MAS final accuracy", "mas", "{:.3f}"),
            ("No-comm acc | answers agreed (no synth)", "nc_acc_when_agree", "{:.3f}"), ("No-comm acc | synthesized", "nc_acc_when_synth", "{:.3f}"),
            ("MAS acc | answers agreed (no synth)", "mas_acc_when_agree", "{:.3f}"), ("MAS acc | synthesized", "mas_acc_when_synth", "{:.3f}")]
    L.append(f"{'metric':48s}" + "".join(f"{c:>12s}" for c in CONDS))
    for name, k, f in keys:
        L.append(f"{name:48s}" + "".join(f"{(f.format(out[c][k]) if out[c][k] is not None else 'n/a'):>12s}" for c in CONDS))
    # stratify the No-comm change D -> S_UNION by whether the agents' own gold coverage rose
    strata = defaultdict(list)
    for i in ids:
        d, s = per_q[i]["D"], per_q[i]["S_UNION"]
        key = ("no gold in team" if not d["team_any_gold"] else
               "per-agent gold coverage rose" if s["agent_any_gold"] > d["agent_any_gold"] else "per-agent gold coverage unchanged")
        strata[key].append((d["nc"], s["nc"], d["acc_r0"], s["acc_r0"]))
    L.append("No-comm D → S_UNION by stratum (questions; No-comm acc D → S_UNION; round-0 acc D → S_UNION):")
    out["strata_D_to_S_UNION"] = {}
    for k, v in strata.items():
        a = np.array(v, float)
        out["strata_D_to_S_UNION"][k] = dict(n=len(v), nc_D=float(a[:, 0].mean()), nc_SU=float(a[:, 1].mean()),
                                             r0_D=float(a[:, 2].mean()), r0_SU=float(a[:, 3].mean()))
        L.append(f"  {k:35s} n={len(v):4d}  No-comm {a[:, 0].mean():.3f} → {a[:, 1].mean():.3f}   round-0 {a[:, 2].mean():.3f} → {a[:, 3].mean():.3f}")
    open(f"{ROOT}/runs/posthoc/EXPLORE_W2.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{ROOT}/runs/posthoc/EXPLORE_W2.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
