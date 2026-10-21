"""Analysis of the final-cohort judge audit (W5), fixed before any annotation. Run once after the user scores the sheet.
Per stratum s (MAS vs No-comm / Pooled / SC): judge–human agreement on the audited responses; on the sampled discordant
questions, the judge gap g_J = mean(j_MAS − j_other) and the human gap g_H = mean(h_MAS − h_other). Human-adjusted
full-cohort difference = Δ_judge − (N_s / 624)·(g_J − g_H), with N_s the number of judge-discordant questions in the
final; bootstrap over the sampled questions (10,000, seed 0). Assumption (stated): judge-concordant questions keep
their ordering (not audited).
  python judge_audit_analyze.py data/judge_audit_final_scored.xlsx  ->  runs/final/JUDGE_AUDIT_FINAL.{txt,json}"""
import json
import sys
from collections import defaultdict

import numpy as np
import openpyxl

from common import ROOT


def main():
    ws = openpyxl.load_workbook(sys.argv[1])["Annotate"]
    lab = {int(r[0]): int(str(r[4]).strip()) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None}
    key = json.load(open(f"{ROOT}/data/judge_audit_final_key.json"))
    assert len(lab) == len(key) and set(lab.values()) <= {0, 1}
    J = {j["idx"]: j for j in map(json.loads, open(f"{ROOT}/runs/final/frames_final_judged.jsonl"))}
    n = len(J)
    by = defaultdict(dict)
    for k in key:
        by[(k["stratum"], k["idx"])][k["role"]] = (int(k["judge"]), lab[k["item"]])
    res, L = {}, ["=== FRAMES final judge audit (blind, 80 responses) ==="]
    h_all = [lab[k["item"]] for k in key]; j_all = [int(k["judge"]) for k in key]
    po = np.mean([a == b for a, b in zip(h_all, j_all)]); pa, pb = np.mean(h_all), np.mean(j_all); pe = pa * pb + (1 - pa) * (1 - pb)
    res["overall"] = dict(agreement=float(po), kappa=float((po - pe) / (1 - pe)))
    L.append(f"overall agreement {po:.3f}, kappa {(po - pe) / (1 - pe):.3f}")
    rng = np.random.default_rng(0)
    for other in ("NOCOMM", "POOL", "SC"):
        st = f"MAS_vs_{other}"; qs = [q for (s, q) in by if s == st]
        gJ = np.array([by[(st, q)]["MAS"][0] - by[(st, q)][other][0] for q in qs], float)
        gH = np.array([by[(st, q)]["MAS"][1] - by[(st, q)][other][1] for q in qs], float)
        N = sum(J[i]["MAS"]["verdict"] != J[i][other]["verdict"] for i in J)
        dJ = np.mean([float(J[i]["MAS"]["verdict"]) - float(J[i][other]["verdict"]) for i in J])
        idx = rng.integers(0, len(qs), (10000, len(qs)))
        adj = dJ - (N / n) * (gJ[idx].mean(1) - gH[idx].mean(1))
        agree = np.mean([by[(st, q)][r][0] == by[(st, q)][r][1] for q in qs for r in ("MAS", other)])
        res[st] = dict(n_questions=len(qs), N_discordant=N, judge_diff=float(dJ), judge_gap_sample=float(gJ.mean()),
                       human_gap_sample=float(gH.mean()), agreement=float(agree),
                       human_adjusted_diff=[float(dJ - (N / n) * (gJ.mean() - gH.mean())), *map(float, np.percentile(adj, [2.5, 97.5]))])
        r = res[st]
        L.append(f"{st}: audited {len(qs)} of {N} discordant; agreement {agree:.3f}; sample gap judge {gJ.mean():+.3f} vs human {gH.mean():+.3f}; "
                 f"full-cohort diff judge {dJ*100:+.1f} pp -> human-adjusted {r['human_adjusted_diff'][0]*100:+.1f} pp "
                 f"[{r['human_adjusted_diff'][1]*100:+.1f}, {r['human_adjusted_diff'][2]*100:+.1f}]")
    open(f"{ROOT}/runs/final/JUDGE_AUDIT_FINAL.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/runs/final/JUDGE_AUDIT_FINAL.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
