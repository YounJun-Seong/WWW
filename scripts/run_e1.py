"""E1 — causal evidence-transfer intervention (DESIGN_DRAFT.md, Experiment E1). Generation only; gold never read.
Same questions, the Distributed (D) evidence layout and the *frozen* round-0 answers (read from the frozen D run), so
only the revision input changes. The final combiner is the frozen one.
  ANS      partner answer only                      (prompts/revise_e1_ans.txt)
  ANS_RSN  partner answer + reasoning               (revise_e1_ansrsn.txt)
  ANS_EV   partner answer + genuine partner passages (revise_e1_ansev.txt)
  ANS_IRR  partner answer + irrelevant passages: the partner's passages of the NEXT question in the sorted list (cyclic)
  EV       partner passages only, no answer         (revise_e1_ev.txt)
Output roles use the judge's names: A1m/B1m = the revisions under the mode, MAS = the combined final answer.
Reused from the frozen D run (not regenerated): No-comm (A1n/B1n) and FULL = MAS (answer + reasoning + passages).
  python run_e1.py --dataset frames|hotpot --mode ANS --out runs/e1/frames_ANS_gen.jsonl [--n N]"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, prompt
from run_webqa import SLOT, call, combine, load

MODES = {"ANS": "revise_e1_ans", "ANS_RSN": "revise_e1_ansrsn", "ANS_EV": "revise_e1_ansev", "ANS_IRR": "revise_e1_ansev",
         "EV": "revise_e1_ev"}
FROZEN = dict(frames="runs/final/frames_final_gen.jsonl", hotpot="runs/final/hotpot_final_gen.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["frames", "hotpot"])
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = sorted(load(a.dataset, "final"), key=lambda q: q["idx"])
    R0 = {g["idx"]: g for g in map(json.loads, open(f"{ROOT}/{FROZEN[a.dataset]}"))}
    nxt = {q["idx"]: Q[(k + 1) % len(Q)] for k, q in enumerate(Q)}
    sel = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            s0 = 7_000_000 + q["idx"] * 1000
            g0 = R0[q["idx"]]
            out = dict(idx=q["idx"], mode=a.mode, A0=g0["A0"], B0=g0["B0"])
            for me, other, slot in (("A", "B", "A1m"), ("B", "A", "B1m")):
                own, part = g0[f"{me}0"], g0[f"{other}0"]
                pev = nxt[q["idx"]][other] if a.mode == "ANS_IRR" else q[other]
                kw = dict(question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=own["text"],
                          partner_answer=part["answer"], partner_reasoning=part["text"],
                          partner_evidence=fmt_passages(pev, start=len(q[me]) + 1))
                out[slot] = call(prompt(MODES[a.mode], **kw), 0.0, s0 + SLOT[slot])
            out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
            if a.mode == "ANS_IRR":
                out["irrelevant_source_idx"] = nxt[q["idx"]]["idx"]
            return dict(out, error=None)
        except Exception:
            return dict(idx=q["idx"], mode=a.mode, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, sel))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"E1 {a.dataset} {a.mode}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
