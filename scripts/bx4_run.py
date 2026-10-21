"""BX4 runs (DESIGN_BX4.md). Gold is read only by SUFF (the audit judge sees the reference answer by design).
  python bx4_run.py --dataset frames|hotpot --mode READER --out runs/bx4/<ds>_READER_gen.jsonl
  python bx4_run.py --dataset frames|hotpot --mode SUFF   --out runs/bx4/suff_<ds>.jsonl      (judge server)
  python bx4_run.py --dataset hotpot --mode EVAGG_<F_SPLIT|F_CONC|F_SHAR> --out runs/bx4/hotpot_EVAGG_<cell>_gen.jsonl"""
import argparse
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import ROOT, chat, fmt_passages, normalize, prompt
from run_webqa import call
from rx_run import Ctx, rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--mode", required=True)
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cx = Ctx(a.dataset)
    if a.mode.startswith("EVAGG_"):
        cell = a.mode[6:]; GEN = rd(f"runs/bx/hotpot_B3_{cell}_gen.jsonl")
        cells = rd("data/bx/b3_hotpot.jsonl"); ids = sorted(GEN)
    else:
        ids = sorted(i for i in cx.Q if len(cx.X[i]["G_inj"]) == 2)
    ids = ids if a.n < 0 else ids[:a.n]
    if a.mode == "SUFF":
        df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t") if a.dataset == "frames" else None
        lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json")) if a.dataset == "hotpot" else None

    def job(i):
        try:
            q = cx.Q[i]; s0 = 7_000_000 + i * 1000
            if a.mode == "READER":
                inj = [cx.pas(x) for x in cx.X[i]["G_inj"]]
                return dict(idx=i, mode="READER", READER=call(prompt("answer", question=q["question"], evidence=fmt_passages(inj)), 0.0, s0 + 500), error=None)
            if a.mode == "SUFF":
                inj = [cx.pas(x) for x in cx.X[i]["G_inj"]]
                gold = str(df.loc[i, "Answer"]) if df is not None else lab[str(i)]
                c = chat(prompt("sufficiency", question=q["question"], gold=gold, evidence=fmt_passages(inj)), 0.0, 512, 0,
                         base_url=os.environ["WQ_JUDGE_URL"], model=os.environ["WQ_JUDGE_MODEL"],
                         extra={"chat_template_kwargs": {"enable_thinking": False}})
                t = c["texts"][0]; m = re.findall(r"Verdict:\s*\**\s*(SUFFICIENT|INSUFFICIENT|AMBIGUOUS)", t or "", re.I)
                return dict(idx=i, verdict=(m[-1].upper() if m else "UNPARSED"), text=t, error=None)
            g = GEN[i]; c = cells[i]["cells"][cell]
            A, B = g["A1m"], g["B1m"]
            if A["answer"] and B["answer"] and normalize(A["answer"]) == normalize(B["answer"]):
                return dict(idx=i, mode=a.mode, EVAGG=dict(answer=A["answer"], synthesized=False, calls=[]), error=None)
            s = call(prompt("synthesize_ev", question=q["question"], answer_1=A["answer"], reasoning_1=A["text"],
                            answer_2=B["answer"], reasoning_2=B["text"], evidence=fmt_passages(c["A"] + c["B"])), 0.0, s0 + 310)
            return dict(idx=i, mode=a.mode, EVAGG=dict(answer=s["answer"], synthesized=True, calls=[s]), error=None)
        except Exception:
            return dict(idx=i, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, ids))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"BX4 {a.dataset} {a.mode}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
