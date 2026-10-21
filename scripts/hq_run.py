"""HQ generation (DESIGN_HQ.md). Agent A = Source T (table rows), agent B = Source P (linked passages). Gold is never read.
  BASE        A0, B0 (round 0, own top 5), A1n / B1n + NOCOMM (LOCAL_NOCOMM), UNI (one agent on both lists, slot 2)
  PEER_EV     revise_e1_ev with the partner's top 5                          (needs BASE)
  PEER_FULL   revise_mas with the partner's answer, reasoning and top 5        (needs BASE)
  REMOTE_Q    revise_retrieved with top 5 from the other source for the question (= partner's list)   (needs BASE)
  REMOTE_GEN  query (slot 400/401) -> top 5 from the other source -> revise_retrieved               (needs BASE)
  python hq_run.py --mode BASE --out runs/hq/hq_BASE_gen.jsonl [--n N]"""
import argparse
import json
import os
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, prompt
from run_webqa import SLOT, call, combine
from bx_run import QSLOT, parse_query

OTHER = dict(A="B", B="A")


def load_sample():
    return [json.loads(l) for l in open(f"{ROOT}/data/hq/sample.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["BASE", "PEER_EV", "PEER_FULL", "REMOTE_Q", "REMOTE_GEN"])
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = load_sample(); S = S if a.n < 0 else S[:a.n]
    base = None if a.mode == "BASE" else {r["k"]: r for r in map(json.loads, open(f"{ROOT}/runs/hq/hq_BASE_gen.jsonl"))}
    idx = None
    if a.mode == "REMOTE_GEN":
        from rx_build import tok
        I = pickle.load(open(f"{ROOT}/data/hq/index.pkl", "rb"))

        def search(source, query, k=5):
            import numpy as np
            bm, docs = (I["bmT"], I["rows"]) if source == "A" else (I["bmP"], I["passages"])
            sc = bm.get_scores(tok(query)); j = np.argpartition(-sc, k)[:k]
            return [docs[t] for t in sorted(j, key=lambda t: (-sc[t], t))]
        idx = search

    def job(s):
        try:
            k = s["k"]; s0 = 9_000_000 + k * 1000; q = s["question"]; ev = {x: fmt_passages(s[x]) for x in "AB"}
            ids = lambda ps: [p["id"] for p in ps]
            if a.mode == "BASE":
                out = dict(k=k, mode="BASE")
                out["A0"] = call(prompt("answer", question=q, evidence=ev["A"]), 0.0, s0 + SLOT["A0"])
                out["B0"] = call(prompt("answer", question=q, evidence=ev["B"]), 0.0, s0 + SLOT["B0"])
                for me, sn in (("A", "A1n"), ("B", "B1n")):
                    out[sn] = call(prompt("revise_nocomm", question=q, evidence=ev[me], own_reasoning=out[f"{me}0"]["text"]), 0.0, s0 + SLOT[sn])
                out["NOCOMM"] = combine(dict(question=q), out["A1n"], out["B1n"], s0 + SLOT["SYNn"])
                out["UNI"] = call(prompt("answer", question=q, evidence=fmt_passages(s["A"] + s["B"])), 0.0, s0 + SLOT["POOL"])
                return dict(out, error=None)
            b = base[k]; out = dict(k=k, mode=a.mode, A0=b["A0"], B0=b["B0"], meta={})
            if a.mode == "REMOTE_GEN":
                out["qgen"] = {}
            for me, slot in (("A", "A1m"), ("B", "B1m")):
                other = OTHER[me]; own = b[f"{me}0"]; kw = dict(question=q, evidence=ev[me], own_reasoning=own["text"])
                if a.mode == "PEER_EV":
                    pev = s[other]; m = prompt("revise_e1_ev", partner_evidence=fmt_passages(pev, start=6), **kw)
                elif a.mode == "PEER_FULL":
                    pev = s[other]; part = b[f"{other}0"]
                    m = prompt("revise_mas", partner_answer=part["answer"], partner_reasoning=part["text"],
                               partner_evidence=fmt_passages(pev, start=6), **kw)
                elif a.mode == "REMOTE_Q":
                    pev = s[other]; m = prompt("revise_retrieved", partner_evidence=fmt_passages(pev, start=6), **kw)
                else:
                    c = call(prompt("query", **kw), 0.0, s0 + QSLOT[me]); out["qgen"][me] = c
                    query, ok = parse_query(c["text"], q); pev = idx(other, query)
                    out["meta"][me] = dict(query=query, query_parsed=ok)
                    m = prompt("revise_retrieved", partner_evidence=fmt_passages(pev, start=6), **kw)
                out[slot] = call(m, 0.0, s0 + SLOT[slot])
                out["meta"].setdefault(me, {})["received_ids"] = ids(pev)
            out["MAS"] = combine(dict(question=q), out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
            return dict(out, error=None)
        except Exception:
            return dict(k=s["k"], mode=a.mode, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, S))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"HQ {a.mode}: n={len(rows)} errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
