"""BX generation (DESIGN_BX.md). Model, prompts (+ query / revise_retrieved), combiner and seed scheme as frozen; round-0
A0/B0 are the frozen ones. Gold is never read here (B3 cells come from bx_build.py).
REVISION modes (each agent X revises once with material from the OTHER side, then the frozen combiner):
  PEER_EV_<k>   the partner's top-k passages (partner retrieval with the question, BM25 order), revise_e1_ev framing
  CQ_Q_<k>      the same top-k passages, retrieved by X itself from the remote index, revise_retrieved framing
  CQ_GEN_<k>    X writes ONE query (prompt `query`, slot 400/401), BM25 top-k from the remote index, revise_retrieved
LOCAL_SC        k*=4 samples (T=0.7) on own evidence per agent, majority (slots 410+j / 420+j)
B3_<CELL>       HotpotQA support-recall x placement cell, full pipeline (A0, B0, A1m, B1m, A1n, B1n, MAS, NOCOMM)
  python bx_run.py --dataset frames|hotpot --mode CQ_GEN_3 --out runs/bx/frames_CQ_GEN_3_gen.jsonl [--n N]"""
import argparse
import json
import os
import re
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, normalize, prompt
from run_webqa import SLOT, call, combine
from rx_run import FROZEN, Ctx, rd

QSLOT = dict(A=400, B=401)
SCSLOT = dict(A=410, B=420)
K_SC = 4


def parse_query(text, question):
    m = re.findall(r"Query:\s*(.+)", text or "")
    q = m[-1].strip().strip('"').strip() if m else ""
    return (q or question), bool(q)


class BX:
    def __init__(self, ds, mode):
        self.cx = Ctx(ds); self.ds = ds
        self.rank = rd(f"data/bx/rank_hotpot.jsonl") if ds == "hotpot" else None
        self.search = None
        if mode.startswith("CQ_GEN"):
            from bx_build import frames_remote, hotpot_remote
            self.search = frames_remote() if ds == "frames" else hotpot_remote()

    def ranked(self, q, side):
        return list(q[side]) if self.ds == "frames" else [q[side][j] for j in self.rank[q["idx"]][side]]

    def remote(self, q, side, query, k):
        return self.search(side, query, k) if self.ds == "frames" else self.search(q[side], query, k)


def key(ds, p):
    return p["pid"] if ds == "frames" else p["title"]


def run_rev(bx, mode, q, g0):
    fam, k = mode.rsplit("_", 1); k = int(k)
    s0 = 7_000_000 + q["idx"] * 1000
    out = dict(idx=q["idx"], mode=mode, A0=g0["A0"], B0=g0["B0"], meta={})
    if fam == "CQ_GEN":
        out["qgen"] = {}
    for me, other, slot in (("A", "B", "A1m"), ("B", "A", "B1m")):
        own = g0[f"{me}0"]; meta = {}
        if fam == "CQ_GEN":
            c = call(prompt("query", question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=own["text"]),
                     0.0, s0 + QSLOT[me])
            qtext, ok = parse_query(c["text"], q["question"]); out["qgen"][me] = c
            pev = bx.remote(q, other, qtext, k); pname = "revise_retrieved"
            peer5 = {key(bx.ds, p) for p in bx.ranked(q, other)}
            meta.update(query=qtext, query_parsed=ok, overlap_with_peer_top5=sum(key(bx.ds, p) in peer5 for p in pev))
        else:
            pev = bx.ranked(q, other)[:k]
            pname = "revise_e1_ev" if fam == "PEER_EV" else "revise_retrieved"
        out[slot] = call(prompt(pname, question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=own["text"],
                                partner_evidence=fmt_passages(pev, start=len(q[me]) + 1)), 0.0, s0 + SLOT[slot])
        meta.update(ids=[key(bx.ds, p) for p in pev], sent_n=len(pev), sent_words=sum(len(p["text"].split()) for p in pev),
                    gold_sent=sum(bx.cx.gold(q, p) for p in pev))
        out["meta"][me] = meta
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    return out


def majority(sm):
    norm = [normalize(s["answer"]) for s in sm if s["answer"]]; c = Counter(norm)
    return next((s["answer"] for s in sm if s["answer"] and c[normalize(s["answer"])] == max(c.values())), None) if c else None


def run_local_sc(q):
    s0 = 7_000_000 + q["idx"] * 1000
    out = dict(idx=q["idx"], mode="LOCAL_SC")
    for side in "AB":
        p = prompt("answer", question=q["question"], evidence=fmt_passages(q[side]))
        sm = [call(p, 0.7, s0 + SCSLOT[side] + j) for j in range(K_SC)]
        out[f"{side}_SC"] = dict(answer=majority(sm), k=K_SC, samples=sm)
    return out


def run_b3(bx, cells, mode, q):
    c = cells[q["idx"]]
    if not c["eligible"]:
        return dict(idx=q["idx"], mode=mode, skipped=True)
    q = dict(q, A=c["cells"][mode[3:]]["A"], B=c["cells"][mode[3:]]["B"])
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    out = dict(idx=q["idx"], mode=mode, holder=c["holder"], ids=dict(A=[p["title"] for p in q["A"]], B=[p["title"] for p in q["B"]]),
               gold=dict(A=sum(bx.cx.gold(q, p) for p in q["A"]), B=sum(bx.cx.gold(q, p) for p in q["B"])))
    out["A0"] = call(prompt("answer", question=q["question"], evidence=ev["A"]), 0.0, s0 + SLOT["A0"])
    out["B0"] = call(prompt("answer", question=q["question"], evidence=ev["B"]), 0.0, s0 + SLOT["B0"])
    for me, other, sm, sn in (("A", "B", "A1m", "A1n"), ("B", "A", "B1m", "B1n")):
        own, part = out[f"{me}0"], out[f"{other}0"]
        out[sm] = call(prompt("revise_mas", question=q["question"], evidence=ev[me], own_reasoning=own["text"],
                              partner_answer=part["answer"], partner_reasoning=part["text"],
                              partner_evidence=fmt_passages(q[other], start=len(q[me]) + 1)), 0.0, s0 + SLOT[sm])
        out[sn] = call(prompt("revise_nocomm", question=q["question"], evidence=ev[me], own_reasoning=own["text"]),
                       0.0, s0 + SLOT[sn])
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    out["NOCOMM"] = combine(q, out["A1n"], out["B1n"], s0 + SLOT["SYNn"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["frames", "hotpot"])
    ap.add_argument("--mode", required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert re.fullmatch(r"(PEER_EV|CQ_Q|CQ_GEN)_[135]|LOCAL_SC|B3_(NONE|P_EXCL|P_SHAR|F_SPLIT|F_CONC|F_SHAR)", a.mode), a.mode
    assert not (a.mode.startswith("B3_") and a.dataset != "hotpot"), "B3 is HotpotQA only (DESIGN_BX.md)"
    bx = BX(a.dataset, a.mode)
    G0 = rd(FROZEN[a.dataset])
    cells = rd("data/bx/b3_hotpot.jsonl") if a.mode.startswith("B3_") else None
    Q = sorted(bx.cx.Q.values(), key=lambda q: q["idx"]); Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            if a.mode == "LOCAL_SC":
                r = run_local_sc(q)
            elif a.mode.startswith("B3_"):
                r = run_b3(bx, cells, a.mode, q)
            else:
                r = run_rev(bx, a.mode, q, G0[q["idx"]])
            return dict(r, error=None)
        except Exception:
            return dict(idx=q["idx"], mode=a.mode, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"BX {a.dataset} {a.mode}: n={len(rows)} skipped={sum(bool(r.get('skipped')) for r in rows)} "
          f"errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
