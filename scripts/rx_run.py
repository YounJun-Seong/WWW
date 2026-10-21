"""RX generation (WWW-final round). Generation only: gold is never read here — gold-informed interventions come from the
assignments built in advance by rx_build.py (data/rx/). Model, prompts, combiner and seed scheme are the frozen ones.

REVISION-ONLY modes (the frozen round-0 A0/B0 of the D run are reused; only the partner message changes; EV prompt =
prompts/revise_e1_ev.txt, i.e. partner passages only, no partner answer):
  R1_FULL       partner's passages, unchanged (= E1 EV)
  R1_GOLDRM     partner's passages with the gold ones removed
  R1_GOLDREP    gold passages replaced, in place, by length-matched hard negatives
  R1_NONGOLDRM  only the partner's gold passages are sent
  R2_I1a/I1b    injection: one gold passage (the first / the second of G_inj)            [receiver's own evidence fixed]
  R2_I2         injection: the full gold set G_inj
  R2_I3         injection: G_inj + 3 hard negatives (shuffled)
  R2_I4         injection: G_inj + the partner's round-0 answer (revise_e1_ansev prompt)
  R10_SHORT     HotpotQA: only the supporting SENTENCES of the partner's gold paragraphs
  R10_HARDNEG   every partner passage replaced by a length-matched hard negative
  R10_RANDNEG   the partner-side passages of the next question (random, unrelated)
  R12_N3/N6     partner's passages + 3 / 6 hard negatives (shuffled)
  R12_GOLDLAST  partner's passages with the gold ones moved to the end
LAYOUT modes (full pipeline A0, B0, A1m, B1m, A1n, B1n, MAS, NOCOMM with new per-agent lists):
  E2H_<CELL>    HotpotQA E2 cell;  OVL<o>  HotpotQA overlap o=1..5 (W4 rule);  DPAD  own 5 + 5 hard-negative pads
  SEED<s>_D / SEED<s>_SU   partition seed s: Distributed / Shared-union
SINGLE-AGENT: UNI   UNI_ONE (answer on A5∪B5), UNI_REV (+ one self-revision), UNI_SC (k samples at T=0.7, majority)
AGGREGATION:  AGG   on frozen revisions (D MAS, D No-comm, E1 EV): ALWAYS_* (synthesizer always) and EVSYN_* (on
                    disagreement, a synthesizer that also sees the union of both agents' passages)
  python rx_run.py --dataset frames|hotpot --mode R1_GOLDRM --out runs/rx/frames_R1_GOLDRM_gen.jsonl [--n N]"""
import argparse
import json
import os
import random
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, fmt_passages, normalize, prompt
from run_webqa import SLOT, call, combine, load

FROZEN = dict(frames="runs/final/frames_final_gen.jsonl", hotpot="runs/final/hotpot_final_gen.jsonl")
E1EV = dict(frames="runs/e1/frames_EV_gen.jsonl", hotpot="runs/e1/hotpot_EV_gen.jsonl")
REV = {"R1_FULL", "R1_GOLDRM", "R1_GOLDREP", "R1_NONGOLDRM", "R2_I1a", "R2_I1b", "R2_I2", "R2_I3", "R2_I4",
       "R10_SHORT", "R10_HARDNEG", "R10_RANDNEG", "R12_N3", "R12_N6", "R12_GOLDLAST"}
NONE = "(no passages)"


def rd(p):
    return {x["idx"]: x for x in map(json.loads, open(f"{ROOT}/{p}"))}


class Ctx:
    def __init__(self, ds):
        self.ds = ds
        self.Q = {q["idx"]: q for q in load(ds, "final")}
        self.X = rd(f"data/rx/assign_{ds}.jsonl")
        if ds == "frames":
            self.P = {p["pid"]: p for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
            self.gold = lambda q, p: p["pid"] in set(self.X[q["idx"]]["gold_pids"])
            self.pas = lambda x: self.P[x]                 # FRAMES assignments store pids
        else:
            self.gold = lambda q, p: p["title"] in set(self.X[q["idx"]]["gold_titles"])
            self.pas = lambda x: x                         # HotpotQA assignments store paragraph dicts

    def hn(self, q):
        return [self.pas(x) for x in self.X[q["idx"]]["HN"]]


def wc(p):
    return len(p["text"].split())


def length_matched(targets, pool, used):
    out = []
    for t in targets:
        cand = [p for p in pool if id(p) not in used]
        if not cand:
            return None
        best = min(cand, key=lambda p: (abs(wc(p) - wc(t)), pool.index(p)))
        used.add(id(best)); out.append(best)
    return out


def message(cx, mode, q, me, other, g0):
    """-> (prompt name, partner passages, extra kw, meta) or None when the question is not eligible for the mode."""
    part = q[other]
    isg = [cx.gold(q, p) for p in part]
    meta = dict(partner_n=len(part), partner_gold=sum(isg))
    rng = random.Random(2027 + q["idx"] + (0 if me == "A" else 1))
    if mode == "R1_FULL":
        return "revise_e1_ev", part, {}, meta
    if mode == "R1_GOLDRM":
        return "revise_e1_ev", [p for p, g in zip(part, isg) if not g], {}, meta
    if mode == "R1_GOLDREP":
        rep = length_matched([p for p, g in zip(part, isg) if g], cx.hn(q), set())
        if rep is None:
            return None
        it = iter(rep)
        return "revise_e1_ev", [next(it) if g else p for p, g in zip(part, isg)], {}, meta
    if mode == "R1_NONGOLDRM":
        return "revise_e1_ev", [p for p, g in zip(part, isg) if g], {}, meta
    if mode.startswith("R2_"):
        G = [cx.pas(x) for x in cx.X[q["idx"]]["G_inj"]]
        meta["own_gold"] = sum(cx.gold(q, p) for p in q[me]); meta["G_n"] = len(G)
        if not G or (mode == "R2_I1b" and len(G) < 2):
            return None
        if mode == "R2_I1a":
            return "revise_e1_ev", G[:1], {}, meta
        if mode == "R2_I1b":
            return "revise_e1_ev", G[1:2], {}, meta
        if mode == "R2_I2":
            return "revise_e1_ev", G, {}, meta
        if mode == "R2_I3":
            ps = G + cx.hn(q)[:3]; rng.shuffle(ps)
            return "revise_e1_ev", ps, {}, meta
        if mode == "R2_I4":
            return "revise_e1_ansev", G, dict(partner_answer=g0[f"{other}0"]["answer"],
                                              partner_reasoning=g0[f"{other}0"]["text"]), meta
    if mode == "R10_SHORT":
        sup = cx.X[q["idx"]]["sup_sents"]
        return "revise_e1_ev", [dict(title=p["title"], text=sup[p["title"]]) for p, g in zip(part, isg) if g and p["title"] in sup], {}, meta
    if mode == "R10_HARDNEG":
        rep = length_matched(part, cx.hn(q), set())
        return None if rep is None else ("revise_e1_ev", rep, {}, meta)
    if mode == "R10_RANDNEG":
        return "revise_e1_ev", cx.Q[cx.X[q["idx"]]["RN_from"]][other], {}, meta
    if mode in ("R12_N3", "R12_N6"):
        k = 3 if mode == "R12_N3" else 6
        ps = list(part) + cx.hn(q)[10:10 + k]
        if len(ps) < len(part) + k:
            return None
        rng.shuffle(ps)
        return "revise_e1_ev", ps, {}, meta
    if mode == "R12_GOLDLAST":
        return "revise_e1_ev", [p for p, g in zip(part, isg) if not g] + [p for p, g in zip(part, isg) if g], {}, meta
    raise ValueError(mode)


def run_rev(cx, mode, q, g0):
    s0 = 7_000_000 + q["idx"] * 1000
    out = dict(idx=q["idx"], mode=mode, A0=g0["A0"], B0=g0["B0"], meta={})
    for me, other, slot in (("A", "B", "A1m"), ("B", "A", "B1m")):
        m = message(cx, mode, q, me, other, g0)
        if m is None:
            return dict(idx=q["idx"], mode=mode, skipped=True)
        pname, pev, extra, meta = m
        kw = dict(question=q["question"], evidence=fmt_passages(q[me]), own_reasoning=g0[f"{me}0"]["text"],
                  partner_evidence=fmt_passages(pev, start=len(q[me]) + 1) if pev else NONE, **extra)
        out[slot] = call(prompt(pname, **kw), 0.0, s0 + SLOT[slot])
        out["meta"][me] = dict(meta, sent_n=len(pev), sent_words=sum(wc(p) for p in pev))
    out["MAS"] = combine(q, out["A1m"], out["B1m"], s0 + SLOT["SYNm"])
    return out


def layout(cx, mode, q):
    i = q["idx"]
    if mode.startswith("E2H_"):
        c = cx.X[i]["e2h"]
        if not c["eligible"]:
            return None
        cell = c["cells"][mode[4:]]
        return dict(q, A=cell["A"], B=cell["B"])
    if mode.startswith("OVL"):
        o = int(mode[3:]); A, B = q["A"], q["B"]
        return dict(q, A=A[:5 - o // 2] + B[:o // 2], B=B[:5 - (o + 1) // 2] + A[:(o + 1) // 2])
    if mode == "DPAD":
        pad = cx.X[i]["pad"]
        if len(pad["A"]) < 5 or len(pad["B"]) < 5:
            return None
        return dict(q, A=q["A"] + [cx.pas(x) for x in pad["A"]], B=q["B"] + [cx.pas(x) for x in pad["B"]])
    if mode.startswith("SEED"):
        s, cond = mode[4:].split("_")
        S = cx.seed[int(s)][i]
        A = [cx.P[x] for x in S["A"]] if cx.ds == "frames" else S["A"]
        B = [cx.P[x] for x in S["B"]] if cx.ds == "frames" else S["B"]
        return dict(q, A=A, B=B) if cond == "D" else dict(q, A=A + B, B=list(A + B))
    raise ValueError(mode)


def run_layout(cx, mode, q):
    q = layout(cx, mode, q)
    if q is None:
        return None
    s0 = 7_000_000 + q["idx"] * 1000
    ev = {s: fmt_passages(q[s]) for s in "AB"}
    key = "pid" if cx.ds == "frames" else "title"
    out = dict(idx=q["idx"], mode=mode, ids=dict(A=[p[key] for p in q["A"]], B=[p[key] for p in q["B"]]),
               gold=dict(A=sum(cx.gold(q, p) for p in q["A"]), B=sum(cx.gold(q, p) for p in q["B"])))
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


def run_uni(cx, q, k):
    """POOL / SC slots are reused on purpose: for HotpotQA, A5∪B5 IS the pooled evidence, so UNI_ONE / UNI_SC are
    exactly the frozen POOL / SC calls (cache); for FRAMES (pooled = combined-index top-10) they are new calls."""
    s0 = 7_000_000 + q["idx"] * 1000
    U = q["A"] + q["B"]; ev = fmt_passages(U)
    out = dict(idx=q["idx"], mode="UNI")
    out["UNI_ONE"] = call(prompt("answer", question=q["question"], evidence=ev), 0.0, s0 + SLOT["POOL"])
    out["UNI_REV"] = call(prompt("revise_nocomm", question=q["question"], evidence=ev,
                                 own_reasoning=out["UNI_ONE"]["text"]), 0.0, s0 + 201)
    sm = [call(prompt("answer", question=q["question"], evidence=ev), 0.7, s0 + SLOT["SC"] + j) for j in range(k)]
    norm = [normalize(s["answer"]) for s in sm if s["answer"]]; c = Counter(norm)
    maj = next((s["answer"] for s in sm if s["answer"] and c[normalize(s["answer"])] == max(c.values())), None) if c else None
    out["UNI_SC"] = dict(answer=maj, k=k, samples=sm)
    return out


def run_agg(cx, q, g0, gev):
    s0 = 7_000_000 + q["idx"] * 1000
    out = dict(idx=q["idx"], mode="AGG")
    U = fmt_passages(q["A"] + q["B"])
    for n, (src, ra, rb) in enumerate((("MAS", g0, "m"), ("NOCOMM", g0, "n"), ("EV", gev, "m"))):
        a, b = ra[f"A1{rb}"], ra[f"B1{rb}"]
        s = call(prompt("synthesize", question=q["question"], answer_1=a["answer"], reasoning_1=a["text"],
                        answer_2=b["answer"], reasoning_2=b["text"]), 0.0, s0 + 300 + n)
        out[f"ALWAYS_{src}"] = dict(answer=s["answer"], synthesized=True, calls=[s])
        if a["answer"] and b["answer"] and normalize(a["answer"]) == normalize(b["answer"]):
            out[f"EVSYN_{src}"] = dict(answer=a["answer"], synthesized=False, calls=[])
        else:
            s = call(prompt("synthesize_ev", question=q["question"], answer_1=a["answer"], reasoning_1=a["text"],
                            answer_2=b["answer"], reasoning_2=b["text"], evidence=U), 0.0, s0 + 310 + n)
            out[f"EVSYN_{src}"] = dict(answer=s["answer"], synthesized=True, calls=[s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["frames", "hotpot"])
    ap.add_argument("--mode", required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cx = Ctx(a.dataset)
    if a.mode.startswith("SEED"):
        s = a.mode[4:].split("_")[0]
        cx.seed = {int(s): rd(f"data/rx/seed{s}_{a.dataset}.jsonl")}
    G0 = rd(FROZEN[a.dataset]); GEV = rd(E1EV[a.dataset]) if a.mode == "AGG" else None
    k = json.load(open(f"{ROOT}/data/K_FROZEN_{a.dataset}.json"))["k_star"] if a.mode == "UNI" else 0
    Q = sorted(cx.Q.values(), key=lambda q: q["idx"]); Q = Q if a.n < 0 else Q[:a.n]

    def job(q):
        try:
            if a.mode in REV:
                r = run_rev(cx, a.mode, q, G0[q["idx"]])
            elif a.mode == "UNI":
                r = run_uni(cx, q, k)
            elif a.mode == "AGG":
                r = run_agg(cx, q, G0[q["idx"]], GEV[q["idx"]])
            else:
                r = run_layout(cx, a.mode, q)
                r = r if r is not None else dict(idx=q["idx"], mode=a.mode, skipped=True)
            return dict(r, error=None)
        except Exception:
            return dict(idx=q["idx"], mode=a.mode, error=traceback.format_exc())
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, Q))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"RX {a.dataset} {a.mode}: n={len(rows)} skipped={sum(bool(r.get('skipped')) for r in rows)} "
          f"errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    main()
