"""BX2-C latency and cost audit (DESIGN_BX2.md §C), FRAMES, 100 pre-drawn questions. Each system is executed with its
real parallel structure (the two agents' calls of one round run concurrently); every LLM call and every BM25 search is
timed. Calls are UNCACHED (uchat): the systems of one question share payloads, so a cache — even a fresh one per run —
would turn repeated calls into hits (found by the smoke run on 10-18, before any measurement was used).
Payloads are identical to the frozen runs, so each output is also compared with the frozen cached output.
  python bx2_latency.py --cond isolated|loaded --rep R --part 0|1|all --port P --out runs/bx2/lat_<cond>_r<R>_<part>.jsonl
(WQ_CACHE must point at a fresh, empty directory; WQ_BASE_URL / WQ_MODEL at a dedicated server.)"""
import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, answer_of, chat, fmt_passages, normalize, prompt
from run_webqa import MAXTOK, SLOT, load
from bx_build import frames_remote
from bx_run import QSLOT, parse_query
from rx_build import BM25Okapi, tok

SYSTEMS = ["LOCAL_ONE", "PEER_EV_5", "FULL", "CQ_Q_5", "CQ_GEN_5", "UNI_ONE", "POOL"]
SERIAL_RETRIEVAL = dict(LOCAL_ONE=1, PEER_EV_5=1, FULL=1, CQ_Q_5=2, CQ_GEN_5=2, UNI_ONE=1, POOL=1)
FROZEN_CACHE = f"{ROOT}/data/llm_cache"


def sample_ids():
    ids = sorted(json.load(open(f"{ROOT}/data/frames_split.json"))["final"])
    return random.Random(20261018).sample(ids, 100)


class Timer:
    def __init__(self):
        self.calls, self.retr, self.lock = [], [], threading.Lock()


_client = {}


def uchat(msgs, temp, seed):
    """UNCACHED request with exactly common.chat's payload (several systems of one question share payloads, so any
    cache — even a fresh per-run one — would turn the second system's identical call into a hit)."""
    from openai import OpenAI
    if "c" not in _client:
        _client["c"] = OpenAI(base_url=os.environ["WQ_BASE_URL"], api_key="EMPTY", timeout=900)
    r = _client["c"].chat.completions.create(model=os.environ["WQ_MODEL"], messages=msgs, temperature=temp,
                                             max_tokens=MAXTOK, seed=seed, n=1)
    return dict(texts=[r.choices[0].message.content], input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens)


def tcall(T, stage, msgs, temp, seed):
    payload = dict(model=os.environ["WQ_MODEL"], messages=msgs, temperature=temp, max_tokens=MAXTOK, seed=seed, n=1, extra={})
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
    hit = False                                               # uncached client: every call reaches the server
    t0 = time.perf_counter(); c = uchat(msgs, temp, seed); dt = time.perf_counter() - t0
    fz = f"{FROZEN_CACHE}/{key}.json"; frozen = json.load(open(fz)) if os.path.exists(fz) else None
    t = c["texts"][0]
    rec = dict(stage=stage, wall=dt, hit=hit, input_tokens=c["input_tokens"], output_tokens=c["output_tokens"],
               frozen_exists=frozen is not None,
               same_text=(frozen["texts"][0] == t) if frozen else None,
               same_answer=(normalize(answer_of(frozen["texts"][0]) or "") == normalize(answer_of(t) or "")) if frozen else None)
    with T.lock:
        T.calls.append(rec)
    return dict(text=t, answer=answer_of(t))


def tsearch(T, stage, fn, *a):
    t0 = time.perf_counter(); r = fn(*a); dt = time.perf_counter() - t0
    with T.lock:
        T.retr.append(dict(stage=stage, wall=dt))
    return r


def par(*fns):
    with ThreadPoolExecutor(len(fns)) as ex:
        return [f.result() for f in [ex.submit(fn) for fn in fns]]


def combine_t(T, q, a, b, seed):
    if a["answer"] and b["answer"] and normalize(a["answer"]) == normalize(b["answer"]):
        return a["answer"]
    return tcall(T, "synth", prompt("synthesize", question=q["question"], answer_1=a["answer"], reasoning_1=a["text"],
                                    answer_2=b["answer"], reasoning_2=b["text"]), 0.0, seed)["answer"]


def run_system(S, q, R):
    """returns (answer, Timer, e2e). R = retrieval helpers."""
    T = Timer(); s0 = 7_000_000 + q["idx"] * 1000; qs = q["question"]; t0 = time.perf_counter()

    def own(side):
        ps = tsearch(T, f"retrieve_{side}", R["part"], side, qs, 5)
        return ps, tcall(T, f"{side}0", prompt("answer", question=qs, evidence=fmt_passages(ps)), 0.0, s0 + SLOT[f"{side}0"])
    if S == "LOCAL_ONE":
        ans = own("A")[1]["answer"]
    elif S in ("PEER_EV_5", "FULL"):
        (pa, a0), (pb, b0) = par(lambda: own("A"), lambda: own("B"))
        P, G = dict(A=pa, B=pb), dict(A=a0, B=b0)

        def rev(me, other, slot):
            if S == "PEER_EV_5":
                m = prompt("revise_e1_ev", question=qs, evidence=fmt_passages(P[me]), own_reasoning=G[me]["text"],
                           partner_evidence=fmt_passages(P[other], start=6))
            else:
                m = prompt("revise_mas", question=qs, evidence=fmt_passages(P[me]), own_reasoning=G[me]["text"],
                           partner_answer=G[other]["answer"], partner_reasoning=G[other]["text"], partner_evidence=fmt_passages(P[other], start=6))
            return tcall(T, slot, m, 0.0, s0 + SLOT[slot])
        a1, b1 = par(lambda: rev("A", "B", "A1m"), lambda: rev("B", "A", "B1m"))
        ans = combine_t(T, q, a1, b1, s0 + SLOT["SYNm"])
    elif S in ("CQ_Q_5", "CQ_GEN_5"):
        pa, a0 = own("A"); query = qs
        if S == "CQ_GEN_5":
            c = tcall(T, "qgen", prompt("query", question=qs, evidence=fmt_passages(pa), own_reasoning=a0["text"]), 0.0, s0 + QSLOT["A"])
            query, _ = parse_query(c["text"], qs)
        rem = tsearch(T, "retrieve_remote", R["part"], "B", query, 5)
        ans = tcall(T, "A1m", prompt("revise_retrieved", question=qs, evidence=fmt_passages(pa), own_reasoning=a0["text"],
                                     partner_evidence=fmt_passages(rem, start=6)), 0.0, s0 + SLOT["A1m"])["answer"]
    elif S == "UNI_ONE":
        pa, pb = par(lambda: tsearch(T, "retrieve_A", R["part"], "A", qs, 5), lambda: tsearch(T, "retrieve_B", R["part"], "B", qs, 5))
        ans = tcall(T, "UNI_ONE", prompt("answer", question=qs, evidence=fmt_passages(pa + pb)), 0.0, s0 + SLOT["POOL"])["answer"]
    elif S == "POOL":
        U = tsearch(T, "retrieve_central", R["central"], qs, 10)
        ans = tcall(T, "POOL", prompt("answer", question=qs, evidence=fmt_passages(U)), 0.0, s0 + SLOT["POOL"])["answer"]
    return ans, T, time.perf_counter() - t0


def helpers():
    part = frames_remote()                                    # asserts seed-2027 reproduction
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in P])
    Rf = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}

    def central(query, k):
        sc = bm.get_scores(tok(query))
        return [P[j] for j in sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:k]]
    return dict(part=part, central=central, Rf=Rf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["isolated", "loaded"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--part", default="all")
    ap.add_argument("--inflight", type=int, default=16)
    ap.add_argument("--nq", type=int, default=-1, help="smoke: only the first N questions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q = {q["idx"]: q for q in load("frames", "final")}
    R = helpers()
    bad = [i for i in sample_ids() if [p["pid"] for p in R["central"](Q[i]["question"], 10)] != R["Rf"][i]["U"]]
    assert not bad, f"combined index does not reproduce the frozen pooled lists: {bad[:5]}"
    ids = sample_ids(); random.Random(a.rep).shuffle(ids)
    if a.part != "all":
        ids = ids[:50] if a.part == "0" else ids[50:]
    ids = ids if a.nq < 0 else ids[:a.nq]
    inflight = [0]; lk = threading.Lock(); rows = []

    def do_q(i):
        order = list(SYSTEMS); random.Random(a.rep * 100000 + i).shuffle(order)
        out = []
        for S in order:
            with lk:
                inflight[0] += 1; n_in = inflight[0]
            ans, T, e2e = run_system(S, Q[i], R)
            with lk:
                inflight[0] -= 1
            out.append(dict(idx=i, system=S, cond=a.cond, rep=a.rep, part=a.part, gpu=os.environ.get("BX2_GPU"),
                            inflight_at_start=n_in, answer=ans, e2e=e2e, calls=T.calls, retr=T.retr,
                            serial_retrieval_rounds=SERIAL_RETRIEVAL[S]))
        return out
    with ThreadPoolExecutor(1 if a.cond == "isolated" else a.inflight) as ex:
        for out in ex.map(do_q, ids):
            rows += out
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    hits = sum(c["hit"] for r in rows for c in r["calls"])
    print(f"BX2 latency {a.cond} rep {a.rep} part {a.part}: {len(rows)} system runs, cache hits {hits}")


if __name__ == "__main__":
    main()
