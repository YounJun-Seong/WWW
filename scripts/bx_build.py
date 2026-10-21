"""BX builder (DESIGN_BX.md). Gold is read ONLY here, to build the B3 cells; generation never reads gold.
  data/bx/b3_hotpot.jsonl     B3 support-recall x placement cells (HotpotQA final)
  data/bx/rank_hotpot.jsonl   positions of each agent's 5 paragraphs in BM25(question) order (the order a peer / a remote index
                              returns them; FRAMES lists are already in BM25 rank order)
Also provides the remote indexes used at generation time (bx_run.py imports them):
  frames_remote()   BM25 per side of the seed-2027 page partition; asserts that top-5 with the question reproduces the
                    frozen retrieval lists for every final question before returning
  hotpot_remote()   BM25 over every distinct dev-distractor paragraph (as rx_build.hotpot), with a (title, text) lookup
  python bx_build.py"""
import json
import os
import random

import pandas as pd

from common import ROOT
from run_webqa import load
from rx_build import BM25Okapi, tok

OUT = f"{ROOT}/data/bx"


def frames_remote():
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    owner = json.load(open(f"{ROOT}/data/partition.json"))["owner"]
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    idx = {s: [p for p in P if owner[p["page"]] == s] for s in "AB"}
    bm = {s: BM25Okapi([tok(p["title"] + " " + p["text"]) for p in idx[s]]) for s in "AB"}

    def search(side, query, k):
        sc = bm[side].get_scores(tok(query))
        return [idx[side][j] for j in sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:k]]
    bad = [q["idx"] for q in load("frames", "final")
           if [p["pid"] for p in search("A", q["question"], 5)] != R[q["idx"]]["A"]
           or [p["pid"] for p in search("B", q["question"], 5)] != R[q["idx"]]["B"]]
    assert not bad, f"remote index does not reproduce the frozen FRAMES retrieval for {len(bad)} questions: {bad[:5]}"
    return search


def hotpot_remote():
    h = pd.read_parquet(f"{ROOT}/data/hotpot_dev_distractor.parquet")
    corpus, seen = [], {}
    for i in range(len(h)):
        r = h.iloc[i]
        for t, s in zip(r["context"]["title"], r["context"]["sentences"]):
            key = (t, " ".join(s))
            if key not in seen:
                seen[key] = len(corpus); corpus.append(dict(title=t, text=" ".join(s)))
    bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in corpus])

    def order(paras, query):
        """positions of a closed remote collection (the partner's paragraphs), ranked by BM25 fitted on the dev corpus."""
        sc = bm.get_scores(tok(query))
        s = [sc[seen[(p["title"], p["text"])]] for p in paras]
        return sorted(range(len(paras)), key=lambda j: (-s[j], j))

    def search(paras, query, k):
        return [paras[j] for j in order(paras, query)[:k]]
    search.order = order
    return search


def b3(rx):
    F = sorted(map(json.loads, open(f"{ROOT}/data/hotpot_final.jsonl")), key=lambda q: q["idx"])
    rows = []
    for q in F:
        i = q["idx"]; gold = set(q["gold_titles"]); allp = q["A"] + q["B"]
        G = [p for p in allp if p["title"] in gold]
        nong = [p for p in allp if p["title"] not in gold]
        keys = {(p["title"], p["text"]) for p in nong}
        fill = nong + [p for p in rx[i]["HN"] if (p["title"], p["text"]) not in keys]
        if len(G) != 2 or len(fill) < 10:
            rows.append(dict(idx=i, eligible=False, g=len(G), n_fill=len(fill))); continue
        g1, g2 = G
        H = "A" if i % 2 == 0 else "B"; O = "B" if H == "A" else "A"
        f = fill
        cells = {"NONE": {H: f[0:5], O: f[5:10]},
                 "P_EXCL": {H: [g1] + f[0:4], O: f[4:9]},
                 "P_SHAR": {H: [g1] + f[0:4], O: [g1] + f[4:8]},
                 "F_SPLIT": {H: [g1] + f[0:4], O: [g2] + f[4:8]},
                 "F_CONC": {H: [g1, g2] + f[0:3], O: f[3:8]},
                 "F_SHAR": {H: [g1, g2] + f[0:3], O: [g1, g2] + f[3:6]}}
        for c in cells.values():
            for s in "AB":
                assert len(c[s]) == 5
                random.Random(2027 + i).shuffle(c[s])
            sh = {(p["title"], p["text"]) for p in c["A"]} & {(p["title"], p["text"]) for p in c["B"]}
            assert not (sh - {(g["title"], g["text"]) for g in G}), "a filler is shared between agents"
        rows.append(dict(idx=i, eligible=True, holder=H, cells=cells))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rx = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/rx/assign_hotpot.jsonl"))}
    rows = b3(rx)
    with open(f"{OUT}/b3_hotpot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"B3 cells: eligible {sum(r['eligible'] for r in rows)}/{len(rows)}")
    search = hotpot_remote()
    with open(f"{OUT}/rank_hotpot.jsonl", "w") as f:
        for q in load("hotpot", "final"):
            f.write(json.dumps(dict(idx=q["idx"], **{s: search.order(q[s], q["question"]) for s in "AB"})) + "\n")
    print("HotpotQA partner rankings written")
    frames_remote(); print("FRAMES remote index: seed-2027 retrieval reproduced exactly for all final questions")


if __name__ == "__main__":
    main()
