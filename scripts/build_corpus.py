"""Passages (~120 words, title-prefixed) from the frozen Wikipedia snapshot; global page-level 50/50 A/B partition
(seed 2027); BM25 top-k retrieval per FRAMES question for A (k=5), B (k=5) and pooled A∪B (k=10). Deterministic.

  python scripts/build_corpus.py  ->  data/passages.jsonl, data/partition.json, data/retrieval_frames.jsonl
"""
import ast
import glob
import hashlib
import json
import os
import random
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from common import ROOT
from fetch_wiki import CACHE, links_of

WORDS, SEED = 120, 2027
tok = lambda s: re.findall(r"\w+", s.lower())


def main():
    pages = {}                                   # exactly the manifest's fetched pages (amendment W1)
    for m in map(json.loads, open(f"{ROOT}/data/wiki_manifest.jsonl")):
        if not m["missing"]:
            pages[m["title"]] = json.load(open(f"{CACHE}/{hashlib.sha1(m['title'].encode()).hexdigest()}.json"))
    titles = sorted(pages)
    rng = random.Random(SEED)
    owner = {t: ("A" if rng.random() < 0.5 else "B") for t in titles}
    passages = []
    for t in titles:
        words = " ".join(pages[t]["blocks"]).split()
        for j in range(0, len(words), WORDS):
            passages.append(dict(pid=len(passages), page=t, title=pages[t].get("resolved") or t,
                                 text=" ".join(words[j:j + WORDS]), source=owner[t]))
    with open(f"{ROOT}/data/passages.jsonl", "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    json.dump(dict(seed=SEED, owner=owner), open(f"{ROOT}/data/partition.json", "w"))
    idx = {s: [p for p in passages if p["source"] == s] for s in "AB"}
    bm = {s: BM25Okapi([tok(p["title"] + " " + p["text"]) for p in idx[s]]) for s in "AB"}
    bm["U"] = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in passages]); idx["U"] = passages
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    with open(f"{ROOT}/data/retrieval_frames.jsonl", "w") as f:
        for i, r in df.iterrows():
            gold = sorted(links_of(r["wiki_links"]) & set(pages))
            q = tok(r["Prompt"]); out = dict(idx=int(i), gold_pages=gold, gold_owner={g: owner[g] for g in gold})
            for s, k in (("A", 5), ("B", 5), ("U", 10)):
                sc = bm[s].get_scores(q)
                top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:k]
                out[s] = [idx[s][j]["pid"] for j in top]
                out[f"{s}_gold_retrieved"] = sorted({idx[s][j]["page"] for j in top} & set(gold))
            f.write(json.dumps(out) + "\n")
    print(f"pages {len(titles)} (A {sum(v == 'A' for v in owner.values())}), passages {len(passages)}")
    for p in ("passages.jsonl", "partition.json", "retrieval_frames.jsonl"):
        print(hashlib.sha256(open(f'{ROOT}/data/{p}', 'rb').read()).hexdigest()[:16], p)


if __name__ == "__main__":
    main()
