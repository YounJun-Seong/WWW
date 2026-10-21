"""E3 — per-domain BM25 retrieval for the topic partition (same passages, tokenizer and k as build_corpus.py; only the
partition changes: PEOPLE vs WORLD from data/topic_partition.json instead of the random page partition). The pooled
index U is unchanged, so the Pooled and SC arms of the frozen FRAMES final run are reused.  ->  data/retrieval_topic.jsonl"""
import json
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from common import ROOT
from fetch_wiki import links_of

tok = lambda s: re.findall(r"\w+", s.lower())


def main():
    dom = json.load(open(f"{ROOT}/data/topic_partition.json"))["domain"]
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    idx = {s: [p for p in P if dom[p["page"]] == s] for s in ("PEOPLE", "WORLD")}
    bm = {s: BM25Okapi([tok(p["title"] + " " + p["text"]) for p in idx[s]]) for s in idx}
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    pages = set(dom)
    with open(f"{ROOT}/data/retrieval_topic.jsonl", "w") as f:
        for i, r in df.iterrows():
            gold = sorted(links_of(r["Wiki_links"] if "Wiki_links" in r else r["wiki_links"]) & pages)
            out = dict(idx=int(i), gold_pages=gold, gold_owner={g: ("A" if dom[g] == "PEOPLE" else "B") for g in gold})
            for s, key in (("PEOPLE", "A"), ("WORLD", "B")):
                sc = bm[s].get_scores(tok(r["Prompt"]))
                top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:5]
                out[key] = [idx[s][j]["pid"] for j in top]
                out[f"{key}_gold_retrieved"] = sorted({idx[s][j]["page"] for j in top} & set(gold))
            f.write(json.dumps(out) + "\n")
    print(f"passages PEOPLE {len(idx['PEOPLE'])}, WORLD {len(idx['WORLD'])}")


if __name__ == "__main__":
    main()
