"""E2 — overlap × gold-placement factorial: evidence assignments (deterministic; written before any E2 generation).
Candidate pool per FRAMES final question = pooled BM25 top-30 (same index and tokenizer as build_corpus.py), then the
frozen A and B top-5 lists. Gold passages G = passages from the question's gold pages (dataset wiki_links, not the
answer): at most 2, preferring two different gold pages, in pool order. Fillers F = passages from non-gold pages, in pool
order: S = F[0:3] (shared filler), F1 = F[3:8], F2 = F[8:13]. Eligible: |G| >= 1 and |F| >= 13. Each agent holds 5:
  EXCL_LOW   holder = G + F1[:5-g]          other  = F2[:5]                  (no shared passage)
  EXCL_HIGH  holder = G + S + F1[:2-g]      other  = S + F2[:2]              (3 shared fillers)
  SHAR_LOW   A = G + F1[:5-g]               B = G + F2[:5-g]                 (only gold shared)
  SHAR_HIGH  A = G + S + F1[:2-g]           B = G + S + F2[:2-g]             (gold + 3 fillers shared)
The team's gold content is identical in all four cells. The exclusive holder alternates (A if idx even, else B). Each
list is shuffled with random.Random(2027 + idx) so the gold position is not fixed.  ->  data/e2_assign.jsonl"""
import json
import random
import re

from rank_bm25 import BM25Okapi

from common import ROOT
import pandas as pd

tok = lambda s: re.findall(r"\w+", s.lower())


def main():
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    final = sorted(json.load(open(f"{ROOT}/data/frames_split.json"))["final"])
    bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in P])
    n_ok = 0
    with open(f"{ROOT}/data/e2_assign.jsonl", "w") as f:
        for i in final:
            sc = bm.get_scores(tok(df.loc[i, "Prompt"]))
            top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:30]
            pool = list(dict.fromkeys([P[j]["pid"] for j in top] + R[i]["A"] + R[i]["B"]))
            page = {p["pid"]: p["page"] for p in (P[j] for j in pool)} if False else {pid: P[pid]["page"] for pid in pool}
            gold = set(R[i]["gold_pages"])
            gp = [pid for pid in pool if page[pid] in gold]
            G = []
            for pid in gp:
                if len(G) < 2 and (not G or page[pid] != page[G[0]]):
                    G.append(pid)
            for pid in gp:
                if len(G) < 2 and pid not in G:
                    G.append(pid)
            F = [pid for pid in pool if page[pid] not in gold]
            rec = dict(idx=i, eligible=bool(G) and len(F) >= 13, n_gold_pool=len(gp), G=G)
            if rec["eligible"]:
                n_ok += 1; g = len(G); S, F1, F2 = F[0:3], F[3:8], F[8:13]
                hold = "A" if i % 2 == 0 else "B"; oth = "B" if hold == "A" else "A"
                cells = {"EXCL_LOW": {hold: G + F1[:5 - g], oth: F2[:5]},
                         "EXCL_HIGH": {hold: G + S + F1[:2 - g], oth: S + F2[:2]},
                         "SHAR_LOW": {"A": G + F1[:5 - g], "B": G + F2[:5 - g]},
                         "SHAR_HIGH": {"A": G + S + F1[:2 - g], "B": G + S + F2[:2 - g]}}
                for c in cells.values():
                    for s in "AB":
                        assert len(c[s]) == 5
                        random.Random(2027 + i).shuffle(c[s])
                rec.update(holder=hold, cells=cells)
            f.write(json.dumps(rec) + "\n")
    print(f"E2 eligible questions: {n_ok} / {len(final)}")


if __name__ == "__main__":
    main()
