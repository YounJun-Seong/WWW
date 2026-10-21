"""RX (WWW-final round) — every assignment the RX experiments use, built deterministically BEFORE any RX generation.
Gold-informed construction (like E2): gold passages/paragraphs and the gold answer string are read HERE, only to build
the interventions and to exclude answer-leaking distractors. Generation never reads gold.

Per dataset (FRAMES final 624 / HotpotQA final 1000), per question, written to data/rx/assign_<ds>.jsonl:
  gold flags          FRAMES: passage from a gold page (dataset wiki_links) — page-level, NOT fact-level;
                      HotpotQA: paragraph whose title is a supporting-fact title (the complete 2-paragraph support set)
  HN                  hard negatives: topical (BM25 on the question) passages that are NOT gold, NOT in the team's
                      A5 ∪ B5, and do NOT contain the normalised gold answer. FRAMES: pooled BM25 over the corpus;
                      HotpotQA: BM25 over every paragraph of the dev distractor set, other questions only.
  RN_from             random negatives: the team passages of the next question in idx order (cyclic), as E1 ANS_IRR
  G_inj               injection gold set: HotpotQA = the 2 supporting paragraphs; FRAMES = up to 2 gold-page passages
                      (E2 rule: pooled top-30 order, prefer two different gold pages)
  sup_sents           HotpotQA only: supporting sentences of each gold paragraph (for the sentence-level message)
  e2h                 HotpotQA E2 cells (same rule as build_e2.py; fillers = own 8 non-gold paragraphs, then HN)
  pad                 D_PAD padding: A gets HN[0:5], B gets HN[5:10] (exclusive, length-matched to S_UNION's 10)
Partition seeds (Exp 5), data/rx/seed<s>_<ds>.jsonl:
  FRAMES   page-level 50/50 partition with random.Random(s) (same rule as build_corpus.py), BM25 top-5 per side
  HotpotQA per-question 5/5 paragraph split with random.Random(s + idx) (same rule as hotpot_prep.py)
  python rx_build.py"""
import hashlib
import json
import os
import random
import re

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from common import ROOT, normalize
from run_webqa import load

tok = lambda s: re.findall(r"\w+", s.lower())


class BM25Okapi:
    """Vectorised port of rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25, same idf floor). Scores equal the
    library's up to floating-point summation order; equivalence is CHECKED in seeds_frames() by reproducing the frozen
    seed-2027 retrieval lists exactly before any new seed is written."""
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.vocab, rows, cols, tfs, dl, nd = {}, [], [], [], [], Counter()
        for d, doc in enumerate(corpus):
            c = Counter(doc); dl.append(len(doc))
            for w, f in c.items():
                j = self.vocab.setdefault(w, len(self.vocab)); rows.append(d); cols.append(j); tfs.append(f); nd[w] += 1
        N = len(corpus); avgdl = sum(dl) / N; dl = np.array(dl, float)
        idf = np.zeros(len(self.vocab)); neg = []
        for w, f in nd.items():
            v = math.log(N - f + 0.5) - math.log(f + 0.5); idf[self.vocab[w]] = v
            if v < 0:
                neg.append(self.vocab[w])
        idf[neg] = epsilon * idf.mean()
        self.idf = idf
        tf = np.array(tfs, float); r = np.array(rows)
        w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[r] / avgdl))
        self.M = sparse.csc_matrix((w, (r, np.array(cols))), shape=(N, len(self.vocab)))

    def get_scores(self, query):
        c = Counter(j for j in (self.vocab.get(t) for t in query) if j is not None)
        if not c:
            return np.zeros(self.M.shape[0])
        js = list(c); wq = np.array([self.idf[j] * c[j] for j in js])
        return np.asarray(self.M[:, js] @ wq).ravel()
OUT = f"{ROOT}/data/rx"
SEEDS = (2028, 2029, 2030)


def leaks(ans, text):
    a = normalize(ans)
    return len(a) >= 2 and f" {a} " in f" {normalize(text)} "


def frames():
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    Q = sorted(load("frames", "final"), key=lambda q: q["idx"])
    bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in P])
    rows = []
    for k, q in enumerate(Q):
        i = q["idx"]; ans = str(df.loc[i, "Answer"]); gold = set(R[i]["gold_pages"])
        sc = bm.get_scores(tok(q["question"]))
        top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:80]
        team = {p["pid"] for p in q["A"] + q["B"]}
        pool30 = list(dict.fromkeys([P[j]["pid"] for j in top[:30]] + R[i]["A"] + R[i]["B"]))
        gp = [pid for pid in pool30 if P[pid]["page"] in gold]
        G = []
        for pid in gp:
            if len(G) < 2 and (not G or P[pid]["page"] != P[G[0]]["page"]):
                G.append(pid)
        for pid in gp:
            if len(G) < 2 and pid not in G:
                G.append(pid)
        HN = [P[j]["pid"] for j in top if P[j]["page"] not in gold and P[j]["pid"] not in team
              and not leaks(ans, P[j]["text"])][:20]
        nxt = Q[(k + 1) % len(Q)]
        rows.append(dict(idx=i, gold_pids=sorted(p["pid"] for p in q["A"] + q["B"] if p["page"] in gold),
                         G_inj=G, HN=HN, RN_from=nxt["idx"],
                         pad=dict(A=HN[0:5], B=HN[5:10]), n_hn=len(HN),
                         answer_in_team=[p["pid"] for p in q["A"] + q["B"] if leaks(ans, p["text"])]))
    return rows


def hotpot():
    h = pd.read_parquet(f"{ROOT}/data/hotpot_dev_distractor.parquet")
    lab = json.load(open(f"{ROOT}/data/hotpot_final_labels.json"))
    F = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/hotpot_final.jsonl"))}
    corpus, seen = [], set()                              # every distinct dev paragraph, with its owning question row
    for i in range(len(h)):
        r = h.iloc[i]
        for t, s in zip(r["context"]["title"], r["context"]["sentences"]):
            key = (t, " ".join(s))
            if key not in seen:
                seen.add(key); corpus.append(dict(title=t, text=" ".join(s), row=i))
    bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in corpus])
    Q = sorted(F.values(), key=lambda q: q["idx"])
    rows = []
    for k, q in enumerate(Q):
        i = q["idx"]; r = h.iloc[i]; ans = lab[str(i)]; gold = set(q["gold_titles"])
        own = {(p["title"], p["text"]) for p in q["A"] + q["B"]}
        sc = bm.get_scores(tok(q["question"]))
        top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:200]
        HN = [dict(title=corpus[j]["title"], text=corpus[j]["text"]) for j in top
              if corpus[j]["row"] != i and corpus[j]["title"] not in gold
              and (corpus[j]["title"], corpus[j]["text"]) not in own and not leaks(ans, corpus[j]["text"])][:20]
        paras = {t: s for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
        sup = {}
        for t, sid in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
            if t in paras and int(sid) < len(paras[t]):
                sup.setdefault(t, []).append(int(sid))
        sup_sents = {t: " ".join(paras[t][j] for j in sorted(set(v))) for t, v in sup.items()}
        allp = q["A"] + q["B"]
        G = [p for p in allp if p["title"] in gold]
        nong = [p for p in allp if p["title"] not in gold]                 # in the question's original context order
        fill = nong + [p for p in HN if (p["title"], p["text"]) not in {(x["title"], x["text"]) for x in nong}]
        rec = dict(idx=i, gold_titles=sorted(gold), G_inj=G, HN=HN, RN_from=Q[(k + 1) % len(Q)]["idx"],
                   sup_sents=sup_sents, pad=dict(A=HN[0:5], B=HN[5:10]), n_hn=len(HN),
                   answer_in_team=[p["title"] for p in allp if leaks(ans, p["text"])])
        g = len(G)
        if g == 2 and len(fill) >= 13:
            S, F1, F2 = fill[0:3], fill[3:8], fill[8:13]
            hold = "A" if i % 2 == 0 else "B"; oth = "B" if hold == "A" else "A"
            cells = {"EXCL_LOW": {hold: G + F1[:5 - g], oth: F2[:5]},
                     "EXCL_HIGH": {hold: G + S + F1[:2 - g], oth: S + F2[:2]},
                     "SHAR_LOW": {"A": G + F1[:5 - g], "B": G + F2[:5 - g]},
                     "SHAR_HIGH": {"A": G + S + F1[:2 - g], "B": G + S + F2[:2 - g]}}
            for c in cells.values():
                for s in "AB":
                    assert len(c[s]) == 5
                    random.Random(2027 + i).shuffle(c[s])
            rec["e2h"] = dict(eligible=True, holder=hold, cells=cells)
        else:
            rec["e2h"] = dict(eligible=False, g=g, n_fill=len(fill))
        rows.append(rec)
    return rows, h


def seeds_frames():
    P = [json.loads(l) for l in open(f"{ROOT}/data/passages.jsonl")]
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    Q = sorted(load("frames", "final"), key=lambda q: q["idx"])
    titles = sorted(json.load(open(f"{ROOT}/data/partition.json"))["owner"])   # build_corpus.py's exact page list (2,515)
    out = {}
    for s in (2027,) + SEEDS:
        rng = random.Random(s); owner = {t: ("A" if rng.random() < 0.5 else "B") for t in titles}
        idx = {side: [p for p in P if owner[p["page"]] == side] for side in "AB"}
        bm = {side: BM25Okapi([tok(p["title"] + " " + p["text"]) for p in idx[side]]) for side in "AB"}
        rows = []
        for q in Q:
            gold = set(R[q["idx"]]["gold_pages"]); qt = tok(q["question"]); rec = dict(idx=q["idx"], seed=s)
            for side in "AB":
                sc = bm[side].get_scores(qt)
                top = sorted(range(len(sc)), key=lambda j: (-sc[j], j))[:5]
                rec[side] = [idx[side][j]["pid"] for j in top]
                rec[f"{side}_gold_retrieved"] = sorted({idx[side][j]["page"] for j in top} & gold)
            rec["gold_owner"] = {g: owner[g] for g in gold if g in owner}
            rows.append(rec)
        if s == 2027:      # must reproduce build_corpus.py's frozen lists exactly, or nothing is written
            bad = [r["idx"] for r in rows if r["A"] != R[r["idx"]]["A"] or r["B"] != R[r["idx"]]["B"]]
            assert not bad, f"FastBM25 does not reproduce the frozen seed-2027 retrieval: {len(bad)} questions, e.g. {bad[:5]}"
            print(f"  equivalence check: seed 2027 retrieval reproduced exactly for {len(rows)} questions")
            continue
        out[s] = (rows, sum(v == "A" for v in owner.values()))
    return out


def seeds_hotpot(h):
    F = sorted(map(json.loads, open(f"{ROOT}/data/hotpot_final.jsonl")), key=lambda q: q["idx"])
    out = {}
    for s in SEEDS:
        rows = []
        for q in F:
            r = h.iloc[q["idx"]]
            paras = [dict(title=t, text=" ".join(x)) for t, x in zip(r["context"]["title"], r["context"]["sentences"])]
            perm = list(range(len(paras))); random.Random(s + q["idx"]).shuffle(perm)
            a, b = sorted(perm[:5]), sorted(perm[5:])
            gold = set(q["gold_titles"])
            rows.append(dict(idx=q["idx"], seed=s, A=[paras[j] for j in a], B=[paras[j] for j in b],
                             A_gold=sorted({paras[j]["title"] for j in a} & gold),
                             B_gold=sorted({paras[j]["title"] for j in b} & gold)))
        out[s] = rows
    return out


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], os.path.relpath(path, ROOT), len(rows))


def main():
    os.makedirs(OUT, exist_ok=True)
    fr = frames(); dump(f"{OUT}/assign_frames.jsonl", fr)
    print(f"  FRAMES: G_inj sizes {sum(len(r['G_inj']) == 2 for r in fr)}x2 {sum(len(r['G_inj']) == 1 for r in fr)}x1 "
          f"{sum(not r['G_inj'] for r in fr)}x0; HN>=10 {sum(r['n_hn'] >= 10 for r in fr)}/{len(fr)}")
    hp, h = hotpot(); dump(f"{OUT}/assign_hotpot.jsonl", hp)
    print(f"  HotpotQA: E2 eligible {sum(r['e2h']['eligible'] for r in hp)}/{len(hp)}; HN>=10 {sum(r['n_hn'] >= 10 for r in hp)}; "
          f"sup_sents for both gold {sum(len(r['sup_sents']) == 2 for r in hp)}")
    for s, (rows, na) in seeds_frames().items():
        dump(f"{OUT}/seed{s}_frames.jsonl", rows); print(f"  seed {s}: A pages {na}")
    for s, rows in seeds_hotpot(h).items():
        dump(f"{OUT}/seed{s}_hotpot.jsonl", rows)


if __name__ == "__main__":
    main()
