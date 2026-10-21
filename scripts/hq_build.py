"""HQ builder (DESIGN_HQ.md): two SOURCE indexes from the HybridQA release and the frozen 800-question sample.
  Source T = one document per table row (all 15,316 tables); Source P = every distinct linked passage (by URL).
  data/hq/sample.jsonl   k, question_id, question, answer, table_id, answer_nodes, A (own top-5 rows), B (own top-5
                         passages), traced support (rows / passage URLs) and coverage flags (analysis only)
  data/hq/index.pkl      both BM25 indexes + documents (used at generation time by REMOTE_GEN)
  data/hq/MANIFEST.json  sha256 of the raw files, corpus sizes
Gold (answer text, answer nodes) is stored for analysis only; hq_run.py never passes it to a prompt.
  python hq_build.py"""
import ast
import glob
import hashlib
import json
import os
import pickle
import random

from common import ROOT
from rx_build import BM25Okapi, tok

RAW = f"{ROOT}/data/hybridqa_raw"; WT = glob.glob(f"{RAW}/WikiTables-WithLinks-*")[0]; OUT = f"{ROOT}/data/hq"
N, SEED, K = 800, 20261020, 5


def cell(v):
    return v[0] if isinstance(v, list) else str(v)


def build_corpora():
    rows, passages, seen = [], [], {}
    for f in sorted(glob.glob(f"{WT}/tables_tok/*.json")):
        tid = os.path.basename(f)[:-5]; t = json.load(open(f))
        hdr = [cell(h) for h in t["header"]]
        for r, row in enumerate(t["data"]):
            vals = [cell(c) for c in row]
            rows.append(dict(id=f"{tid}#r{r}", table_id=tid, row=r,
                             title=f"{t['title']} — {t.get('section_title') or ''} (table row {r + 1})",
                             text="; ".join(f"{h}: {v}" for h, v in zip(hdr, vals) if str(v).strip())))
        req = json.load(open(f"{WT}/request_tok/{tid}.json"))
        for url, txt in req.items():
            if url not in seen:
                seen[url] = len(passages)
                passages.append(dict(id=url, title=url.split("/wiki/")[-1].replace("_", " "), text=txt))
    return rows, passages


def top(bm, docs, query, k=K):
    sc = bm.get_scores(tok(query))
    import numpy as np
    idx = np.argpartition(-sc, k)[:k] if len(sc) > k else range(len(sc))
    return [docs[j] for j in sorted(idx, key=lambda j: (-sc[j], j))][:k]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, passages = build_corpora()
    print(f"Source T rows {len(rows)}, Source P passages {len(passages)}")
    bmT = BM25Okapi([tok(d["title"] + " " + d["text"]) for d in rows])
    bmP = BM25Okapi([tok(d["title"] + " " + d["text"]) for d in passages])
    pickle.dump(dict(rows=rows, passages=passages, bmT=bmT, bmP=bmP), open(f"{OUT}/index.pkl", "wb"))
    dev = json.load(open(f"{RAW}/dev.traced.json"))
    pick = random.Random(SEED).sample(range(len(dev)), N)
    with open(f"{OUT}/sample.jsonl", "w") as fo:
        for k, j in enumerate(pick):
            q = dev[j]; nodes = ast.literal_eval(q["answer-node"]) if isinstance(q["answer-node"], str) else q["answer-node"]
            A, B = top(bmT, rows, q["question"]), top(bmP, passages, q["question"])
            sup_rows = sorted({f"{q['table_id']}#r{n[1][0]}" for n in nodes})
            sup_pass = sorted({n[2] for n in nodes if n[3] == "passage" and n[2]})
            fo.write(json.dumps(dict(k=k, question_id=q["question_id"], question=q["question"], answer=q["answer-text"],
                                     table_id=q["table_id"], answer_nodes=nodes, A=A, B=B, sup_rows=sup_rows, sup_passages=sup_pass,
                                     A_has_sup=any(d["id"] in sup_rows for d in A), B_has_sup=any(d["id"] in sup_pass for d in B))) + "\n")
    S = [json.loads(l) for l in open(f"{OUT}/sample.jsonl")]
    man = dict(raw={os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()
                    for f in (f"{RAW}/WikiTables.zip", f"{RAW}/dev.traced.json")},
               n_rows=len(rows), n_passages=len(passages), n_sample=len(S), seed=SEED,
               coverage=dict(A_has_traced_row=sum(s["A_has_sup"] for s in S), B_has_traced_passage=sum(s["B_has_sup"] for s in S),
                             passage_type_answers=sum(bool(s["sup_passages"]) for s in S)))
    json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1); print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
