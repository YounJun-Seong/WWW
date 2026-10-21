"""Gate checks before any FRAMES generation (stage = snapshot | index).

  snapshot  the manifest lists exactly the titles linked by the 824 FRAMES questions (links_of); every page that is not
            missing has an integer revid and a timestamp <= CUTOFF, and its cached file has the same revid; no page failed
            with a fetch error. A page the API reports as missing at the cutoff, or a page with no body text, is allowed
            and is counted.
  index     the retrieval file covers all 824 FRAMES questions with A = 5, B = 5 and U = 10 passages; every passage
            id exists; A and B passages come only from their own source; U passages come from A ∪ B
Exit code 1 on any violation.  ->  data/SNAPSHOT_VERIFY_<stage>.txt
"""
import hashlib
import json
import sys

from common import ROOT
import pandas as pd

from fetch_wiki import CACHE, CUTOFF, links_of


def snapshot():
    M = [json.loads(l) for l in open(f"{ROOT}/data/wiki_manifest.jsonl")]
    want = set().union(*(links_of(s) for s in pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")["wiki_links"]))
    have = [r["title"] for r in M]
    complete = set(have) == want and len(have) == len(want)
    bad, empty, miss = [], [], []
    err = [r["title"] for r in M if r.get("error")]
    for r in M:
        if r["missing"]:
            miss.append(r["title"]); continue
        c = json.load(open(f"{CACHE}/{hashlib.sha1(r['title'].encode()).hexdigest()}.json"))
        if not (isinstance(r["revid"], int) and r["timestamp"] <= CUTOFF and c.get("revid") == r["revid"]):
            bad.append(r["title"])
        elif not c.get("blocks"):
            empty.append(r["title"])
    L = [f"titles linked {len(want)}; manifest {len(have)}; title set complete {complete}; fetched {len(M) - len(miss)}; "
         f"missing at cutoff {len(miss) - len(err)}; fetch errors {len(err)}; revid/timestamp/cache violations {len(bad)}; "
         f"fetched but no body text {len(empty)}; cutoff {CUTOFF}",
         f"latest revision timestamp {max(r['timestamp'] for r in M if not r['missing'])}",
         "missing: " + json.dumps(sorted(set(miss) - set(err))), "errors: " + json.dumps(err),
         "violations: " + json.dumps(bad), "no body text: " + json.dumps(empty)]
    return L, complete and not (err or bad)


def index():
    P = {p["pid"]: p["source"] for p in map(json.loads, open(f"{ROOT}/data/passages.jsonl"))}
    R = [json.loads(l) for l in open(f"{ROOT}/data/retrieval_frames.jsonl")]
    bad = [r["idx"] for r in R if not (len(r["A"]) == 5 and len(r["B"]) == 5 and len(r["U"]) == 10
                                       and all(P.get(j) == "A" for j in r["A"]) and all(P.get(j) == "B" for j in r["B"])
                                       and all(j in P for j in r["U"]))]
    nog = sum(not r["gold_pages"] for r in R)
    L = [f"questions {len(R)} (expected 824); passages {len(P)}; violations {len(bad)}; questions with no gold page in snapshot {nog}",
         "violations: " + json.dumps(bad)]
    return L, len(R) == 824 and not bad


if __name__ == "__main__":
    stage = sys.argv[1]
    L, ok = (snapshot if stage == "snapshot" else index)()
    L.append("PASS" if ok else "FAIL")
    open(f"{ROOT}/data/SNAPSHOT_VERIFY_{stage}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)
