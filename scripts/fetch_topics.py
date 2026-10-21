"""E3 — topic (domain) partition of the frozen FRAMES corpus. For every fetched page (manifest, 2,515), the Wikidata item
(pageprops.wikibase_item, batches of 50) and its P31 "instance of" values (wbgetentities, batches of 50) are fetched
serially and politely. Domain: "PEOPLE" if P31 contains Q5 (human), else "WORLD". Pages without an item -> WORLD.
These are current Wikidata values (not revision-pinned); they define the partition only, never the evidence text.
  python fetch_topics.py  ->  data/topic_partition.json"""
import json
import time

import requests

from common import ROOT
from fetch_wiki import UA, get

WD = "https://www.wikidata.org/w/api.php"


def wd_get(params, tries=10):
    for i in range(tries):
        try:
            r = requests.get(WD, params={**params, "format": "json", "maxlag": 5}, headers=UA, timeout=120)
            if r.status_code == 200 and "error" not in r.json():
                return r.json()
            wait = float(r.headers.get("Retry-After", 0) or 0)
        except (requests.RequestException, ValueError):
            wait = 0
        time.sleep(min(600, max(wait, 5 * 2 ** i)))
    raise RuntimeError("wikidata failed")


def main():
    M = [m for m in map(json.loads, open(f"{ROOT}/data/wiki_manifest.jsonl")) if not m["missing"]]
    name = {m["title"]: (m["resolved"] or m["title"]) for m in M}
    qid = {}
    names = sorted(set(name.values()))
    for k in range(0, len(names), 50):
        res = get(dict(action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(names[k:k + 50]), redirects=1))
        norm = {x["from"]: x["to"] for x in res["query"].get("normalized", []) + res["query"].get("redirects", [])}
        got = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in res["query"]["pages"]}
        for t in names[k:k + 50]:
            u = t
            for _ in range(3):
                u = norm.get(u, u)
            qid[t] = got.get(u)
        time.sleep(1.0)
    ids = sorted({q for q in qid.values() if q})
    p31 = {}
    for k in range(0, len(ids), 50):
        res = wd_get(dict(action="wbgetentities", ids="|".join(ids[k:k + 50]), props="claims"))
        for q, ent in res.get("entities", {}).items():
            p31[q] = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in ent.get("claims", {}).get("P31", [])]
        time.sleep(1.0)
    dom = {t: ("PEOPLE" if qid.get(name[t]) and "Q5" in p31.get(qid[name[t]], []) else "WORLD") for t in name}
    json.dump(dict(rule="P31 contains Q5 -> PEOPLE else WORLD", qid={t: qid.get(name[t]) for t in name}, domain=dom),
              open(f"{ROOT}/data/topic_partition.json", "w"), indent=0)
    print(f"pages {len(dom)}: PEOPLE {sum(v == 'PEOPLE' for v in dom.values())}, WORLD {sum(v == 'WORLD' for v in dom.values())}, "
          f"no item {sum(not qid.get(name[t]) for t in name)}")


if __name__ == "__main__":
    main()
