"""Fetch the Wikipedia pages linked by FRAMES at a fixed snapshot time (last revision at or before CUTOFF), render them
to plain text, and cache one JSON per page. Polite: few workers, retries with backoff, generic research User-Agent.

  PYTHONPATH=.pylib python scripts/fetch_wiki.py   ->  data/wiki_cache/<sha1(title)>.json, data/wiki_manifest.jsonl
"""
import ast
import hashlib
import json
import re
import os
import time
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup

CUTOFF = "2024-09-19T23:59:59Z"          # FRAMES public release; the evidence snapshot is frozen at this time
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "webqa-research/0.1 (academic, low-rate)"}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = f"{ROOT}/data/wiki_cache"


def title_of(url):
    t = url.split("/wiki/", 1)[1].split("#", 1)[0]
    while "%" in t and urllib.parse.unquote(t) != t:        # some FRAMES links are percent-encoded twice
        t = urllib.parse.unquote(t)
    return t.replace("_", " ").strip()


def links_of(cell):
    """Titles in one FRAMES wiki_links cell. Some list items hold several URLs joined by ', ' (amendment W1)."""
    return {title_of(u) for item in ast.literal_eval(cell) for u in re.split(r",\s*(?=https?://)", item) if "/wiki/" in u}


def get(params, tries=10):
    """Serial, polite: on 429/5xx honour Retry-After (default exponential backoff, capped at 10 min)."""
    for i in range(tries):
        try:
            r = requests.get(API, params={**params, "format": "json", "formatversion": 2, "maxlag": 5}, headers=UA, timeout=120)
            if r.status_code == 200 and "error" not in r.json():
                return r.json()
            wait = float(r.headers.get("Retry-After", 0) or 0)
        except (requests.RequestException, ValueError):
            wait = 0
        time.sleep(min(600, max(wait, 5 * 2 ** i)))
    raise RuntimeError(f"failed: {params}")


def text_of(html):
    soup = BeautifulSoup(html, "lxml")
    for bad in soup.select("style, script, sup.reference, .mw-editsection, .navbox, .reflist, .mw-references-wrap, .metadata"):
        bad.decompose()
    blocks = []
    for el in soup.find_all(["h2", "h3", "h4", "p", "li", "tr", "dd"]):
        t = " ".join(el.get_text(" ", strip=True).split())
        if t:
            blocks.append(t)
    return blocks


def fetch(title):
    path = f"{CACHE}/{hashlib.sha1(title.encode()).hexdigest()}.json"
    if os.path.exists(path):
        return json.load(open(path))
    q = get(dict(action="query", prop="revisions", titles=title, redirects=1, rvlimit=1, rvstart=CUTOFF, rvdir="older",
                 rvprop="ids|timestamp"))
    page = q["query"]["pages"][0]
    rec = dict(title=title, resolved=page.get("title"), missing=("revisions" not in page))
    if not rec["missing"]:
        rev = page["revisions"][0]
        p = get(dict(action="parse", oldid=rev["revid"], prop="text", disablelimitreport=1))
        rec.update(revid=rev["revid"], timestamp=rev["timestamp"], blocks=text_of(p["parse"]["text"]))
    json.dump(rec, open(path, "w"))
    time.sleep(1.0)
    return rec


def main():
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    split = json.load(open(f"{ROOT}/data/frames_split.json"))
    links = lambda rows: set().union(*(links_of(s) for s in rows))
    pilot_titles = sorted(links(df.loc[split["pilot"], "wiki_links"]))
    titles = pilot_titles + sorted(links(df["wiki_links"]) - set(pilot_titles))   # pilot pages first
    print(f"{len(titles)} unique pages", flush=True)
    recs = []
    for i, t in enumerate(titles):            # one request at a time
        try:
            recs.append(fetch(t))
        except RuntimeError as e:
            recs.append(dict(title=t, missing=True, error=str(e)))
        if i % 100 == 0:
            print(f"{i}/{len(titles)}", flush=True)
    with open(f"{ROOT}/data/wiki_manifest.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: r.get(k) for k in ("title", "resolved", "missing", "revid", "timestamp", "error")}
                               | {"n_blocks": len(r.get("blocks", []))}) + "\n")
    print(f"done: {sum(not r['missing'] for r in recs)} fetched, {sum(r['missing'] for r in recs)} missing", flush=True)


if __name__ == "__main__":
    main()
