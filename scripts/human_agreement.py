"""Judge–human agreement on the 50-item blind sheet (DESIGN_DRAFT.md §3). Integrity check against the blind CSV, then
the key is joined: overall agreement, Cohen's kappa, per-stratum agreement, disagreement list.
  python human_agreement.py data/human_validation_scored.xlsx  ->  runs/HUMAN_AGREEMENT.{txt,json}"""
import csv
import json
import sys

import openpyxl
import pandas as pd

from common import ROOT, normalize


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ws = openpyxl.load_workbook(sys.argv[1])["Annotate"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    blind = list(csv.reader(open(f"{ROOT}/data/human_validation_sheet.csv", newline="")))[1:]
    assert len(rows) == len(blind) == 50, "row count"
    for r, b in zip(rows, blind):
        assert int(r[0]) == int(b[0]) and r[1] == b[1] and str(r[2]) == b[2] and str(r[3]) == b[3], f"item {r[0]} altered"
    lab = {}
    for r in rows:
        v = str(r[4]).strip() if r[4] is not None else ""
        assert v in ("0", "1"), f"item {r[0]}: '{v}'"
        lab[int(r[0])] = (int(v), r[5])
    key = {k["item"]: k for k in json.load(open(f"{ROOT}/data/human_validation_key.json"))}
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    items = sorted(lab)
    h = [lab[i][0] for i in items]; j = [int(key[i]["judge"]) for i in items]
    po, k = kappa(h, j)
    res = dict(n=len(items), agreement=po, cohen_kappa=k, human_true=sum(h), judge_true=sum(j),
               judge_lenient=sum(1 for x, y in zip(h, j) if y == 1 and x == 0),
               judge_strict=sum(1 for x, y in zip(h, j) if y == 0 and x == 1), strata={})
    for st in ("judge_true", "judge_false", "ambiguous"):
        S = [i for i in items if key[i]["stratum"] == st]
        res["strata"][st] = dict(n=len(S), agreement=sum(lab[i][0] == int(key[i]["judge"]) for i in S) / len(S),
                                 human_true=sum(lab[i][0] for i in S))
    b = {int(x[0]): x for x in blind}
    res["disagreements"] = [dict(item=i, idx=key[i]["idx"], role=key[i]["role"], stratum=key[i]["stratum"],
                                 judge=bool(key[i]["judge"]), human=bool(lab[i][0]), note=lab[i][1],
                                 question=b[i][1], gold=b[i][2], response=b[i][3])
                            for i in items if lab[i][0] != int(key[i]["judge"])]
    json.dump(res, open(f"{ROOT}/runs/HUMAN_AGREEMENT.json", "w"), indent=1, ensure_ascii=False)
    L = [f"judge–human agreement (FRAMES pilot, blind, n={res['n']}): {po:.3f}; Cohen's kappa {k:.3f}",
         f"human TRUE {sum(h)}, judge TRUE {sum(j)}; judge lenient (judge T, human F) {res['judge_lenient']}, "
         f"judge strict (judge F, human T) {res['judge_strict']}",
         *[f"  {st}: n={v['n']} agreement {v['agreement']:.3f} (human TRUE {v['human_true']})" for st, v in res["strata"].items()],
         "disagreements:"]
    for d in res["disagreements"]:
        L.append(f"  item {d['item']} [{d['stratum']}, {d['role']}] judge {d['judge']} human {d['human']} note={d['note']!r}\n"
                 f"    gold: {d['gold'][:120]}\n    resp: {d['response'][:200]}")
    open(f"{ROOT}/runs/HUMAN_AGREEMENT.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
