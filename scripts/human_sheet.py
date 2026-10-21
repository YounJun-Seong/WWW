"""50-item stratified human validation sheet for the FRAMES judge (DESIGN_DRAFT.md §3), drawn from pilot judgments:
20 judge-TRUE, 20 judge-FALSE, 10 ambiguous (judge verdict disagrees with normalized gold containment). Items are
(question, role) pairs with the arm hidden; seed 2027. Writes a blind CSV for annotation and a separate key file."""
import argparse
import csv
import json
import random

import pandas as pd

from common import ROOT, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    a = ap.parse_args()
    df = pd.read_csv(f"{ROOT}/data/frames_test.tsv", sep="\t")
    items = []
    for r in map(json.loads, open(a.judged)):
        for role, v in r.items():
            if isinstance(v, dict) and "verdict" in v and v["response"]:
                gold = str(df.loc[r["idx"], "Answer"])
                contain = normalize(gold) in normalize(v["response"])
                items.append(dict(idx=r["idx"], role=role, gold=gold, response=v["response"], judge=v["verdict"],
                                  ambiguous=(contain != v["verdict"])))
    rng = random.Random(2027)
    amb = rng.sample([i for i in items if i["ambiguous"]], min(10, sum(i["ambiguous"] for i in items)))
    rest = [i for i in items if i not in amb]
    pick = amb + rng.sample([i for i in rest if i["judge"]], 20) + rng.sample([i for i in rest if not i["judge"]], 20)
    rng.shuffle(pick)
    with open(f"{ROOT}/data/human_validation_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "question", "gold_answer", "system_response", "human_correct (1/0)", "note"])
        for n, i in enumerate(pick, 1):
            w.writerow([n, df.loc[i["idx"], "Prompt"], i["gold"], i["response"], "", ""])
    json.dump([dict(item=n, idx=i["idx"], role=i["role"], judge=i["judge"], stratum="ambiguous" if i in amb else
                    ("judge_true" if i["judge"] else "judge_false")) for n, i in enumerate(pick, 1)],
              open(f"{ROOT}/data/human_validation_key.json", "w"), indent=1)
    print(f"sheet: {len(pick)} items (ambiguous {len(amb)})")


if __name__ == "__main__":
    main()
