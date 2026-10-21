"""HotpotQA dev distractor: pilot 500 and final 1,000 (disjoint, seed 2027); per question a 5/5 random paragraph
split (seed 2027 + index); gold paragraphs from supporting-fact titles.  ->  data/hotpot_{pilot,final}.jsonl"""
import json
import random

import pandas as pd

from common import ROOT


def main():
    h = pd.read_parquet(f"{ROOT}/data/hotpot_dev_distractor.parquet")
    order = random.Random(2027).sample(range(len(h)), 1500)
    for name, ids in (("pilot", sorted(order[:500])), ("final", sorted(order[500:]))):
        labels = {}
        with open(f"{ROOT}/data/hotpot_{name}.jsonl", "w") as f:
            for i in ids:
                r = h.iloc[i]
                paras = [dict(title=t, text=" ".join(s)) for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
                perm = list(range(len(paras))); random.Random(2027 + i).shuffle(perm)
                a, b = sorted(perm[:5]), sorted(perm[5:])
                gold = sorted(set(r["supporting_facts"]["title"]))
                labels[str(int(i))] = r["answer"]                       # gold kept out of the generation input
                f.write(json.dumps(dict(idx=int(i), id=r["id"], question=r["question"], type=r["type"],
                                        level=r["level"], A=[paras[j] for j in a], B=[paras[j] for j in b],
                                        gold_titles=gold, A_gold=sorted({paras[j]["title"] for j in a} & set(gold)),
                                        B_gold=sorted({paras[j]["title"] for j in b} & set(gold)))) + "\n")
        json.dump(labels, open(f"{ROOT}/data/hotpot_{name}_labels.json", "w"))
        print(name, len(ids))


if __name__ == "__main__":
    main()
