"""Label-free execution check for the evidence-distribution experiment (Experiment W2), run on the pilot 200 before the
final. PASS needs, for every condition: 0 errors; answer parsed in >= 95% of the calls of every role; <= 2% of calls hit
the token limit; A and B pid lists identical; S_UNION and S_UNION_T7 pids equal the Distributed A top-5 + B top-5;
S_5 pids equal U[:5]; max input tokens < 32,768. It also reports (without a criterion) the round-0 answer-agreement
rate. No gold or judge is read.  python check_distribution.py pilot runs/dist  ->  runs/dist/CHECK_<split>.txt"""
import json
import sys

from common import ROOT, normalize

ROLES = ("A0", "B0", "A1m", "B1m", "A1n", "B1n")


def main():
    split, d = sys.argv[1], sys.argv[2]
    R = {r["idx"]: r for r in map(json.loads, open(f"{ROOT}/data/retrieval_frames.jsonl"))}
    L, ok_all = [], True
    for cond in ("S_UNION", "S_5", "S_UNION_T7"):
        G = [json.loads(l) for l in open(f"{d}/{cond}_{split}_gen.jsonl")]
        err = sum(g["error"] is not None for g in G); G = [g for g in G if g["error"] is None]
        calls = [g[r] for g in G for r in ROLES] + [c for g in G for x in ("MAS", "NOCOMM") for c in g[x]["calls"]]
        parse = {r: sum(g[r]["answer"] is not None for g in G) / len(G) for r in ROLES}
        trunc = sum(c["finish"] != "stop" for c in calls) / len(calls)
        want = (lambda i: R[i]["A"] + R[i]["B"]) if cond != "S_5" else (lambda i: R[i]["U"][:5])
        pids = all(g["pids"]["A"] == g["pids"]["B"] == want(g["idx"]) for g in G)
        mx = max(c["input_tokens"] for c in calls)
        agree = sum(bool(g["A0"]["answer"]) and normalize(g["A0"]["answer"]) == normalize(g["B0"]["answer"] or "") for g in G) / len(G)
        ok = err == 0 and min(parse.values()) >= 0.95 and trunc <= 0.02 and pids and mx < 32768
        ok_all &= ok
        L.append(f"{cond} {split}: n={len(G) + err} errors={err} parse_min={min(parse.values()):.3f} trunc={trunc:.4f} "
                 f"pids_ok={pids} max_input={mx} round0_agreement={agree:.3f} -> {'ok' if ok else 'FAIL'}")
    L.append("PASS" if ok_all else "FAIL")
    open(f"{d}/CHECK_{split}.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
