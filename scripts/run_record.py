"""Independent run record for one experiment: config, seeds, code commit, question ids, model versions, wall time, results.

  python run_record.py <name> <dataset> <split> <gen.jsonl> <results.txt> <log> [judge]  ->  <dir of results>/RUN_RECORD_<name>.json
"""
import hashlib
import json
import os
import shutil
import re
import subprocess
import sys

from common import ROOT

HUB = "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()


def snap(m):
    return sorted(os.listdir(f"{HUB}/models--{m.replace('/', '--')}/snapshots"))


def main():
    name, ds, split, gen, res, log = sys.argv[1:7]
    judge = len(sys.argv) > 7
    G = [json.loads(l) for l in open(gen)]
    stamps = re.findall(r"^(\d\d-\d\d \d\d:\d\d:\d\d) ", open(log).read(), re.M)
    data = [f"data/hotpot_{split}.jsonl"] if ds == "hotpot" else ["data/frames_split.json", "data/passages.jsonl",
                                                                   "data/partition.json", "data/retrieval_frames.jsonl",
                                                                   "data/wiki_manifest.jsonl"]
    rec = dict(
        name=name, dataset=ds, split=split, n=len(G), errors=sum(g["error"] is not None for g in G),
        question_ids=sorted(g["idx"] for g in G),
        code=dict(commit_at_record=git("rev-parse", "HEAD"), tags_at_head=git("tag", "--points-at", "HEAD").split(),
                  freeze_tags={t: git("rev-list", "-n1", t) for t in git("tag").split()},
                  files={p: sha(f"{ROOT}/{p}")[:16] for p in ["scripts/run_webqa.py", "scripts/analyze.py", "scripts/common.py",
                                                              "scripts/judge.py", *[f"prompts/{f}" for f in sorted(os.listdir(f'{ROOT}/prompts'))]]}),
        data={p: sha(f"{ROOT}/{p}")[:16] for p in data},
        model=dict(agent="Qwen/Qwen2.5-14B-Instruct", agent_snapshot=snap("Qwen/Qwen2.5-14B-Instruct"), dtype="bfloat16",
                   **(dict(judge="Qwen/Qwen3-32B-FP8", judge_snapshot=snap("Qwen/Qwen3-32B-FP8"), judge_thinking=False) if judge else {}),
                   server="vLLM " + subprocess.run([os.path.join(os.environ.get("VLLM_BIN", os.path.dirname(shutil.which("vllm") or "/usr/bin/vllm")), "python"), "-c",
                                                    "import vllm;print(vllm.__version__)"], capture_output=True, text=True).stdout.strip()),
        config=dict(max_tokens=1024, temperature=dict(agents=0.0, synthesizer=0.0, judge=0.0, SC=0.7, PLACEBO_A=0.7),
                    k_star=json.load(open(f"{ROOT}/data/K_FROZEN_{ds}.json"))["k_star"],
                    evidence=dict(per_agent=5, pooled=10) | ({"retrieval": "BM25Okapi, 120-word passages"} if ds == "frames" else {"retrieval": "none (5/5 paragraph split)"})),
        seeds=dict(split=2027, partition=2027, generation="7,000,000 + idx*1000 + slot", bootstrap=0, bootstrap_B=10000),
        wall_time=dict(first_log=stamps[0] if stamps else None, last_log=stamps[-1] if stamps else None),
        results=dict(file=os.path.relpath(res, ROOT), sha256_16=sha(res)[:16], text=open(res).read()),
        reuse_rule=("FRAMES pilot questions are not reused as independent final evidence if any setting changes after the pilot"
                    if ds == "frames" else "pilot and final are disjoint (intersection 0); reported separately"))
    out = f"{os.path.dirname(os.path.abspath(res))}/RUN_RECORD_{name}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print("run record:", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
