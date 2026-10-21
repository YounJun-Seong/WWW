# When Does Communication Help? Evidence Access and System Utility in Multi-Agent Web QA

Anonymous code, frozen protocols, and frozen outputs for the WWW 2027 submission. Every number in the paper can be
recomputed from the files in `runs/` without re-running any model (all generations and judgments are stored).

## Repository layout

| Path | Contents |
|---|---|
| `protocols/` | Pre-registration documents, each frozen (committed) before the corresponding runs: `00_original_design.md` (original confirmatory study), `DESIGN_BX.md` (B1–B4), `DESIGN_BX2.md` (integration audit, framing control, latency, statistics audit), `DESIGN_BX3.md` (execution-robustness regeneration), `DESIGN_BX4.md` (sufficiency audit, matched placement transitions, evidence-aware aggregation), `DESIGN_HQ.md` (HybridQA source-constrained validation). Deviations are logged at the end of each file. |
| `prompts/` | All prompts (agent, revision variants, synthesizer, judge, query generation, sufficiency audit). |
| `scripts/` | Generation, judging, and analysis code (Python), plus the job queues (shell) that were used to run them. |
| `data/` | Question splits, retrieval lists, intervention assignments (`data/rx`, `data/bx`), HybridQA sample (`data/hq`), and annotation sheets. |
| `runs/` | Frozen generations (`*_gen.jsonl`), judgments (`*_judged.jsonl`), and analysis outputs (`RESULTS_*.txt/json`). |
| `paper/` | Figure scripts, figures, and caption drafts. |

## Manuscript map

| Paper | Analysis | Script(s) | Result file(s) |
|---|---|---|---|
| Sec. 4, cohorts, judge audit | FRAMES / HotpotQA final runs, judge audit | `run_webqa.py`, `judge.py`, `analyze.py`, `judge_audit_final.py`, `human_agreement.py` | `runs/final/RESULTS_*_final.txt`, `runs/final/JUDGE_AUDIT_FINAL.txt`, `runs/HUMAN_AGREEMENT.txt` |
| Table 1, Sec. 5.1 (RQ1) | distributed vs shared-union allocation, DiD | `run_distribution.py`, `analyze_distribution.py`, `run_dist_hotpot.py`, `analyze_dist_hotpot.py` | `runs/dist/`, `runs/dist_hotpot/` |
| Table 1, length control and seeds | DPAD, four partition seeds | `rx_build.py`, `rx_run.py`, `rx_analyze.py` (`r4`, `r5`) | `runs/rx/RESULTS_r4_*`, `runs/rx/RESULTS_r5_*` |
| Table 2, Sec. 5.2 (RQ2) | cached-state message interventions (E1) | `run_e1.py`, `analyze_e123.py` | `runs/e1/RESULTS_e1_*` |
| Sec. 5.2, App. C | supporting-evidence removal / replacement (R1), injection (R2), message noise (R10, R12) | `rx_run.py`, `rx_judge.py`, `rx_analyze.py` (`r1`, `r2`, `r10`, `r12`) | `runs/rx/RESULTS_r1_*`, `r2_*`, `r10_*`, `r12_*`; 32B: `runs/rx32/` |
| Sec. 5.2, App. C | sufficiency audit (model proxies; human sheets) | `bx4_run.py`, `bx4_sheets.py`, `bx4_analyze.py a` | `runs/bx4/RESULTS_a.txt`, `runs/bx4/human/` |
| Sec. 5.3 (RQ3), Table 4 | FRAMES E2 placement factorial, HotpotQA replication (R3) | `build_e2.py`, `run_e23.py`, `analyze_e123.py`, `rx_analyze.py r3` | `runs/e2/`, `runs/rx/RESULTS_r3_hotpot.*` |
| Fig. 2, Sec. 5.3 | support availability × placement (B3), four-category outcome partition | `bx_build.py`, `bx_run.py`, `bx_analyze.py b3`, `bx2_analyze.py a` | `runs/bx/RESULTS_b3_hotpot.*`, `runs/bx2/RESULTS_a.*` |
| Sec. 5.3 | matched-question transitions, evidence-aware aggregation, oracle selection | `bx4_run.py`, `bx4_analyze.py b c` | `runs/bx4/RESULTS_b.*`, `runs/bx4/RESULTS_c.*` |
| Sec. 5.3 | evidence-origin framing control on F_SPLIT | `bx2_run.py`, `bx2_analyze.py b` | `runs/bx2/RESULTS_b.*` |
| Fig. 3, Sec. 5.4 (RQ4) | access tiers R0–R5, equal-budget transfer (B1/B4), single-agent baselines (R6) | `bx_run.py`, `bx_analyze.py b14`, `rx_analyze.py r6` | `runs/bx/RESULTS_b14_*`, `runs/rx/RESULTS_r6_*` |
| Sec. 5.4 | evidence-aware controller (B2) | `bx_analyze.py b2`, `rx_offline.py x7` | `runs/bx/RESULTS_b2_*`, `runs/rx/RESULTS_x7_*` |
| Sec. 3.5, 4 | uncached latency audit, modeled remote-search delay | `bx2_latency.py`, `bx2_analyze.py c` | `runs/bx2/RESULTS_c.*`, `runs/bx2/lat_*.jsonl` |
| Sec. 5.4, cross-source | HybridQA source-constrained validation | `hq_build.py`, `hq_run.py`, `hq_analyze.py` | `runs/hq/RESULTS_hq.*` |
| App. F | execution-robustness regeneration | `bx3_queue.sh`, `bx3_analyze.py` | `runs/bx3/RESULTS_bx3.*` |
| Statistics audit | denominators, TOST, independent recomputation, judge-free scoring, leakage | `bx2_analyze.py d`, `rx_analyze.py` (`RX_SCORE=contain` or `f1`) | `runs/bx2/RESULTS_d.*`, `runs/rx/RESULTS_*_contain.*`, `*_f1.*` |
| App. B | graded overlap, topic partitions, backbones, aggregation | `run_overlap.py`, `analyze_overlap.py`, `build_e3.py`, `analyze_backbone.py`, `rx_analyze.py r11` | `runs/overlap/`, `runs/e3/`, `runs/backbones/`, `runs/rx/RESULTS_r11_*` |
| Figures | Fig. 1–3 | `paper/make_figs_bx.py` | `paper/figures/`, `paper/FIGURE_CAPTIONS.md` |

Supplementary (not in the main text): AgentWebBench diagnostic (`scripts/awb_*`, `runs/awb/`) and offline policy analyses (`rx_offline.py`, `runs/rx/RESULTS_x8_*`, `RESULTS_a1..a6_*`).

## Reproducing the numbers

```bash
pip install -r requirements.txt
export WQ_ROOT=$(pwd) PYTHONPATH=$(pwd)/scripts
cd scripts
python rx_analyze.py r1 frames      # any analysis reads only runs/ and data/
python bx_analyze.py b14 hotpot
python bx4_analyze.py b
python hq_analyze.py
```

Re-running generation requires a vLLM 0.11 server for `Qwen/Qwen2.5-14B-Instruct` (snapshot `cf98f3b`) and, for
FRAMES judging, `Qwen/Qwen3-32B-FP8` (snapshot `aa55da1`). Set `WQ_BASE_URL` / `WQ_MODEL` and `WQ_JUDGE_URL` /
`WQ_JUDGE_MODEL`, plus optionally `PYTHON`, `VLLM_BIN`, and `HF_HUB_CACHE` for the queue scripts. The response cache
used during the original runs is not distributed. Decoding is greedy with fixed per-call seeds, but regenerated text
can differ with serving configuration (see `protocols/DESIGN_BX3.md` and App. F).

## Data

- FRAMES (`data/frames_test.tsv`): google/frames-benchmark. Wikipedia passages are rebuilt with `fetch_wiki.py` and
  `build_corpus.py` (`data/passages.jsonl` is not distributed; retrieval lists are in `data/retrieval_frames.jsonl`).
- HotpotQA distractor dev (`data/hotpot_dev_distractor.parquet`), CC BY-SA 4.0.
- HybridQA: download `WikiTables-WithLinks` and `released_data/dev.traced.json` from the official release into
  `data/hybridqa_raw/`, then run `hq_build.py` (checksums in `data/hq/MANIFEST.json`).
- Wikipedia-derived text is CC BY-SA. Gold annotations are used only to construct interventions and to score; they
  are never shown to the agents (see the protocols).

## Status labels

Original allocation, message, and placement analyses follow `protocols/00_original_design.md`. RX, BX, BX2, BX3, BX4,
and HQ are post-hoc extensions, each frozen before its own runs; they are not independent confirmatory replications.
