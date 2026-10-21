# BX4 — evidence-sufficiency audit, matched placement transitions, evidence-aware aggregation

Frozen (git tag `bx4-freeze`) on 2026-10-19 before any BX4 run. POST-HOC overall; pre-specified within BX4. Original
RX/BX/BX2/BX3 results are unchanged. Gold is used only to build samples and to score/annotate, never in agent prompts.

## A. Supporting-evidence sufficiency audit (R2 injection, I2 = both injected supporting passages)

Unit: question with |G_inj| = 2 (FRAMES 583, HotpotQA 1000); the injected set G_inj is the same for both receivers of a
question, so sufficiency is a property of (question, G_inj). Outcome: R2_I2 revised correctness of receivers that were
WRONG at round 0 (frozen round 0), agent level, cluster bootstrap by question.

A1 model proxies, on ALL questions (labelled "model proxy" wherever reported; not human verification):
- SUFF-JUDGE: the frozen judge model (Qwen3-32B-FP8 aa55da1, T = 0, thinking off, max 512 tokens) sees question,
  reference answer and ONLY the two injected passages, and returns SUFFICIENT / INSUFFICIENT / AMBIGUOUS
  (prompt `sufficiency`; it is told not to use outside knowledge).
- READER: the agent model (Qwen2.5-14B) answers with ONLY the two injected passages as evidence (frozen `answer`
  prompt, T = 0, seed slot 500), scored like every other answer (frozen judge / EM). Reader-correct = behaviourally
  sufficient for this model (it can also reflect parametric knowledge; reported as such).
- "Doubly sufficient" = SUFF-JUDGE SUFFICIENT and READER correct.
Primary (pre-specified): failure rate (still wrong after I2) among initially-wrong receivers, overall and within
SUFFICIENT, within INSUFFICIENT/AMBIGUOUS, and within doubly-sufficient, per dataset, with 95 % CIs; plus the share
of all I2 failures that occur on SUFFICIENT items.
A2 human verification (prepared, not performed here): 150 questions per dataset drawn with
`random.Random(20261019)` among questions with ≥ 1 initially-wrong receiver; two blind annotator workbooks (different
random orders; no model output, no correctness shown) + a key file. Labels: Sufficient / Insufficient / Ambiguous,
"from the passages alone; do not use outside knowledge". Analysis of human labels uses the same estimators as A1.

## B. Matched-question placement transitions (offline, HotpotQA B3 F_SPLIT vs F_CONC)

Same 1000 questions, same supporting paragraphs, same distractor pool, 5 passages per agent in both cells. Per
question: 2×2 table of final correctness (split × concentrated), exact McNemar test; for discordant questions, the
stage at which the losing condition failed (no correct revised candidate vs lost in synthesis); and the same
comparison stratified by round-0 candidate presence in each cell (both / split only / concentrated only / neither).
Limitation stated in advance: allocation changes round-0 states, so this explains the total allocation effect but does
not remove initial-state differences.

## C. Evidence-aware aggregation on the same revised candidates (HotpotQA B3 F_SPLIT, F_CONC, F_SHAR)

The revised answers A1m / B1m of the original B3 runs are kept fixed. Aggregators:
- EXISTING: the frozen combiner (as reported)
- EVIDENCE-AWARE: on disagreement, the `synthesize_ev` synthesizer (as RX R11 EVSYN) that additionally sees the union of
  both agents' passages (seed slot 310); on agreement, the agreed answer
- ORACLE candidate selection (diagnostic upper bound, not a system): correct iff any revised answer is correct
Primary: the F_SPLIT − F_CONC final-accuracy gap under EXISTING vs EVIDENCE-AWARE vs ORACLE (paired bootstrap), and
lost-in-synthesis counts under each aggregator.

## Not run

A live-Web / real-index experiment (not controllable before the deadline; recommended as follow-up).

## Files

`prompts/sufficiency.txt`, `scripts/bx4_run.py`, `scripts/bx4_sheets.py`, `scripts/bx4_analyze.py`,
`scripts/bx4_queue.sh`; outputs `runs/bx4/`, annotation workbooks `runs/bx4/human/`.

## Serving note (fixed before running)

Other users' jobs occupy ~38 GB of GPU0 and ~70 GB of GPU1 at freeze time. BX4 runs on GPU0 only, sequentially, with
vLLM gpu-memory-utilization 0.45 (agent) / 0.48 (judge) instead of 0.70; model snapshots, prompts and decoding are
unchanged (only KV-cache capacity, hence throughput, differs). Previously judged answer strings reuse the shared judge
cache as always.

## Deviations log

- 2026-10-19 18:58 — the agent server at gpu-memory-utilization 0.45 / max-model-len 32768 failed to start (2.3 GiB KV
  cache < 6 GiB for one 32768-token request). Relaunched with utilization 0.48 and max-model-len 16384 (agent) /
  8192 (judge), keeping ≥ 3 GB headroom for the co-resident jobs. BX4 prompts are far below these limits (READER
  ≈ 1.5k, evidence-aware synthesis ≈ 5k, sufficiency judge ≈ 1.5k + 512 tokens), so no request is truncated; nothing
  had been generated before the relaunch.
- 2026-10-19 19:16 / 23:19 — the judge server failed twice: first OOM when another user's new jobs claimed GPU0 memory
  during startup; then at utilization 0.48 the 32B judge left only 0.19 GiB KV cache (< 2 GiB for one 8192-token
  request). Relaunched at utilization 0.56 once the co-resident jobs used 32 GB per GPU (≈ 3.5 GB headroom kept).
  Judge model, prompt and decoding unchanged; previously judged strings reuse the shared judge cache.
- 2026-10-19 23:24 — at utilization 0.56 the judge allocated its KV cache (6.5 GiB) but ran out of memory during CUDA-graph
  capture; relaunched with `--enforce-eager` (no CUDA graphs). Same model, prompt, decoding; only execution mode differs.
