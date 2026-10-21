# BX2 round — B3 integration audit, ownership-framing control, latency audit, statistics / leakage audit

Written and frozen (git tag `bx2-freeze`) on 2026-10-18 BEFORE any BX2 analysis output or generation. Everything in
R1–R12 (RX) and B1–B4 (BX) is unchanged and is cited, never recomputed under a new definition; BX2 adds new analyses on
the existing BX outputs plus two small new runs. All BX2 results are POST-HOC overall (designed after seeing RX and BX)
and pre-specified within BX2. Success of BX2 means answering each question with a controlled, fully reported result —
not obtaining a significant or positive result. Conventions as RX/BX: HotpotQA EM, FRAMES frozen judge (aa55da1);
question-level paired bootstrap (10,000, seed 0) for question-level outcomes; question-level cluster bootstrap for
agent-level outcomes (the two agents of a question are resampled together).

## Freeze checklist

- [x] RX / BX results are cited as they are; every BX2 output goes to `runs/bx2/` with its own file names
- [x] B3: all-agent analysis = PRIMARY; initially-wrong conditional analysis = SECONDARY (not a causal total effect)
- [x] citation-based evidence-use indicator is named a PROXY (no citation ≠ no use)
- [x] F_SPLIT re-presentation: the single prompt difference and the primary outcome are fixed below
- [x] latency: 100-question sample, 3 repetitions, randomised interleaved order, isolated vs fixed-concurrency load
- [x] B4 ±2 pp margin: verified pre-specified (DESIGN_BX.md at tag bx-freeze, commit 97fcc5c 2026-10-15 22:13:02;
      first BX output 22:18). TOST with 90 % CIs is a NEW BX2 definition, reported next to the original 95 % reading
- [x] gold-leakage audit of the inference and query-generation paths
- [x] denominators, missing / failed items, bootstrap type and alternative scorings fixed below
- [x] this document and all BX2 code committed and tagged `bx2-freeze` before running

## A. B3 evidence-integration audit (offline, HotpotQA B3 outputs; gold used only in analysis)

Cells F_SPLIT, F_CONC, F_SHAR (full support, 2 of 2) are the targets; NONE / P_EXCL / P_SHAR are reported with the
same definitions for completeness. After communication every agent in the three F cells has both supporting
paragraphs in its revision prompt; they differ in WHERE the two paragraphs sit for that agent:
`own+own` (F_SHAR both agents; F_CONC holder), `own+partner` (F_SPLIT both agents), `partner+partner` (F_CONC other).

A1 PRIMARY (all agents of the same 1000 questions, whatever their round-0 correctness; total effect of placement):
- agent-level revised-answer accuracy (A1m / B1m EM) by agent configuration: F_SPLIT agents, F_CONC holder, F_CONC
  other, F_SHAR agents; contrasts F_SPLIT − F_SHAR, F_SPLIT − F_CONC(holder), F_SPLIT − F_CONC(other)
- team final accuracy F_SPLIT − F_CONC, F_SPLIT − F_SHAR, F_CONC − F_SHAR (question-level paired)
A2 SECONDARY (agents wrong at round 0 only): revised accuracy by configuration. Conditional repair rate among
  initially-wrong agents; the conditioning set differs across cells, so it is NOT interpreted as a causal effect.
A3 MECHANISM AUDIT (question level, per cell, same definitions everywhere):
- received: some agent's revision prompt contains both supporting paragraphs (by construction 1.0 in F cells; checked)
- cited (PROXY): an agent's revision cites both supporting paragraphs' passage ids ([P#], own = P1–P5, partner =
  P6–P10); reported per agent and "any agent"
- candidate at round 0: A0 or B0 correct;  candidate at revision: A1m or B1m correct
- preserved: final MAS correct given a revision candidate;  lost in aggregation: revision candidate but final wrong
- final: MAS correct
  Loss decomposition of final errors: (no revision candidate) + (candidate lost in aggregation). Paired differences
  F_SPLIT − F_CONC and F_SPLIT − F_SHAR for every stage.

## B. Ownership-framing control on F_SPLIT (new run, HotpotQA, 1000 questions)

Round-0 A0 / B0 of B3_F_SPLIT are frozen and reused; same passages, contents, order, count and numbering (own P1–P5,
partner P6–P10), same partner answer and reasoning, same seed slots (A1m / B1m / SYNm), same combiner.
- CONTROL = the existing B3_F_SPLIT revisions (`revise_mas`, peer-provided evidence)
- TREATMENT = `revise_mas_local`, identical to `revise_mas` except ONE line: `Their evidence passages:` is replaced
  by `More of your own evidence passages:`. Nothing else changes (the diff is checked by the code before running).
PRIMARY outcome: agent-level revised accuracy, TREATMENT − CONTROL (cluster bootstrap). SECONDARY: team final accuracy
difference (paired) and candidate preservation (final correct | revision candidate). Interpretation is limited to
the attribution of the same passages within an otherwise identical prompt; it is not called a pure "ownership-boundary"
effect beyond that one-line change.

## C. Latency and cost audit (new run, FRAMES)

Sample: 100 FRAMES final questions drawn with `random.Random(20261018).sample(sorted(final ids), 100)` — fixed now,
independent of any result. Systems (exactly the BX definitions, executed with their real parallel structure:
the two agents' calls of one round run concurrently): LOCAL_ONE, PEER_EV_5 team, FULL(MAS) team, CQ_Q_5 single,
CQ_GEN_5 single, UNI_ONE, POOL. Every system includes its own BM25 retrieval step(s), timed separately.
Measured per call: wall time, input / output tokens, cache hit (must be 0: a fresh cache directory per repetition;
any hit is reported); per system: end-to-end latency, LLM time on the critical path, retrieval time, tokens, calls.
Conditions: ISOLATED (one question in flight; a dedicated server per GPU, nothing else on it; the 100 questions are
split 50 / 50 between the two identical GPUs and the GPU id is recorded) and LOADED (16 questions in flight on one
dedicated server; the in-flight count is recorded). 3 repetitions each; within a repetition the 7 systems run in a
random order per question (`random.Random(rep * 100000 + idx)`), questions in a random order per repetition.
The latency servers run with prefix caching disabled (`--no-enable-prefix-caching`), otherwise the frozen settings.
Reported: median and mean end-to-end latency per system and condition with bootstrap CIs over questions.
Reproduction by-product: with the same payloads as the frozen runs and a fresh cache, the share of calls whose text
and whose parsed answer equal the frozen cached output (configuration reproduction vs output variability).
MODELED latency sensitivity (not a measurement): end-to-end = measured LLM critical path + r × (number of serial
retrieval rounds of the system), r ∈ {0.05, 0.2, 0.5, 1, 2, 5} s; reported as the r at which systems swap order.
Local BM25 timings are NOT presented as Web-retrieval latency.

## D. Statistics and leakage audit (offline)

- denominators: for every RX / BX / BX2 analysis, n questions, skipped, failed and the intersection actually used
- B4-P1 equivalence: original 95 % CI reading (pre-specified) + TOST at ±2 pp with the 90 % cluster-bootstrap CI (new)
- bootstrap type: question-level paired for question outcomes, question-level cluster for agent outcomes; confirmed by
  code inspection and by recomputing B4-P1/P2 and B3 primary contrasts in an independent implementation
- alternative scoring for BX FRAMES (as RX A7): judge-free `contain` and `f1 ≥ 0.5` for B4-P1, B4-P2 and the B1-P1
  ladder; direction changes are reported
- leakage: (i) static — no gold field is passed to any prompt renderer in bx_run / rx_run / run_webqa (code check);
  (ii) dynamic — share of CQ_GEN queries containing the normalised gold answer vs share of round-0 answers containing
  it (FRAMES, HotpotQA); queries are model outputs, so containment reflects model knowledge, not leakage, unless the
  prompt held gold (excluded by (i))

## Files

`scripts/bx2_b3.py` (A), `scripts/bx2_run.py` + `prompts/revise_mas_local.txt` (B), `scripts/bx2_latency.py` (C),
`scripts/bx2_audit.py` (D), outputs `runs/bx2/`.

## Deviations log (appended after the freeze; the frozen text above is unchanged)

- 2026-10-18 21:02 — §C: the 2-question latency smoke showed that systems of the same question share identical payloads
  (e.g. A0 is used by LOCAL_ONE, PEER_EV_5, FULL, CQ_Q_5, CQ_GEN_5), so a fresh cache per run still produced hits for
  the second and later systems. Isolated repetition 1 was stopped (operator, by PID) before writing any output; the
  timing client was switched to an uncached request with the identical payload (`uchat`). The sample, systems,
  repetitions, order, load conditions and servers are unchanged; the servers were kept running.
- 2026-10-18 — §D: the first static-leakage detector was case-insensitive and flagged the prompt name `answer` and the
  fields `answer_1/answer_2` (false positives); replaced by a case-sensitive gold-identifier pattern. Definition unchanged.
- 2026-10-18 — §C: another user's fine-tuning jobs (2 processes, ~3.7 GB per GPU) started on both GPUs at 20:40, just
  before the latency runs; they were not touched. GPU utilisation and co-resident processes are logged every 10 s in
  runs/bx2/gpu_monitor.log and reported with the latency results.
