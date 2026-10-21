# HQ — source-constrained evidence allocation on HybridQA (tables vs. linked passages)

Frozen (git tag `hq-freeze`) on 2026-10-20 before any HQ generation. POST-HOC overall (designed after RX/BX results),
pre-specified within HQ. Results are reported whatever their sign; no contrast, subset or prompt is chosen after
outputs are seen.

## Question

Does the allocation dependence of communication gain hold when agents are separated by real information SOURCE
(Wikipedia tables vs. Wikipedia article text), not by a random document partition — and how does peer evidence
transfer compare with a feasible direct remote-query policy in that setting?

## Data and sources

HybridQA (Chen et al., 2020) release: `WikiTables-WithLinks` (commit f4ed68e5…, 15,316 tables with hyperlinked
passages) and `released_data/dev.traced.json` (3,466 dev questions with answer text and traced answer nodes). Raw
files in `data/hybridqa_raw/` (sha256 recorded in data/hq/MANIFEST.json).
- Source T (tables): every row of every table, one document per row: title "<table title> — <section> (row k)",
  text "Header1: value; Header2: value; …". BM25 index over all rows.
- Source P (passages): every distinct linked passage (deduplicated by URL), title from the URL. BM25 index over all.
- Agent A may access only Source T; agent B only Source P. Each retrieves its top 5 for the question (round 0).
- Evaluation sample: 800 dev questions, `random.Random(20261020).sample`, fixed now. Power (paired SD from existing
  runs): 95 % half-width ≈ 2.5–3.0 pp for the communication gain and ≈ 1.7–2.3 pp for the policy contrast at n = 800.
- Scoring: exact match after the frozen normalisation (primary), token F1 (secondary), against `answer-text`.

## Conditions (agent Qwen2.5-14B-Instruct cf98f3b, frozen prompts, T = 0, frozen combiner, seeds 9·10⁶ + k·1000 + slot)

| condition | agent A (tables) receives | agent B (passages) receives | prompt |
|---|---|---|---|
| LOCAL_NOCOMM | own top 5 | own top 5 | `revise_nocomm` |
| PEER_EV | own + partner's top 5 | own + partner's top 5 | `revise_e1_ev` |
| PEER_FULL | as PEER_EV + partner answer and reasoning | same | `revise_mas` |
| REMOTE_GEN | own + top 5 from Source P for ONE query A writes | own + top 5 from Source T for ONE query B writes | `query` → `revise_retrieved` |
| REMOTE_Q (matched control) | own + top 5 from Source P for the question (= partner's list) | symmetric | `revise_retrieved` |
| UNI (reference) | one agent answers on both top-5 lists | — | `answer` |

All conditions share the same round 0 (A0, B0) and use the frozen combiner for the team answer.

## Pre-specified contrasts (team final EM unless stated; question-level paired bootstrap, 10,000, seed 0)

- HQ-P1 communication gain Δcomm = PEER_EV − LOCAL_NOCOMM
- HQ-P2 feasible-policy comparison Δpolicy = PEER_EV − REMOTE_GEN (NOT a causal effect of communication: the remote
  policy retrieves different documents)
- HQ-P3 matched-evidence control REMOTE_Q − PEER_EV, agent level (identical passages), ±2 pp equivalence (95 % CI and
  TOST 90 % CI)
- Diagnostics: (i) evidence coverage from the traced answer nodes (distant supervision, reported as such): agent A's
  own top 5 contains a traced row; agent B's own top 5 contains a traced passage; support newly obtained through the
  peer message; (ii) candidate stages per condition, the four mutually exclusive categories of BX2/BX4 (correct
  preserved / lost in synthesis / created in synthesis / no correct revised or final answer); (iii) costs (LLM calls,
  tokens, retrieval calls); (iv) PEER_FULL and UNI as references.

Pre-stated reading: PEER_EV > LOCAL_NOCOMM and ≈ REMOTE_GEN supports access recovery under real source boundaries;
PEER_EV > LOCAL but < REMOTE_GEN → communication helps but direct retrieval is the better policy; small Δcomm → the
effect is sensitive to source structure / coverage, and the generalisation claim is narrowed; PEER_EV > REMOTE_GEN →
needs cause analysis (coverage diagnostics), not a general claim.

## Integrity

5-question smoke with 0 errors per job; a failed job is reported, not silently rerun; serving deviations (shared GPU)
are logged here. Files: `scripts/hq_build.py`, `scripts/hq_run.py`, `scripts/hq_analyze.py`, `scripts/hq_queue.sh`;
data `data/hq/`; outputs `runs/hq/`.

## Serving note (fixed before running)

Other users' jobs occupy most of both GPUs at freeze time (GPU0 ≈ 76 GB, GPU1 ≈ 66 GB used). The HQ queue runs on one GPU
with vLLM gpu-memory-utilization 0.48 and max-model-len 16384 (as in BX4), and waits until that much memory is free.
HQ prompts are ≤ ~4k tokens, so nothing is truncated; model, prompts and decoding are unchanged.
- 2026-10-20 22:09 — first server start failed: another user's process took GPU1 memory between the free-memory check
  and startup (nothing generated). The queue now waits and retries the server start; settings unchanged.
