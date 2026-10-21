# BX round — access constraints, equal-budget retrieval vs communication, support recall × placement, deployable policy

Written and frozen (git tag `bx-freeze`) on 2026-10-15 BEFORE any BX generation. Everything below is fixed in advance:
arms, primary contrasts, cost definitions, exclusion and stopping rules. All BX results are POST-HOC with respect to the
original frozen confirmatory reads (they are designed after seeing R1–R12), but pre-specified within BX: no contrast,
threshold, prompt or subset is chosen after BX outputs are seen, and every listed contrast is reported whatever its sign.

Unchanged from RX: agent Qwen2.5-14B-Instruct (snapshot cf98f3b), vLLM 0.11, T=0 except SC sampling (T=0.7), max 1024
output tokens, frozen combiner (agree after normalisation → that answer, else one synthesizer call), frozen judge
(Qwen3-32B-FP8, snapshot aa55da1) for FRAMES, EM for HotpotQA, paired bootstrap over questions (10,000, seed 0),
agent-level quantities resampled by question. Round-0 answers (A0/B0) are the frozen ones and are reused.
Evaluation sets: FRAMES final (624), HotpotQA final (1000). Development set (B2 only): FRAMES pilot (200), HotpotQA pilot (500).

## Research question

Under which information-access and resource constraints does inter-agent communication provide practical value over
centralised retrieval? The study does not assume a winner: each access regime is reported, including regimes where a
single centralised agent dominates.

## B1 + B4 — access regimes and equal-budget transfer (FRAMES and HotpotQA)

Setting. Two document collections (FRAMES: the page-level 50/50 partition of the 2,515-page corpus, BM25 per partition;
HotpotQA: each question's 10 paragraphs split 5/5). Agent X ∈ {A, B} owns one collection; "remote" = the other one.

New arms (agent-level revisions of the frozen round-0, then the frozen combiner for the team answer):

| arm | what reaches agent X from the remote collection | framing (prompt) | extra LLM calls |
|---|---|---|---|
| PEER_EV_k, k∈{1,3,5} | the partner's top-k passages, retrieved by the partner with the question (BM25 rank order) | "another assistant shared" (`revise_e1_ev`) | 0 |
| CQ_Q_k, k∈{1,3,5} | the SAME top-k passages, retrieved by X itself from the remote index with the question | "you searched a separate collection" (`revise_retrieved`) | 0 |
| CQ_GEN_k, k∈{1,3,5} | top-k from the remote index for ONE query that X writes after reading its own evidence and reasoning (`query`) | `revise_retrieved` | 1 query call |
| LOCAL_SC | nothing: k*=4 samples at T=0.7 on own evidence, majority | `answer` | 4 samples |

FRAMES remote index = BM25 over the partner partition (~50k passages; the seed-2027 retrieval lists are re-derived and
must match the frozen ones exactly, else nothing runs). HotpotQA remote index = the partner's 5 paragraphs, scored with
BM25 fitted on all dev-distractor paragraphs (HotpotQA's remote collection is closed: at k=5 every remote arm sees the
same 5 paragraphs; only k<5 can differ). PEER_EV_5 on FRAMES is identical to RX R1_FULL (cache).

Existing arms reused (no new generation): LOCAL_ONE (A0/B0), local self-revision (A1n/B1n) and NOCOMM, the E1 disclosure
ladder ANS / ANS_RSN / ANS_EV / EV / FULL(=MAS), federated central UNI_ONE / UNI_REV / UNI_SC (question sent to each
partition, top-5 each, one agent reads all 10), monolithic central POOL / SC (one combined index, top-10).

Access regimes (an arm is admissible in a regime if it uses only that access):
- R0 local only: LOCAL_ONE, LOCAL_SC, local revision, NOCOMM
- R1 peer, no documents leave the owner: + ANS, ANS_RSN
- R2 peer, documents may be shared by their owner: + ANS_EV, EV, FULL(MAS), PEER_EV_k
- R3 remote query access to the other index: + CQ_Q_k, CQ_GEN_k
- R4 federated full access: + UNI_ONE, UNI_REV, UNI_SC
- R5 monolithic central index: + POOL, SC

Outcomes. Accuracy (team final for two-agent systems; for single-agent systems the mean over the two possible
deployments, X=A and X=B). Costs per question, each reported separately (no single exchange rate is assumed):
LLM calls, input tokens, output tokens, retrieval calls, passages moved across the access boundary, and serial depth
(LLM rounds on the critical path, parallel calls counted once). Latency is NOT measured: calls are cached, so wall time
is meaningless; serial depth is its proxy.

Primary contrasts (pre-registered):
- B4-P1 framing: CQ_Q_5 − PEER_EV_5, agent level (identical passages; tests whether "communication" adds anything beyond
  the transferred evidence). Practical-equivalence region ±2 pp, declared in advance.
- B4-P2 query targeting at equal passage budget: CQ_GEN_k − PEER_EV_k for k = 1, 3, 5, agent level.
- B4-P3 budget curve: team accuracy of PEER_EV_k and CQ_GEN_k as a function of k.
- B1-P1 access ladder: accuracy and every cost of the best admissible system per regime R0–R5; recovery ratio
  (best R2 − best R0) / (best R4 − best R0) with bootstrap CI.
- B1-P2 frontier shift: the accuracy–cost Pareto frontier (tokens; LLM calls; retrieval calls) of the cumulative
  admissible set R0, R0–R1, …, R0–R5.

## B3 — support recall × placement (HotpotQA final; FRAMES not run)

FRAMES gold is page-level and cannot certify that the retrieved passages suffice, so B3 runs on HotpotQA only (the two
supporting paragraphs are a complete support set). Each agent gets 5 paragraphs; fillers are the question's own 8
non-gold paragraphs (original order) then hard negatives (answer-free, from RX), never shared between agents; each list
is shuffled with random.Random(2027 + idx). The holder side alternates with idx parity (as RX E2H). g1 / g2 = the two
supporting paragraphs in context order.

| cell | team support recall | placement | A / B (holder H, other O) |
|---|---|---|---|
| NONE | 0 | — (no-support control) | H: fill 0–4 · O: fill 5–9 |
| P_EXCL | 1 of 2 | exclusive | H: g1 + fill 0–3 · O: fill 4–8 |
| P_SHAR | 1 of 2 | shared | H: g1 + fill 0–3 · O: g1 + fill 4–7 |
| F_SPLIT | 2 of 2 | complementary (one each) | H: g1 + fill 0–3 · O: g2 + fill 4–7 |
| F_CONC | 2 of 2 | concentrated in one agent | H: g1 + g2 + fill 0–2 · O: fill 3–7 |
| F_SHAR | 2 of 2 | shared (both have both) | H: g1 + g2 + fill 0–2 · O: g1 + g2 + fill 3–5 |

Full pipeline per cell (A0, B0, MAS and No-comm revisions, MAS / NOCOMM). Primary contrasts (P-int = MAS − NOCOMM):
- B3-P1 P-int(F_SPLIT) − P-int(F_SHAR) (placement, full recall; replicates R3)
- B3-P2 P-int(P_EXCL) − P-int(P_SHAR) (placement, partial recall)
- B3-P3 P-int(F_SPLIT) − P-int(F_CONC) (complementary vs concentrated full support)
- B3-P4 retrieval floor = MAS accuracy in NONE; utilisation failure = 1 − MAS accuracy in F_SHAR (full support in every
  agent); team accuracy by recall level (0 / 1 / 2) for MAS and NOCOMM.

## B2 — deployable communication policy (offline; dev = pilot, eval = final)

Inputs available after round 0, no gold, no outcomes: answer disagreement, hedging count (RX X7 regex), own / partner
best-BM25 score gap, TF-IDF evidence novelty, missing round-0 answer, round-0 answer length. Communication budgets
b ∈ {0.2, 0.4, 0.6, 0.8} (fraction of questions that communicate). Policies, each ranking questions by a score and
communicating the top-b fraction (threshold = the pilot quantile, applied unchanged to final; realised activation
reported):
- RANDOM — expected value of a random b-fraction (analytic mean of MAS / NOCOMM mixing)
- DISAGREE-RANK — disagreement first, ties by novelty
- NOVELTY-RANK — novelty
- EVIDENCE-AWARE — logistic regression on pilot predicting [MAS correct] − [NOCOMM correct] > 0 from the inputs above
  (standardised on pilot, L2, C = 1, no tuning), ranked by predicted probability
Primary: EVIDENCE-AWARE − RANDOM and EVIDENCE-AWARE − DISAGREE-RANK at each budget on final (paired bootstrap).
Secondary (fixed rules, no training): ROUTE-PEER (disagree → MAS, else NOCOMM) vs ROUTE-RETRIEVE (disagree → the team
answer of CQ_GEN_5, else NOCOMM), with costs. Promotion to a main contribution only if EVIDENCE-AWARE beats both
RANDOM and DISAGREE-RANK at ≥ 2 budgets on both datasets; otherwise it is reported as an operational implication.

## Exclusion, stopping and integrity rules

- A generation job must pass a 5-question smoke run with 0 errors; a full job with any error is reported as failed and
  its output is not used (no silent reruns; infrastructure reruns are logged with the reason).
- Every arm is read on the same question set (skipped questions are reported per arm; contrasts use the intersection).
- No new primary contrast is added after outputs are seen; anything added later is labelled exploratory.
- Negative and null results are reported in the same tables as positive ones.

## Files

`scripts/bx_build.py` (B3 cells, HotpotQA partner rankings → data/bx/), `scripts/bx_run.py` (generation),
`scripts/rx_judge.py` (frozen judge, unchanged), `scripts/bx_analyze.py` (b14, b3, b2), prompts `query.txt`, `revise_retrieved.txt`,
queue `scripts/bx_queue.sh`. Outputs `runs/bx/`.
