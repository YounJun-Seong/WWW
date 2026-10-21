# WWW 2027 — Complementarity vs. utility in partitioned-source Web QA (design; frozen as `www-freeze` once code and prompts are committed)

Decisions confirmed by the user on 2026-10-08:
- **judge:** Qwen3-32B-FP8, local;
- **arms:** a No-communication control is added;
- **pilot:** FRAMES 200 + HotpotQA 500;
- **deadlines:** WWW 2027 abstract Oct 18 and full paper Oct 25, AoE (checked on the official CFP by the user);
- **paper length:** 8 pages, at most 12 including references and appendix.

## 0. Claim scope

Both settings are **gold-source-bounded, frozen-Wikipedia retrieval**: reproducible web-evidence
integration, not live open-web browsing. The paper states this explicitly and makes no
generalization claim to the open web.

## 1. Data, splits and provenance (all recorded)

**FRAMES (test, 824 questions).**
- **Pilot:** 200 questions, seed 2027, in `data/frames_split.json` (sha256 prefix 4a66dab6…).
- **Final:** the other **624** questions.
- **Rule:** if anything is adjusted after the pilot, the pilot questions are never reported as
  independent evidence. The final result comes from the 624 questions under the frozen setting.
  If nothing is adjusted, the pilot is reported as a pilot, and the final is still the 624.

**Evidence.**
- Every page linked by any FRAMES question (2,476 pages) is fetched at its last revision ≤
  2024-09-19 and rendered to text. `data/wiki_manifest.jsonl` records the revid and timestamp.
- Pages are chunked into passages of about 120 words, each prefixed by the page title. Passage ids
  are recorded.

**Source partition.**
- The page-level 50/50 split uses seed 2027 and covers all pages, so it is global and not
  gold-aware: Source A vs Source B.
- Per question, the following are recorded:
  - each gold page's owner (A or B);
  - which gold pages each agent *holds*;
  - which gold pages each agent *retrieved* in its top-k.

**HotpotQA (dev distractor, 7,405 questions).**
- **Pilot:** 500 questions, seed 2027.
- **Final:** up to a further 1,000 questions after the pilot. These are disjoint from the pilot and
  are drawn with the same seed stream.
- For each question, its 10 paragraphs are split 5/5 at random (seed 2027 + question index).
- The agents receive their 5 paragraphs directly; HotpotQA has no retrieval step.
- Supporting-fact titles define the gold paragraphs.

## 2. Arms (backbone Qwen2.5-14B-Instruct, private-cache snapshot cf98f3b; local vLLM, BF16)

Generation settings: T = 0 for agent answers, the synthesizer and the judge. T = 0.7 for SC samples.
max_tokens = 1,024.

**Evidence budget.** Each agent gets **k = 5** passages. Pooled single gets **k = 10** from A ∪ B, the
same total number of candidates and evidence passages. Evidence input tokens are logged for every
call.

| Arm | Definition |
|---|---|
| Single-A, Single-B | answer from the agent's own top-5 (FRAMES: BM25 over its own source; HotpotQA: its 5 paragraphs) |
| Pooled single | one agent with the top-10 from A ∪ B (HotpotQA: all 10 paragraphs); no interaction |
| No-comm | round 0 = Single-A and Single-B. Round 1: each agent revises once with **its own evidence only**, using the same revision template minus the partner block. Final: the shared final combiner. |
| MAS (evidence-sharing) | round 0 = Single-A and Single-B. Round 1: each agent revises once after receiving the **partner's answer, rationale and 5 passages**. Final: the shared final combiner. |
| Final combiner (same for No-comm and MAS) | if the two final answers agree after normalization, that answer; otherwise one synthesizer call that sees the question and the two final answers with their rationales (no raw passages) |
| SC-k (compute-matched) | Pooled-single prompt sampled k times at T = 0.7; majority over normalized answers. k = argmin\|k·t_s − T_MAS\|, from a token probe on 20 pilot questions (tokens only). |

**Compute-matched comparisons** are only those against SC-k, and against No-comm with its token ratio
reported. Comparisons against the Single and Pooled arms are evidence-matched, not compute-matched,
and are labelled as such.

## 3. Scoring

- **HotpotQA:** EM and F1 with the standard normalization.
- **FRAMES:** Qwen3-32B-FP8 judge (private-cache snapshot aa55da1), T = 0, thinking disabled. The
  frozen prompt `prompts/judge.txt` outputs `TRUE` or `FALSE`: does the response contain an answer
  semantically equivalent to the gold answer?
- **Human validation:** 50 FRAMES items, stratified:
  - 20 judge-TRUE;
  - 20 judge-FALSE;
  - 10 "ambiguous", meaning the judge verdict disagrees with normalized string containment.

  The user annotates them blind to the arm. Reported: judge–human agreement, Cohen's κ, and
  disagreement types.

## 4. Estimands (six gates plus evidence-transfer decomposition; paired bootstrap over questions, 10,000, seed 0)

**Primary.**
- **P-int:** Acc(MAS) − Acc(No-comm). This is the effect of exchanging information from identical
  round-0 states.
- **P-util:** Acc(MAS) − Acc(Pooled single), the realized utility against the same evidence without
  interaction.

**Secondary.**
- G1: Acc(MAS) − Acc(SC-k), compute-matched; Acc(MAS) − max(Acc(A), Acc(B)).
- G2 (complementarity):
  - P(A ∪ B correct) − max(P(A), P(B)) at round 0;
  - a placebo version that uses a second, independent Single-A draw at T = 0.7.
- G4 (coverage): gold-answer presence in {A⁰, B⁰} vs {A¹, B¹}, with κ_gain and κ_loss.
- G5 (revision): BRR / HRR / DI on discordant round-0 pairs, for MAS and for No-comm.

**Evidence-transfer decomposition** (per discordant pair: one agent correct, one wrong at round 0).
- **Availability (held):** the partner's *source* contains at least one gold page.
- **Availability (retrieved):** the partner's *top-k* contains at least one gold page.
- **Utilization:** given that the partner's retrieved evidence contained gold, the wrong agent's
  revised answer is correct (MAS). The same rate in No-comm is the baseline.
- **Transfer success** = availability (retrieved) × utilization.

**Failure taxonomy for MAS-wrong questions** (mutually exclusive, in this order):
1. **Retrieval miss:** no gold page in A's or B's top-k.
2. **Insufficient evidence:** some, but not all, gold pages retrieved across A ∪ B. FRAMES only;
   HotpotQA uses supporting paragraphs.
3. **Integration failure:** all gold pages or paragraphs retrieved across A ∪ B, yet MAS is wrong.

Oracle quantities are analysis-only upper bounds.

## 5. Order

1. Code and prompts committed.
2. Tag `www-freeze`.
3. Smoke: 5 pilot questions per dataset. Labels are used only for the judge format check; no
   accuracy is computed.
4. Token probe (fixes k).
5. Pilot run.
6. Sealed pilot analysis.
7. Decision: whether the pilot passed execution checks, and whether any setting changed (recorded).
8. Final run: FRAMES 624 + HotpotQA 1,000.
9. Sealed final analysis.

Servers run only during steps. Other users' GPU processes are never touched.

## Freeze note (2026-10-08)

- Tagged `www-freeze`, with the code, prompts and splits committed.
- **FRAMES corpus.** The passages, the partition and the BM25 retrieval file are produced
  deterministically by the frozen `scripts/build_corpus.py` once the snapshot fetch has finished. The
  fetch is serial and honours Retry-After (Wikipedia returned 429 to the first, 3-way-parallel attempt).
  The hashes of those artifacts are appended here before any FRAMES generation.
- **HotpotQA** needs no corpus, so its pilot runs first.
- **SC k** is fixed per dataset by `token_probe.py` on 20 pilot questions, using tokens only.

## HotpotQA pilot decision (2026-10-08, before the final run)

**Pilot record.**
- The sealed pilot analysis was run once: commit `bde5d79`, `runs/RESULTS_hotpot_pilot.txt`.
- n = 500, 0 generation errors, k* = 4 (`data/K_FROZEN_hotpot.json`).
- P-int (MAS − No-comm) +17.2 pp [+13.6, +20.8].
- P-util (MAS − Pooled) −6.4 pp [−10.0, −2.8].
- MAS − SC-4 −7.8 pp [−11.2, −4.6] (token ratio 0.93).
- The pilot is reported only as a pilot. It is never pooled with the final.

**Decision (user, 2026-10-08): no change.** None of the following changes for the final:
- the model;
- the prompts;
- the split seed;
- the evidence setting;
- the scoring and analysis code (`analyze.py`);
- the arms;
- the CI rules.

**Final-set check.**
- Pilot 500 vs final 1,000: the intersection is **0**, by dataset idx, by HotpotQA question id, and by
  question text.
- All 1,000 label keys match the final inputs.

**Estimands for the final (unchanged).**
- Primary: paired EM differences MAS − No-comm and MAS − Pooled.
- Key secondary: MAS − SC-4 (compute-matched). The other secondaries are as in §4.
- The hypotheses and decision criteria are not changed after seeing the final.

**Added descriptive log** (requested by the user; it does not touch the sealed analysis).
- `scripts/evidence_log.py` writes `runs/EVIDENCE_LOG_hotpot_final.{jsonl,txt}`.
- Per question and per agent it separates:
  - **availability:** which gold paragraphs the agent holds, and which its partner holds;
  - **transmission:** whether each partner-held gold paragraph's text is in the re-rendered MAS revision
    prompt, and whether the call finished normally;
  - **utilization:** EM after revision, whether the revision cites a partner gold passage, and whether
    the answer equals the partner's round-0 answer;
  - **integration failure:** all gold was in the prompt and the answer has F1 = 0.
- Cases without sufficient grounds are labelled **unknown**:
  - the answer is unparsed or the call was truncated;
  - EM 0 but F1 > 0;
  - some gold paragraph is unavailable.
- The script was code-tested on the pilot output. That output is in `runs/posthoc/` and is post-hoc
  and descriptive; the script was not changed afterwards.

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| hotpot_pilot.jsonl | 7461e8af70247612 |
| hotpot_final.jsonl | 370c82dc5321b55f |
| hotpot_final_labels.json | ea03bbce6bf5c375 |
| K_FROZEN_hotpot.json | 7823a2a29015d20a |
| run_webqa.py | a0d5a0dd7ab6a18f |
| analyze.py | ac6b1204c7c52982 |
| common.py | 63c317fdf8aa4835 |
| evidence_log.py | 15062a3aed78a756 |
| answer.txt | 97ab4a638447fbd5 |
| judge.txt | a7f026345d3278e9 |
| revise_mas.txt | d314d03f91f14b16 |
| revise_nocomm.txt | fc92eef7b22dc0d8 |
| synthesize.txt | 4851b401998b21b7 |

`git diff www-freeze` over the scripts and prompts is empty apart from the run wrappers and this
log script.

## Amendment W1 — FRAMES link parsing and snapshot/index gates (2026-10-08, before any FRAMES corpus or generation)

**Found.** A pre-run check of the partial snapshot found a parsing bug in the FRAMES `wiki_links` cells:
- 8 list items join several URLs with ", " (for example "Tim Salmon, https://en.wikipedia.org/wiki/Troy Glaus");
- 1 link is percent-encoded twice.

The frozen `title_of` turned each of these into a single non-existent title, so the API returned it as
missing. This affects 9 questions, 3 of them in the pilot. The fixed parser yields 2,520 linked titles
instead of 2,476.

**Fix** (no FRAMES result of any kind exists yet).
- `fetch_wiki.links_of` splits such items and decodes repeatedly.
- `build_corpus.py` uses `links_of` for gold pages, and reads exactly the manifest's fetched pages
  instead of every cached file.
- The manifest now records fetch errors.
- Because the partition draws owners over the sorted title list, the A/B partition is defined on the
  corrected 2,520-title set.

**New gates before FRAMES generation** (`scripts/verify_snapshot.py`).
- **snapshot:**
  - the manifest title set equals the linked title set;
  - every fetched page has an integer revid with a timestamp ≤ cutoff that matches its cached file;
  - there are zero fetch errors.

  Pages missing at the cutoff, and pages with no body text, are allowed and are counted.
- **index:**
  - all 824 questions have A = 5, B = 5 and U = 10 passages;
  - A and B passages come only from their own source.

One extra fetch pass is allowed, and it only re-tries errors. Otherwise the chain stops.

**Run records.** `scripts/run_record.py` writes `RUN_RECORD_<name>.json` per experiment, with:
- config, seeds, commit and tags;
- question ids;
- model snapshots;
- wall time;
- results.

**Reuse rule (restated).** If any FRAMES setting changes after the pilot, the 200 pilot questions are not
reused as independent final evidence. The final is the disjoint 624.

### Corpus artifacts (built 2026-10-08 22:22 by the frozen build_corpus.py, before any FRAMES generation)
```
pages 2515 (A 1272), passages 103869
ffde624397388435 passages.jsonl
9c96b9bb982a3c0b partition.json
6be168dde90e35d5 retrieval_frames.jsonl
```

## FRAMES pilot decision and final freeze (2026-10-08, before the final run)

**Pilot record.**
- The sealed pilot analysis was run once: commit `c05c913`, `runs/RESULTS_frames_pilot.txt`.
- n = 200, 0 generation errors, k* = 4.
- P-int +7.0 pp [+2.5, +11.5]; P-util −1.5 pp [−7.5, +4.5]; MAS − SC-4 −6.0 pp [−12.5, +0.5].
- These are judge-based, and the 50-item human validation is pending.

**Decision (user): no change.** The final uses unchanged:
- the model and the prompts;
- the BM25 index;
- the Wikipedia snapshot and its revisions;
- the partition seed;
- the judge and the scoring;
- the arms and the bootstrap CI rules.

**Final-set check.** Pilot 200 vs final 624: the intersection is **0** by dataset idx and by question
text (union 824).

**Judge server restart (infrastructure recovery, recorded).**
- On the first pilot attempt the judge server could not start. The private HF cache lacked `refs/main`
  for Qwen3-32B-FP8, so the offline lookup failed. No judge output existed.
- Only `refs/main` was restored, pointing at the existing snapshot `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
  The weights (34.3 GB) and the serve arguments are unchanged.
- The vLLM log confirms it loaded `snapshots/aa55da1…`, with fp8 quantization, `max_model_len` 32768
  and `gpu_memory_utilization` 0.70.
- The judge stage was resumed with `scripts/resume_frames_judge.sh` (commit `338c938`).

**Reporting.**
- Primary: P-int and P-util as defined in §4.
- Key secondary: MAS − SC-4.
- Failure taxonomy and the held / retrieved / transmitted / corrected rates, with explicit denominators,
  from `scripts/evidence_log_frames.py` (descriptive). It was code-tested on the pilot, with the output
  in `runs/posthoc/`.
- The final is kept separately in `runs/final/` and is never pooled with the pilot.
- Judge-based accuracy is not interpreted as definitive until the human validation is done.

**Known limitation found before the final: render-date leak.**
- The revisions are frozen at ≤ 2024-09-19, but `action=parse` expands templates with the *current*
  date. Living persons' "(age N)" and similar values are therefore rendered as of 2026-10.
- Scale:
  - 425 pages carry such a string;
  - 93 of 824 questions retrieve at least one such passage in A, B or U;
  - 28 of the 200 pilot questions are affected.
- The user decided to keep the corpus unchanged. All arms share the corpus. The paper qualifies the
  claim as "frozen revisions, rendered at fetch time".

**Pre-registered sensitivity analysis** (secondary, fixed before the final): `scripts/sensitivity_frames.py`
recomputes P-int, P-util and MAS − SC-k, using the same bootstrap, after excluding the affected
questions. It was code-tested on the pilot (`runs/posthoc/SENSITIVITY_frames_pilot.txt`).

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| frames_split.json | 4a66dab601f75c3c |
| wiki_manifest.jsonl | 4436288686210fbe |
| passages.jsonl | ffde624397388435 |
| partition.json | 9c96b9bb982a3c0b |
| retrieval_frames.jsonl | 6be168dde90e35d5 |
| K_FROZEN_frames.json | 6230b94af157b358 |
| run_webqa.py | a0d5a0dd7ab6a18f |
| analyze.py | ac6b1204c7c52982 |
| common.py | 63c317fdf8aa4835 |
| judge.py | 826e6d124088866d |
| build_corpus.py | da41d9aa6eae674e |
| fetch_wiki.py | 313369c2371dfd11 |
| evidence_log_frames.py | a5a4722247ff0a00 |
| sensitivity_frames.py | 3c616850eb693458 |
| prompts | identical to the HotpotQA freeze |

The passages, partition and retrieval hashes equal those in `logs/build_corpus.log`. The generation,
scoring and corpus code is unchanged since `frames-w1-freeze`.

**Note (recorded after launch; it does not change the frozen run).**
- The snapshot check lists 5 linked titles as missing.
- Two of them, "Alex Hernández (tennis)," and "Globe Life Field,", are residues of the W1 split: trailing
  commas in FRAMES cells. Each affects one final question (idx 540 and 90), which loses one gold page.
- The other three are links that do not resolve at the cutoff (idx 540, 141 and 644).
- All four questions are in the final, and each still has at least one gold page in the snapshot.
- This is reported as a limitation. The frozen corpus is not changed.

## Experiment W2 — evidence distribution on FRAMES (2026-10-09; design decided by the user before any W2 run)

**Question.** With the questions, model, prompts, combiner and seeds fixed, does the communication effect
(MAS − No-comm) depend on whether the two agents hold *different* evidence or the *same* evidence?

**Conditions.**
- **D (Distributed):** the existing FRAMES final run, reused unchanged. A holds its source's top-5 and B
  holds its source's top-5.
- **S_UNION (primary Shared):** A and B both hold the same 10 passages, A top-5 ∪ B top-5. The team's
  union of evidence equals D; only the distribution differs. Per-agent evidence is doubled, which is a
  known confound.
- **S_5 (control):** A and B both hold the pooled BM25 top-5 (U[:5]). Per-agent evidence equals D; the
  union is smaller.
- **S_UNION_T7 (secondary):** as S_UNION, but round 0 is sampled at T = 0.7, so that two homogeneous agents
  can disagree (the P0 analogue). Revision, synthesizer and judge stay at T = 0. Its contrast with D also
  differs in round-0 temperature.

**Arms per condition:** A0, B0, A1m, B1m, A1n, B1n, MAS, No-comm. Pooled and SC come from the D run.
- Templates are unchanged.
- In Shared conditions the MAS partner block repeats the same passages, because the partner holds the same
  evidence.
- Partner passage ids start after the agent's own passages. With D's layout this is P6–P10, identical to
  `run_webqa.py`.
- **Verified:** `run_distribution.py --cond D` on 20 final questions reproduced all 8 roles of the frozen
  final run from the cache, with no server running.

**Questions.**
- **Pilot 200:** execution check only, label-free and fixed in advance (`check_distribution.py`). It needs,
  in every condition:
  - 0 errors;
  - answers parsed in ≥ 95% of calls;
  - ≤ 2% of calls truncated;
  - pid layout as specified;
  - input < 32,768 tokens.

  Nothing is tuned on it; on FAIL the chain stops.
- **Final 624:** the analysis set. It is the same questions as the D run.

**Sealed analysis** (`analyze_distribution.py`, run once). Judge-based; the judge was human-validated with
agreement 0.98 and κ 0.959. Paired bootstrap over questions: 10,000 resamples, seed 0, one shared index
matrix.
- **PRIMARY:** DiD(S_UNION) = [MAS − No-comm]_D − [MAS − No-comm]_S_UNION.
- **Secondary:**
  - DiD(S_5) and DiD(S_UNION_T7);
  - P-int within each condition;
  - MAS_cond − MAS_D and No-comm_cond − No-comm_D;
  - MAS_cond − Pooled;
  - round-0 answer agreement;
  - κ0 → κ_MAS;
  - BRR / HRR on discordant pairs;
  - tokens.
- **Reading, fixed now:**
  - the DiD 95% CI lies above 0 → distribution moderates the communication effect;
  - it lies below 0 → reversed;
  - otherwise → not established.
- D's P-int is recomputed within this analysis, with a different resample matrix. The reported FRAMES
  final numbers remain those in `runs/final/`.

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| run_distribution.py | 0dd4633fe9ec5ee5 |
| check_distribution.py | de799fd96b9639b5 |
| analyze_distribution.py | 871d566a5f325589 |
| run_distribution.sh | 9cf2bc8ed31a7d8b |

The other code, prompts and the corpus are unchanged since `frames-final-freeze`.

## Experiment W3 — backbone replication (W3b) and HotpotQA evidence distribution (W3a) (2026-10-09, user request, before any W3 run)

### W3a — HotpotQA W2

Same design as W2, on the HotpotQA final 1,000 with Qwen2.5-14B. The Distributed condition is the existing
HotpotQA final run.

**Conditions.**
- **S_UNION:** A and B both hold all 10 paragraphs.
- **S_5:** A and B both hold A's 5 paragraphs. HotpotQA has no retrieval, so this is the per-agent-matched
  control.
- **S_UNION_T7:** as S_UNION, with round 0 at T = 0.7.

**Scoring and estimands.**
- Scoring is EM against the frozen labels.
- PRIMARY: DiD(S_UNION). The secondaries, the bootstrap and the reading rule are as in W2
  (`analyze_dist_hotpot.py`).
- **Verified:** `run_dist_hotpot.py --cond D` reproduced all 8 roles of the frozen HotpotQA final run for
  20 questions, from the cache only. The analysis reproduces the sealed P-int (+18.6 pp).

### W3b — backbone replication on FRAMES final 624

**Backbones.** They are copied from a shared HF cache into a private cache, run in BF16, and
reported separately. None is dropped after the fact.

| Backbone | Snapshot | GPU memory utilization |
|---|---|---|
| Qwen2.5-7B-Instruct | a09a354 | 0.70 |
| Qwen2.5-32B-Instruct | 5ede1c9 | 0.92 |
| Llama-3.1-8B-Instruct | 0e9e39f | 0.70 |
| Mistral-Small-24B-Instruct-2501 | 9527884 | 0.85 |

For Mistral, the duplicate consolidated.safetensors is excluded, so the HF shards from the index are used.

**Protocol per backbone.** The prompts, seeds and corpus are frozen.
1. Smoke on 5 pilot questions.
2. Token probe on 20 pilot questions (tokens only, `token_probe_bb.py`; same formula, written to the
   backbone directory).
3. k* for that backbone.
4. FRAMES final 624 with all arms (`run_webqa.py`).
5. S_UNION final 624 (`run_distribution.py`).
6. Judging with Qwen3-32B-FP8 (aa55da1), the frozen judge.
7. The full sealed `analyze.py`, then `analyze_backbone.py`.

**Estimands.**
- PRIMARY, per backbone: P-int = MAS − No-comm, and DiD(S_UNION).
- Secondary: P-util, MAS − SC-k* with its token ratio, accuracies, and label-free execution statistics
  (parse rate, truncation). The execution statistics are reported and never used to exclude a backbone.

**Reading, fixed now.**
- **Replicates:** both primary 95% CIs lie above 0.
- **Partially replicates:** exactly one does.
- **Does not replicate:** otherwise.

**Known caveats, stated in advance.**
- The judge (Qwen3) and the Qwen2.5 backbones are in the same model family.
- The judge was human-validated only on Qwen2.5-14B pilot outputs.
- Weaker backbones may parse answers less often. This is reported, not corrected.

**Infrastructure.**
- Two GPU queues, each with its own fixed-GPU 10-minute free-memory gate:
  - GPU0: Qwen2.5-32B, then Llama-3.1-8B;
  - GPU1: Mistral-24B, then Qwen2.5-7B, then W3a.
- After both queues finish, one judge stage runs.
- No retries.

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| token_probe_bb.py | 38dd6eb83715ec8f |
| run_dist_hotpot.py | 7667a4ee54bd4865 |
| analyze_dist_hotpot.py | 0ab91f5095684d39 |
| analyze_backbone.py | 96c7bafcf61faa81 |
| run_w3_queue.sh | 15fc4846a76da21a |
| run_w3_judge.sh | 487f0161a93e4365 |

All earlier code, prompts and the corpus are unchanged.

**W3 infrastructure amendment I1 (2026-10-09 13:39, before any Qwen2.5-32B output).**
- **Failure.** The Qwen2.5-32B server failed to start on one GPU. vLLM needs 8.0 GiB of KV cache for one
  32,768-token sequence, and only 6.0 GiB was available beside the 62 GB of BF16 weights.
- **Fix.** It is now served with tensor parallelism over both GPUs (`scripts/run_w3_tp2.sh`, utilization
  0.85). The run starts after both single-GPU queues finish, and only when both GPUs have ≥ 70 GB free
  for 10 minutes.
- **Unchanged.** The protocol, prompts, seeds, arms, k\* probe, max_model_len and dtype are unchanged.
- **New order.**
  - GPU0 queue: Llama-3.1-8B.
  - GPU1 queue: unchanged.
  - Then Qwen2.5-32B (TP = 2), then the judge stage.
- **Judge stage.** It waits for all three runs. The first judge process had left its wait loop when it
  saw the 32B failure. It was stopped in the GPU gate, before any judge server started or any output
  was judged.
- **Logs.** The failed attempt's logs are kept as `logs/*_attempt1.log`.

**W3 post-hoc sensitivity S1: Llama-3.1-8B with repetition penalty** (2026-10-09 14:55; user decision after the
label-free execution statistics; no Llama accuracy had been seen).
- **Observation.** In the frozen Llama run (k\* = 5), greedy decoding fell into repetition loops in 28–42% of
  calls, which then hit the 1,024-token limit. Answers parsed in 56–71% of calls, depending on the role.
- **Reporting of the frozen run.** It stays the pre-registered Llama result and is reported unchanged.
- **Additional run.** `scripts/run_w3_llama_rp.sh` is reported separately as POST-HOC SENSITIVITY.
  - Only change: a server-side `repetition_penalty` 1.1, set with `--override-generation-config`.
  - Unchanged: prompts, seeds and arms; k\* = 5, the frozen Llama value, not re-probed.
  - It uses a separate LLM cache, because the cache key does not contain server defaults.
  - Same judge and the same analysis scripts, under the label `llama3.1-8b_rp1.1`.
  - It runs on GPU0 between the Llama queue and the TP = 2 run, and is judged after the main judge stage.

## Experiment W4 — evidence-overlap intervention (2026-10-09; design chosen by the user before any W4 run)

**Design.** FRAMES final 624, Qwen2.5-14B. Each agent always holds 5 passages, the same token budget.
- With o shared passages:
  - A gets A[:5−⌊o/2⌋] + B[:⌊o/2⌋];
  - B gets B[:5−⌈o/2⌉] + A[:⌈o/2⌉].
- So |A ∩ B| = o and |A ∪ B| = 10 − o.
- o = 0 is the frozen D run. Verified: `run_overlap.py --o 0` reproduced all 8 roles for 20 questions from the
  cache alone.
- New runs: o = 1..5. Prompts, roles, combiner and seeds are as in W2.

**Analysis.**
- PRIMARY: the OLS slope of P-int(o) over o = 0..5. Paired bootstrap, 10,000 resamples, seed 0, recomputing
  all six values in each resample.
- Secondary: P-int(o) for each o, P-int(0) − P-int(5), and the slopes of MAS and No-comm.
- Descriptive covariates:
  - per-agent and team gold coverage (the team's union shrinks to 10 − o);
  - round-0 accuracy and answer agreement;
  - synthesizer shares.
- **Reading, fixed now:**
  - slope CI below 0 → the gain *decreases* with overlap;
  - slope CI above 0 → it increases;
  - otherwise → not established.
- A negative slope is not attributed to overlap alone, because union coverage co-varies with o.

**Infrastructure.** The run starts after all of W3. Two agent servers run in parallel (GPU0: o = 1, 3, 5;
GPU1: o = 2, 4), followed by one judge and the sealed `analyze_overlap.py`, run once.

## W5 — final-cohort judge audit (blind; sampling fixed before any annotation)

**Sample.** Drawn from the FRAMES final with seed 2027, using judge-discordant questions and disjoint
questions per stratum:
- 20 MAS–No-comm (91 discordant in all);
- 10 MAS–Pooled (98);
- 10 MAS–SC (103).

Both policies' responses become blind items: 80 in total, in `data/judge_audit_final_sheet.xlsx`.

**Analysis** (`judge_audit_analyze.py`).
- Judge–human agreement and κ.
- Per stratum, the human-adjusted full-cohort difference:
  Δ_judge − (N_s / 624)·(gap_judge − gap_human) on the sample, bootstrapped over the sampled questions.
- **Stated assumption:** judge-concordant questions keep their ordering.

## Post-hoc exploratory analyses (not pre-registered; logs only)

- `runs/posthoc/EXPLORE_W2.txt`: per-condition evidence, gold coverage, round-0 behaviour and combiner path.
- `runs/posthoc/EXPLORE_AGGREGATION.txt`: P-int = revision component + aggregation component.
- The "same-initial-state" ablation that was suggested is already the design. MAS and No-comm branch from
  identical cached round-0 answers in every condition, so P-int *is* that contrast.

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| run_overlap.py | 80adbe7dbe254f53 |
| analyze_overlap.py | 29656ceeed58b61e |
| run_overlap.sh | 1a2ee21b89b82974 |
| judge_audit_final.py | 223f3693deee3015 |
| judge_audit_analyze.py | b246dbeb626f566f |

**Incident (2026-10-09 17:30, infrastructure only).**
- **What happened.** The S1 judge server died at start-up with OOM. GPU0 had just been taken by the W4
  agent server: `run_overlap.sh` waits for "DONE|FAIL" in the S1 log, and it matched S1's intermediate
  line "generation DONE". Both then gated on GPU0.
- **Damage.** No S1 output had been judged. W4 is unaffected and continues.
- **Recovery.** `scripts/resume_llama_rp_judge.sh` runs the S1 judge stage after W4 finishes, with the
  same judge, the same scripts and the same cache. The S1 log's last line will be "SENSITIVITY COMPLETE".

## Experiments E1–E3 (2026-10-09 evening; designs chosen by the user; frozen before any E1–E3 generation)

All three use Qwen2.5-14B, the frozen judge and the frozen combiner. They reuse the FRAMES final questions, which
were already analysed in W2–W4. The paper states that several pre-registered contrasts share this question set.

**E1 — causal evidence-transfer intervention** (FRAMES final 624 and HotpotQA final 1,000).
- **Setup.** The D layout and the frozen round-0 answers, read from the frozen D runs. Only the revision input
  changes.
- **Templates.** `prompts/revise_e1_*.txt` are the MAS template with blocks deleted (verified by diff).
- **Modes:**
  - **ANS:** the partner's answer only;
  - **ANS_RSN:** answer + reasoning;
  - **ANS_EV:** answer + genuine passages;
  - **ANS_IRR:** answer + the partner passages of the *next* question (sorted, cyclic);
  - **EV:** passages only.
- **Reused:** NC (frozen No-comm) and FULL (frozen MAS).
- **PRIMARY:**
  - C−B = ANS_EV − ANS;
  - C−D = ANS_EV − ANS_IRR.
- **Secondary:**
  - each mode − NC;
  - ANS_RSN − ANS, FULL − ANS_EV, EV − ANS_EV;
  - BRR, HRR and partner-answer adoption on the round-0 discordant pairs, which are identical across modes.
- **Reading:**
  - C−B CI above 0 → "genuine evidence adds beyond the partner's answer";
  - C−D CI above 0 → "the gain is sensitive to evidence relevance";
  - both → "evidence-transfer sensitive".

**E2 — overlap × gold placement** (FRAMES final; 427/624 eligible: at least 1 gold passage and at least 13 fillers
in the pooled BM25 top-30 ∪ A ∪ B).
- **Gold set.** G = up to 2 gold-page passages, preferring different pages. Gold *pages* come from the dataset's
  links, not from the answer.
- **Cells (2 × 2).** Each agent holds 5 passages, and the team's gold content is identical in all four cells.

  | Cell | Gold placement | Shared fillers | Shared passages, total |
  |---|---|---|---|
  | EXCL_LOW | one agent only | 0 | 0 |
  | EXCL_HIGH | one agent only | 3 | 3 |
  | SHAR_LOW | both agents | 0 | g |
  | SHAR_HIGH | both agents | 3 | g + 3 |

  The exclusive holder alternates by idx parity, and each list is shuffled with seed 2027 + idx.
- **PRIMARY:**
  - the placement effect, mean over overlap of [P-int(EXCL) − P-int(SHAR)];
  - the overlap effect, mean over placement of [P-int(LOW) − P-int(HIGH)].
- **Secondary:** the interaction and the per-cell values.
- **Reading:** a CI above 0 means "partner-exclusive gold drives the gain" (placement) or "overlap reduces the gain
  with gold content fixed" (overlap). The design is gold-aware: a controlled manipulation, not a deployable policy.

**E3 — topic-partitioned (decentralized) retrieval** (FRAMES final 624).
- **Partition.** Wikidata P31 contains Q5 → PEOPLE (769 pages, 30,222 passages); otherwise WORLD (1,746 pages,
  73,647 passages). These are current Wikidata types; they define the partition only.
- **Retrieval.** Per-domain BM25 top-5. The pooled index is unchanged, so the frozen Pooled and SC-4 arms are reused.
- **Arms:** Distributed MAS, Distributed No-comm, Centralized single (Pooled) and token-matched SC-4.
- **PRIMARY:** P-int and P-util.
- **Secondary:** MAS − SC-4; the DiD of topic vs the random partition; the share of questions whose gold spans both
  domains; held/retrieved availability.

**Execution.**
- Two queues:
  - GPU0: E1 FRAMES ×5, E3, E2 SHAR_LOW, E2 SHAR_HIGH;
  - GPU1: E1 HotpotQA ×5 with its EM analysis, E2 EXCL_LOW, E2 EXCL_HIGH.
- Each job runs a 5-question label-free smoke first.
- One judge stage then runs and waits for the exact marker "QUEUE COMPLETE" in both queues (fixing the W4/S1 race).
- No retries.
- The analyses were code-tested on stand-in copies of frozen outputs, which were then deleted.

**Frozen hashes (sha256, first 16 hex characters).**

| File | Hash |
|---|---|
| prompts/revise_e1_ans.txt | 40feb8f77db3d218 |
| prompts/revise_e1_ansev.txt | 9791670451fc46af |
| prompts/revise_e1_ansrsn.txt | 5ddac7e32642513f |
| prompts/revise_e1_ev.txt | 40ca74fceb86509a |
| scripts/run_e1.py | cfcdeeaeae4f4084 |
| scripts/build_e2.py | 7585ec28ecded64c |
| scripts/fetch_topics.py | 4c82896ad31f0ae4 |
| scripts/build_e3.py | c781404ce145fa02 |
| scripts/run_e23.py | fd6c43dec9353d12 |
| scripts/analyze_e123.py | db73cfd3da00541d |
| scripts/run_e123_queue.sh | 572d3dae55fac31c |
| scripts/run_e123_judge.sh | beffc9a5f3ca1fb7 |
| data/e2_assign.jsonl | 37187e6ae631d490 |
| data/topic_partition.json | d752fcb12bd27c4d |
| data/retrieval_topic.jsonl | 9447183e5fce5f7a |
