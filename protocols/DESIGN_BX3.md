# BX3 — post-hoc execution robustness audit (limited regeneration)

Frozen (git tag `bx3-freeze`) on 2026-10-19 before any BX3 generation. The original evaluation results (RX / BX / BX2)
are the primary results and are NOT replaced; BX3 is reported as a *post-hoc execution robustness audit*. No run is
selected, dropped, or averaged into the primary results after outputs are seen.

## What is regenerated

Exactly the arms of the key contrasts, on the full original evaluation sets, with the original question ids, model
snapshot (Qwen2.5-14B-Instruct cf98f3b), prompts, decoding (T = 0, max 1024 tokens, same seeds) and concurrency
(48 workers, as the original runs). Every generation job uses its own FRESH, EMPTY cache directory (its 5-question smoke
run uses a separate one), so every call reaches the server; calls are not shared across jobs. Each design's own frozen
inputs stay as designed: revision-only arms reuse the frozen round-0 answers (that is the arm's definition); B3 cells
regenerate their full pipeline including round 0.

| contrast | dataset | regenerated arms |
|---|---|---|
| R1 gold removal (FULL − GOLDREP, team final) | FRAMES, HotpotQA | R1_FULL, R1_GOLDREP |
| B3 placement: P-int(F_SPLIT) − P-int(F_SHAR) | HotpotQA | B3_F_SPLIT, B3_F_SHAR |
| B4-P1 framing (CQ_Q_5 − PEER_EV_5, agent level) | FRAMES, HotpotQA | PEER_EV_5, CQ_Q_5 |
| B4-P2 k=5 targeting (CQ_GEN_5 − PEER_EV_5, agent level) | FRAMES, HotpotQA | CQ_GEN_5 (+ PEER_EV_5 above) |

Note: on FRAMES, R1_FULL and PEER_EV_5 have identical payloads; they are regenerated as two independent jobs (two
independent regenerations of the same arm), and each is used only in its own contrast.

Scoring: FRAMES with the frozen judge exactly as before (judge prompt, model aa55da1, T = 0, the shared judge cache, so
an answer string judged before receives the identical verdict); HotpotQA EM. Execution record: vLLM in-flight request
counts sampled every 5 s during each job (exact batch composition is not observable through the API).

## What is reported (fixed now)

For every contrast: original point estimate and 95 % CI (as published in RX/BX), regenerated point estimate and 95 % CI
(same bootstrap: question-level paired, 10,000, seed 0; agent-level contrasts cluster by question), and whether the
direction and the significance verdict are unchanged. For B4-P1 additionally the ±2 pp equivalence verdict (95 % CI
and TOST 90 % CI) in the regeneration, reported SEPARATELY from the fixed-execution equivalence test. Answer-level
agreement between original and regenerated runs per arm. One regeneration is one additional execution: it shows
sensitivity to execution, not invariance across all execution conditions.

## Files

`scripts/bx3_queue.sh`, `scripts/bx3_analyze.py`; outputs `runs/bx3/`.
