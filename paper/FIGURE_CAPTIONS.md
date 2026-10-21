# Figure captions (drafts)

**Figure 1. Evaluation framework.** Two agents each own one document collection. Evidence can reach an agent by
local access (gray), by peer evidence transfer between the two agents (blue, both directions), or by remote retrieval, where the agent
queries the other collection and receives results (orange; dashed = query, solid = results). The three message types are
alternative conditions, not sent together: answer (± reasoning), evidence passages, or answer + evidence. Each agent revises
its answer (① candidate generation), and a combiner returns the team answer (② candidate preservation). We report
accuracy and each cost dimension separately: LLM calls, input/output tokens, retrieval calls and measured latency.
Receiving evidence does not guarantee integrating it, and integrating it does not guarantee that the correct answer
survives aggregation.

**Figure 2. Supporting-evidence placement and integration (HotpotQA, n = 1000).** (A) Team accuracy of MAS and
No-comm when 0, 1 or 2 of the 2 supporting paragraphs are available in the team, by placement (exclusive: one agent
only; shared: both agents; split: one each; concentrated: both with one agent). Error bars are question-level
bootstrap 95% CIs of each accuracy. The numbers above the bars are the communication gain (MAS − No-comm, pp); the
error bars are not CIs of that difference (see Table X for paired CIs). (B) Outcome of every question in the split
and concentrated cells: each cell's 1,000 questions fall into exactly one of four mutually exclusive categories, based on the two revised answers and the final answer:
correct and preserved; lost in synthesis (a revised answer was correct, the final answer is not); created in
synthesis (no revised answer was correct, the final answer is); no correct revised/final answer (neither revised
answer nor the final answer is correct). Final accuracy = preserved +
created (split 49.6 + 0.0; concentrated 58.8 + 0.1 = 58.9). Communication gains are similar (+16.5 vs +16.1 pp), but
final accuracy differs by 9.3 pp, of which 6.0 pp is candidate absence and 3.3 pp is loss in synthesis.

**Figure 3. Accuracy versus token cost under different evidence-access routes.** Each point is one system: accuracy
against mean tokens per question (input + output, all LLM calls). Vertical lines are question-level bootstrap 95% CIs
of each system's accuracy. These are not the paired CIs of the B4 contrasts, so overlapping lines do not imply that
two systems are equivalent. Hollow markers are self-consistency (k = 4): gray = local evidence, green = federated/pooled
evidence. Only UNI-ONE and UNI-SC are plotted for federated/pooled access; POOL (monolithic index) and SC are omitted, not
averaged in. On HotpotQA they are the identical calls (same accuracy and tokens). On FRAMES they differ slightly:
POOL 35.3% vs UNI-ONE 35.1%, SC 37.7% vs UNI-SC 38.5% (full values in Table X). Y-axis ranges differ
between panels; compare systems within a dataset, not effect sizes across datasets. Direct-Q / Direct-Gen = a single
agent retrieving the top-5 from the other collection with the question / with its own query; Peer-EV = the partner's
top-5 passages as evidence-only messages; MAS = full messages (answer, reasoning, evidence).
