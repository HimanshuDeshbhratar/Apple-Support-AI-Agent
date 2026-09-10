# Decision log

Non-obvious choices and why.

1. **Brand = AppleSupport** — Largest, noisy, well-known support voice in the corpus; abundant update/battery/connectivity patterns; Apple-filtered HF mirror made a clean subsample without Kaggle credentials.

2. **Use HF `OpenArchive/AppleConvos` instead of raw Kaggle CSV** — Same upstream dataset, already paired `customer → agent`, ~106k rows, downloadable in seconds. Cited as a mirror, not a different problem.

3. **Subsample ~8k retrieval pairs, not full 3M** — Assignment encourages subsample; full Twitter CSV is unnecessary for a 15-minute reproduce path and would dominate runtime/disk.

4. **11 intents, not Banking77-style 77** — Twitter Apple issues cluster into coarse ops buckets; 77 labels would be sparse and overfit weak supervision. Banking77 kept optional and unused for the headline path.

5. **Rubric labels (priority regex) instead of unsupervised clustering alone** — Clustering on noisy tweets produced unstable names; a written priority rubric is auditable and matches how we’d brief human labellers.

6. **Train sklearn on the same rubric used for gold** — Avoids training on a different definition than we evaluate. Honest limitation called out in REPORT §5: this measures rubric agreement, not Apple-employee truth.

7. **Hold out golden candidates from the retrieval index** — Prevents nearest-neighbor leakage that would inflate simple/agent reply metrics.

8. **TF-IDF retrieval over dense embeddings** — No GPU, fast cold start, deterministic, good enough lexical match for short tweets; sentence-transformers would blow the 15-minute setup for little gain here.

9. **Calibrated LinearSVC for intent** — Strong classic baseline for short text; calibration gives usable confidence for escalation gates without neural training loops.

10. **Deterministic grounded drafts by default; LLM optional** — Reproducible headline results without API keys; LLM can polish but must not be required to “pass.”

11. **Escalate billing + privacy always; not all hardware** — Early policy escalated all hardware and crushed precision; hardware often needs a normal diagnostic first tweet.

12. **Simple baseline = neighbor intent, not the gold rubric** — Using `refine_intent` for the simple baseline made intent accuracy ~1.0 and was circular. Neighbor intent is the honest “retrieval-only” control.

13. **Reply quality ≠ string match to historical agent text** — Apple replies are procedural (“DM us”); exact match would reward link-dropping, not helpfulness.

14. **Separate human judge panel (n=36) from golden intent/action labels** — Needed because the intrinsic auto-judge shared DNA with reference quality bands and reported fake 100% agreement until we judged *agent drafts* for issue-specific helpfulness.

15. **Ship offline metrics as headline; document gameability** — The assignment’s hard part is trust. Exposing that trivial cans beat the agent on reply score is intentional, not an accident to hide.
