# Report — Apple Support AI Agent

## 1. Problem framing

**Brand:** AppleSupport (Twitter customer support).  
**Job:** first-response agent that (1) classifies intent, (2) drafts a reply grounded in how Apple historically handled similar tweets, (3) decides auto-handle vs escalate with a reason.

**What “good” means for this brand**
- Fast triage into a small, operational intent set (not 77 banking-fine intents).
- Replies that sound like Apple’s public Twitter posture: empathy + diagnostics + DM for private details — **without inventing refunds, warranties, or legal outcomes**.
- Escalate when a wrong auto-reply is costly: billing disputes, theft/hack claims, account lockouts, legal/fraud language.

**What we chose not to build**
- Full multi-turn dialogue manager / ticket CRM.
- Fine-tuned generative LLM as the only path (API optional; offline path must reproduce).
- Exact-match reply scoring against historical tweets (agents paraphrase; style ≠ gold string).
- End-to-end automation of refunds or device replacements.

Proof > system: the deliverable is an evaluation story, not a chatbot UI.

## 2. Method (brief)

- Data: Apple-filtered pairs from Customer Support on Twitter via Hugging Face `OpenArchive/AppleConvos` (~106k). Subsample ~8k for retrieval; hold out candidates for eval.
- Intents: 11 classes from observed Apple themes (battery, iOS regressions, connectivity, Apple ID/iCloud, billing, privacy, …).
- Intent model: TF-IDF + calibrated LinearSVC on rubric labels.
- Reply: TF-IDF retrieval of similar customer→agent pairs, then intent-conditioned draft adapted from retrieved evidence.
- Escalation: sensitive intents + keyword risks + low confidence / weak match.
- Golden set: 200 held-out examples labelled with an explicit rubric (`data/golden/LABELING_NOTES.md`).
- Baselines: *trivial* canned policy; *simple* nearest-neighbor copy + neighbor intent.

## 3. Results vs baselines

From `results/RESULTS.md` (offline, no API key):

| System | Intent Acc | Intent Macro-F1 | Action Acc | Escalate F1 | Reply Score |
|---|---:|---:|---:|---:|---:|
| trivial | 0.090 | 0.015 | 0.730 | 0.00 | 0.978 |
| simple_retrieval | 0.445 | 0.457 | 0.775 | 0.49 | 0.707 |
| **agent** | **0.825** | **0.825** | **0.820** | **0.65** | **0.858** |

**Reading**
- Intent: agent ≫ simple ≫ trivial. Simple only copies the nearest neighbor’s weak label; lexical neighbors are often cross-intent (update threads dominate Apple Twitter).
- Action: agent slightly edges simple; both beat trivial (which never escalates and looks “accurate” only because most gold labels are `auto_handle`).
- Reply score: agent beats simple (cleaner, intent-aware drafts vs raw historical paste). Trivial scores *highest* — see §5.

**Judge agreement:** on 36 independently human-labelled agent drafts, exact-match with the automated reply judge is about **0.39**. The auto judge over-rates generic empathy + diagnostic templates; humans penalize mismatched substance (Mac issue asked for iOS version, MDM policy asked for device model, etc.). Optional `--llm-judge` is available when an API key is set.

## 4. Failure analysis (top 5 modes)

Examples drawn from `results/failure_candidates.jsonl` / prediction diffs.

### F1 — Multi-intent tweets collapsed to one label
**Example:** charging fails *after* an iOS update. Gold may prefer `battery_charging` or `ios_update_bugs` depending on priority rules; the classifier flips.  
**Hypothesis:** single-label taxonomy is under-specified for Apple’s “update caused X” pattern. Hierarchical or multi-label intents would help.

### F2 — Frustration / profanity without clear issue → `other` + missed escalate
**Example:** customer vents about a $1k phone with no actionable symptom. Model predicts `other` and may auto-handle; gold wants escalate on billing/complaint cues.  
**Hypothesis:** escalation features under-weight pure outrage + price mentions when intent is unclear.

### F3 — Neighbor retrieval pulls DM-heavy Apple templates into the wrong product
**Example:** Mac mini / High Sierra crash → draft still talks like iPhone/iOS Twitter support.  
**Hypothesis:** corpus is iPhone-skewed; no device-type gate in retrieval.

### F4 — Gold escalate on “severe tone” vs agent auto-handle on fixable bugs
**Example:** Wi-Fi auto-enable annoyance after 11.0.3. Intent correct, but gold escalates on complaint tone while agent treats as routine connectivity.  
**Hypothesis:** tone-based gold labels are stricter than the production policy we encoded; policy mismatch, not pure model error.

### F5 — Mid-thread tweets lack context
**Example:** “Just on the home screen for now.” Without prior turns, intent/hardware guesses are noise.  
**Hypothesis:** we scored single tweets; Apple’s real workflow is thread-aware. Thread reconstruction was de-scoped.

## 5. What is misleading about my headline number?

**Reply score is the most misleading headline.**

1. **Trivial baseline scores ~0.98** because the canned sentence is engineered to hit the same surface rubric (thanks/help + ask model/iOS + good length). That does **not** mean canned replies are better support — only that the cheap judge is gameable.
2. **Human↔auto agreement ≈ 0.39** on real agent drafts. When humans judge *issue-specific* helpfulness, many “good” auto scores become `ok`/`weak`. So **0.86 agent reply score ≠ 86% human-good replies**.
3. **Action accuracy is inflated by class imbalance** (~75% gold `auto_handle`). A never-escalate bot still looks decent on accuracy; escalate F1 is the more honest ops metric (~0.65 for the agent).
4. **Intent accuracy uses a rubric that also generated training labels.** It is consistent and held-out by *example*, but it is not an independently panel-labelled taxonomy from professional Apple agents. Treat 0.83 as “agreement with our operational rubric,” not ground truth from Apple.
5. **Historical Apple replies are not gold drafts.** Many are “join us in DM” with a link. Overlap with history can reward DM boilerplate.

If forced to quote one trust number for “should we ship a first responder?”, prefer **escalate F1 + human spot-check pass rate**, not reply score.

## 6. What I’d do with one more week

1. Rebuild threads (`in_response_to_tweet_id`) and evaluate on **conversation state**, not single tweets.
2. Replace/augment the reply judge with a **blind human panel** (n≥100) and calibrate LLM-as-judge to that panel (Cohen’s κ).
3. Add a lightweight **device router** (iPhone / iPad / Mac / Watch) before retrieval.
4. Multi-label intents for “update caused battery/connectivity” bundles.
5. Active learning: sample model disagreements with the rubric for a second human pass on the golden set.
6. Online shadow mode metrics: agent suggestion acceptance rate by real agents (even a tiny pilot).

## 7. Reproducibility

`python scripts/run_all.py` downloads the Apple subset (if needed), rebuilds subsample/models/golden metrics, and writes `results/`. Wall clock on a modern laptop is typically **under 2 minutes** after dependency install (well under the 15-minute bar).
