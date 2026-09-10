# Apple Support AI Agent (Hiver SDE Intern Take-Home)

Turn messy Twitter support threads into a **trustworthy** first-response agent for **AppleSupport**: classify intent, draft a grounded reply, and decide auto-handle vs escalate — then **prove** it with a golden set, baselines, and a judge-agreement study.

## What "good" means here

For Apple Support on Twitter, a good first response usually:
1. Names the right issue family (battery vs billing vs update, etc.).
2. Asks for the missing diagnostic detail *or* moves sensitive cases to DM.
3. Escalates legal/fraud/account-takeover/billing/privacy instead of inventing policy.

This repo optimizes for **trust under evaluation**, not a full multi-turn helpdesk.

## Quick start (reproduce headline results in <15 min)

```bash
# from repo root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_all.py
```

Then open:
- `results/RESULTS.md` — headline table
- `results/metrics.json` — full metrics
- `REPORT.md` — analysis
- `DECISION_LOG.md` — non-obvious decisions

**Optional LLM mode** (drafting + LLM-as-judge):

```bash
copy .env.example .env   # add OPENAI_API_KEY
python -m src.evaluate --use-llm-agent --llm-judge
```

Default path needs **no API key** (TF-IDF retrieval + calibrated LinearSVC + rule-grounded drafts).

## Demo one message

```bash
python -m src.demo --text "@AppleSupport my iPhone 7 on iOS 11 drains overnight and gets hot while charging"
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| Download Apple pairs | `python scripts/download_data.py` | `data/raw/train.csv` |
| Clean + subsample | `python -m src.prepare_data` | `data/processed/apple_pairs.csv` |
| Train intent + index | `python -m src.train` | `*.joblib` |
| Build golden set | `python scripts/build_golden_set.py` | `data/golden/golden_set.jsonl` |
| Evaluate + baselines | `python -m src.evaluate` | `results/` |

## Dataset

- **Primary source:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Thought Vector / Stuart Axelbrooke).
- **Mirror used (Apple-filtered pairs):** [`OpenArchive/AppleConvos`](https://huggingface.co/datasets/OpenArchive/AppleConvos) on Hugging Face (~106k `text_input` / `text_response` pairs).
- We **subsample ~8k** retrieval pairs + **400** holdout candidates; golden eval uses **200** held-out labelled examples (retrieval index excludes them).

## System design (short)

1. **Intent (11 classes):** TF-IDF + calibrated `LinearSVC`, trained on a documented priority rubric (`src/labeling.py`).
2. **Reply:** retrieve top historical Apple replies (TF-IDF cosine), then draft an intent-conditioned reply grounded in that evidence (optional LLM rewrite if `OPENAI_API_KEY` is set).
3. **Escalation:** keyword risk + sensitive intents + weak retrieval/confidence → `escalate` with a written reason.

## Evaluation

- **Golden set:** 200 rubric-labelled examples — see `data/golden/LABELING_NOTES.md`.
- **Baselines:**
  - *Trivial:* majority-ish canned reply, never escalates, constant intent.
  - *Simple:* nearest-neighbor historical reply + neighbor intent + shared escalation helper.
- **Reply judge:** deterministic rubric (+ optional LLM judge). **Human agreement** on 36 agent drafts lives in `data/golden/judge_agreement.jsonl`.

## Headline results (default offline run)

See `results/RESULTS.md` after `run_all.py`. Typical shape:

| System | Intent Acc | Action Acc | Reply Score |
|---|---:|---:|---:|
| trivial | ~0.09 | ~0.73 | high (misleading) |
| simple_retrieval | ~0.45 | ~0.78 | mid |
| **agent** | **~0.82–0.85** | **~0.82** | **high but judge-disagrees often** |

Read **REPORT.md → “What is misleading about my headline number?”** before trusting reply score.

## Project layout

```
configs/config.yaml          # brand, intents, thresholds
src/                         # prepare, train, agent, evaluate
scripts/                     # download, golden set, run_all
data/golden/                 # golden_set + judge agreement + notes
results/                     # metrics + predictions + failures
REPORT.md                    # ≤6 page analysis
DECISION_LOG.md              # 10–15 decisions
```

## Citations / borrowing

- Dataset: Thought Vector — Customer Support on Twitter (Kaggle); Apple subset via OpenArchive/AppleConvos (HF).
- Classical IR/classification: scikit-learn `TfidfVectorizer`, `LinearSVC`, `CalibratedClassifierCV`.
- Optional LLM API: OpenAI Python SDK (`gpt-4o-mini` by default in config).
- Assignment framing: Hiver SDE Intern take-home brief.

## License note

Upstream Twitter support corpus is third-party data (see Kaggle / HF cards). This repo is a take-home artifact; do not redistribute the raw CSV if upstream terms forbid it. Raw download is gitignored; `scripts/download_data.py` fetches it.
