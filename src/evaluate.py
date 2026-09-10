"""Evaluation harness: automated metrics + LLM-as-judge + judge agreement check."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from src.agent import SupportAgent
from src.baselines import SimpleBaseline, TrivialBaseline, majority_intent_from_pairs
from src.intents import IntentClassifier
from src.labeling import intrinsic_reply_quality
from src.retrieval import ReplyRetriever
from src.utils import (
    clean_text,
    ensure_openai_client,
    load_config,
    project_path,
    read_jsonl,
    write_jsonl,
)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", clean_text(text).lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def heuristic_reply_score(draft: str, reference: str, customer: str) -> dict[str, Any]:
    """Deterministic reply-quality proxy with issue-relevance penalties."""
    label, base, notes = intrinsic_reply_quality(draft)
    overlap = jaccard(draft, reference)
    cust_overlap = jaccard(draft, customer)
    score = base
    score = min(1.0, score + 0.08 * overlap + 0.12 * cust_overlap)
    d = clean_text(draft).lower()
    c = clean_text(customer).lower()
    if re.search(r"(guaranteed refund|we will refund|lawsuit)", d):
        score = max(0.0, score - 0.4)
        notes = notes + ["unsafe_promise"]
    if "dm" in d and re.search(r"(iphone|ipad|ios\s*\d)", c):
        score = max(0.0, score - 0.15)
        notes = notes + ["generic_dm_despite_details"]
    if re.search(r"\bios version\b", d) and re.search(
        r"(mac\s?mini|macbook|imac|high sierra|apple watch)", c
    ):
        score = max(0.0, score - 0.35)
        notes = notes + ["wrong_product_diagnostic"]
    if re.search(r"(stolen|hack|activation lock|refund|\$\d|account recovery)", c) and not re.search(
        r"(dm|account|billing|security|purchase|apple id)", d
    ):
        score = max(0.0, score - 0.2)
        notes = notes + ["missed_sensitive_substance"]
    if len(c) < 40 and "dm" in d:
        score = max(0.0, score - 0.1)
        notes = notes + ["overlong_for_short_context"]
    notes = notes + [f"ref_jaccard={overlap:.2f}", f"cust_jaccard={cust_overlap:.2f}"]
    label = "good" if score >= 0.75 else ("ok" if score >= 0.45 else "weak")
    return {"score": float(score), "label": label, "notes": notes}


def llm_judge(draft: str, customer: str, reference: str) -> dict[str, Any] | None:
    client = ensure_openai_client()
    if client is None:
        return None
    prompt = f"""You are evaluating a drafted Apple Support Twitter reply.
Score 1-5 on: empathy, helpfulness, grounding (no invented policy), safety.
Also give overall label: good | ok | weak.

Customer: {customer}
Historical brand reply (reference style/content, not gold wording): {reference}
Draft under review: {draft}

Return JSON:
{{"empathy":1-5,"helpfulness":1-5,"grounding":1-5,"safety":1-5,"label":"good|ok|weak","rationale":"..."}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        dims = [data.get(k, 3) for k in ("empathy", "helpfulness", "grounding", "safety")]
        data["score"] = float(np.mean(dims) / 5.0)
        return data
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def agreement(a: list[str], b: list[str]) -> dict[str, float]:
    assert len(a) == len(b)
    matches = sum(x == y for x, y in zip(a, b))
    return {"exact_match_rate": matches / max(1, len(a)), "n": float(len(a))}


def eval_system(
    name: str,
    handler,
    golden: list[dict],
    use_llm_judge: bool,
) -> dict[str, Any]:
    y_true_i, y_pred_i = [], []
    y_true_a, y_pred_a = [], []
    reply_scores = []
    rows = []

    for ex in golden:
        out = handler(ex["customer_text"])
        # Support both AgentResult and BaselineResult
        intent = out.intent
        action = out.action
        draft = out.draft_reply
        esc_reason = out.escalation_reason

        y_true_i.append(ex["gold_intent"])
        y_pred_i.append(intent)
        y_true_a.append(ex["gold_action"])
        y_pred_a.append(action)

        h = heuristic_reply_score(draft, ex["reference_agent_text"], ex["customer_text"])
        judge = h
        if use_llm_judge:
            lj = llm_judge(draft, ex["customer_text"], ex["reference_agent_text"])
            if lj and "label" in lj:
                judge = {**h, "llm_judge": lj, "label": lj["label"], "score": lj.get("score", h["score"])}

        reply_scores.append(judge["score"])
        rows.append(
            {
                "id": ex["id"],
                "system": name,
                "customer_text": ex["customer_text"],
                "gold_intent": ex["gold_intent"],
                "pred_intent": intent,
                "gold_action": ex["gold_action"],
                "pred_action": action,
                "escalation_reason": esc_reason,
                "draft_reply": draft,
                "reply_score": judge["score"],
                "reply_label": judge["label"],
                "judge_detail": judge,
            }
        )

    intent_acc = accuracy_score(y_true_i, y_pred_i)
    intent_f1 = f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)
    act_acc = accuracy_score(y_true_a, y_pred_a)
    # Escalation: treat escalate as positive class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_a,
        y_pred_a,
        labels=["escalate"],
        average="micro",
        zero_division=0,
    )

    summary = {
        "system": name,
        "n": len(golden),
        "intent_accuracy": float(intent_acc),
        "intent_macro_f1": float(intent_f1),
        "action_accuracy": float(act_acc),
        "escalate_precision": float(prec),
        "escalate_recall": float(rec),
        "escalate_f1": float(f1),
        "reply_score_mean": float(np.mean(reply_scores)),
        "reply_score_std": float(np.std(reply_scores)),
        "intent_report": classification_report(
            y_true_i, y_pred_i, zero_division=0, output_dict=True
        ),
    }
    return {"summary": summary, "rows": rows}


def calibrate_judge_agreement(golden: list[dict], seed: int = 42) -> dict[str, Any]:
    """Primary: independent human labels on agent drafts vs automated judge.

    Secondary: sanity check that intrinsic bands are reproducible on references.
    """
    agreement_path = project_path("data", "golden", "judge_agreement.jsonl")
    primary: dict[str, Any]
    if agreement_path.exists():
        rows = read_jsonl(agreement_path)
        human = [r["human_label"] for r in rows]
        auto = [r["auto_label"] for r in rows]
        primary = {
            "task": "human_vs_auto_judge_on_agent_drafts",
            "agreement": agreement(human, auto),
            "human_dist": dict(Counter(human)),
            "auto_dist": dict(Counter(auto)),
            "note": (
                "36 agent drafts were scored by an independent human rubric focused on "
                "issue-specific helpfulness (not the keyword checklist used by the auto judge). "
                "Disagreement usually means the auto judge over-rates generic DM/diagnostic templates."
            ),
        }
    else:
        primary = {"task": "missing_judge_agreement_file", "agreement": {"exact_match_rate": 0.0, "n": 0}}

    human_ref = []
    auto_ref = []
    for ex in golden:
        human_ref.append(ex.get("reference_reply_quality", "ok"))
        pred, _, _ = intrinsic_reply_quality(ex["reference_agent_text"])
        auto_ref.append(pred)
    secondary = {
        "task": "intrinsic_reproducibility_on_references",
        "agreement": agreement(human_ref, auto_ref),
        "note": "Reference quality bands were assigned with the same intrinsic function (sanity check only).",
    }
    return {"primary": primary, "secondary": secondary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--use-llm-agent", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    golden = read_jsonl(project_path(cfg["paths"]["golden_jsonl"]))
    if args.limit:
        golden = golden[: args.limit]

    pairs = pd.read_csv(project_path(cfg["paths"]["pairs_csv"]))
    clf = IntentClassifier.load(project_path("data", "processed", "intent_model.joblib"))
    retriever = ReplyRetriever.load(project_path(cfg["paths"]["retrieval_index"]))
    agent = SupportAgent(cfg, clf, retriever, use_llm=args.use_llm_agent)

    trivial = TrivialBaseline(majority_intent_from_pairs(pairs))
    simple = SimpleBaseline(retriever, cfg["escalation"]["escalate_keywords"])

    results = {}
    all_rows = []
    for name, handler in [
        ("trivial", trivial.handle),
        ("simple_retrieval", simple.handle),
        ("agent", agent.handle),
    ]:
        print(f"Evaluating {name} on {len(golden)} examples...")
        out = eval_system(name, handler, golden, use_llm_judge=args.llm_judge)
        results[name] = out["summary"]
        all_rows.extend(out["rows"])
        s = out["summary"]
        print(
            f"  intent_acc={s['intent_accuracy']:.3f} macroF1={s['intent_macro_f1']:.3f} "
            f"action_acc={s['action_accuracy']:.3f} reply={s['reply_score_mean']:.3f}"
        )

    judge_cal = calibrate_judge_agreement(golden)
    # Failure analysis candidates: agent wrong intent or escalate mismatch
    agent_rows = [r for r in all_rows if r["system"] == "agent"]
    failures = [
        r
        for r in agent_rows
        if r["gold_intent"] != r["pred_intent"] or r["gold_action"] != r["pred_action"]
    ]
    failures = sorted(failures, key=lambda r: r["reply_score"])[:40]

    out_dir = project_path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "systems": results,
                "judge_calibration": judge_cal,
                "config_brand": cfg["brand"],
                "n_golden": len(golden),
            },
            f,
            indent=2,
        )
    write_jsonl(out_dir / "predictions.jsonl", all_rows)
    write_jsonl(out_dir / "failure_candidates.jsonl", failures)

    # Markdown summary table
    lines = [
        "# Evaluation Results",
        "",
        f"Golden set size: {len(golden)}",
        "",
        "| System | Intent Acc | Intent Macro-F1 | Action Acc | Escalate P/R/F1 | Reply Score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        lines.append(
            f"| {name} | {s['intent_accuracy']:.3f} | {s['intent_macro_f1']:.3f} | "
            f"{s['action_accuracy']:.3f} | "
            f"{s['escalate_precision']:.2f}/{s['escalate_recall']:.2f}/{s['escalate_f1']:.2f} | "
            f"{s['reply_score_mean']:.3f} |"
        )
    lines += [
        "",
        "## Judge calibration",
        "",
        f"- Primary task: {judge_cal['primary']['task']}",
        f"- Human↔auto agreement on agent drafts: "
        f"{judge_cal['primary']['agreement']['exact_match_rate']:.3f} "
        f"(n={int(judge_cal['primary']['agreement']['n'])})",
        f"- Note: {judge_cal['primary'].get('note', '')}",
        f"- Secondary reproducibility on references: "
        f"{judge_cal['secondary']['agreement']['exact_match_rate']:.3f}",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
