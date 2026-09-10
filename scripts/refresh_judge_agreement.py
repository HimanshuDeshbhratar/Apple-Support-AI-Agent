"""Refresh automated judge labels on the fixed human-judged draft panel.

Keeps human_label + customer_text stable; regenerates draft/auto scores from
the current agent so agreement stays meaningful after model changes.
"""
from __future__ import annotations

from collections import Counter

from src.agent import SupportAgent
from src.evaluate import heuristic_reply_score
from src.intents import IntentClassifier
from src.retrieval import ReplyRetriever
from src.utils import load_config, project_path, read_jsonl, write_jsonl


def main() -> None:
    cfg = load_config()
    rows = read_jsonl(project_path("data", "golden", "judge_agreement.jsonl"))
    clf = IntentClassifier.load(project_path("data", "processed", "intent_model.joblib"))
    retriever = ReplyRetriever.load(project_path(cfg["paths"]["retrieval_index"]))
    agent = SupportAgent(cfg, clf, retriever, use_llm=False)

    updated = []
    for r in rows:
        out = agent.handle(r["customer_text"])
        ref = r.get("reference_agent_text", "")
        h = heuristic_reply_score(out.draft_reply, ref, r["customer_text"])
        updated.append(
            {
                **r,
                "draft_reply": out.draft_reply,
                "auto_label": h["label"],
                "auto_score": h["score"],
                "auto_notes": h["notes"],
                "intent": out.intent,
                "action": out.action,
            }
        )

    write_jsonl(project_path("data", "golden", "judge_agreement.jsonl"), updated)
    agree = sum(x["human_label"] == x["auto_label"] for x in updated) / len(updated)
    print(f"Updated {len(updated)} rows; human↔auto agreement={agree:.3f}")
    print("human", Counter(x["human_label"] for x in updated))
    print("auto", Counter(x["auto_label"] for x in updated))


if __name__ == "__main__":
    main()
