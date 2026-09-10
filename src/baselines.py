"""Baselines for comparison against the full agent.

1) Trivial: always predict majority intent + copy a canned reply + never escalate
2) Simple: TF-IDF nearest-neighbor copy of historical reply + rule intent + keyword escalate
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.escalate import decide_escalation
from src.retrieval import ReplyRetriever
from src.utils import clean_text


CANNED = (
    "Thanks for reaching out to Apple Support. Please tell us your device model "
    "and iOS version so we can help."
)


@dataclass
class BaselineResult:
    intent: str
    draft_reply: str
    action: str
    escalation_reason: str
    name: str


class TrivialBaseline:
    def __init__(self, majority_intent: str = "ios_update_bugs"):
        self.majority_intent = majority_intent

    def handle(self, customer_text: str) -> BaselineResult:
        return BaselineResult(
            intent=self.majority_intent,
            draft_reply=CANNED,
            action="auto_handle",
            escalation_reason="trivial baseline never escalates",
            name="trivial",
        )


class SimpleBaseline:
    def __init__(self, retriever: ReplyRetriever, escalate_keywords: list[str]):
        self.retriever = retriever
        self.escalate_keywords = escalate_keywords

    def handle(self, customer_text: str) -> BaselineResult:
        hits = self.retriever.retrieve(customer_text, top_k=1)
        # Simple = copy nearest historical reply; intent = that neighbor's label
        if hits:
            intent = hits[0].intent
            reply = clean_text(hits[0].agent_text)
            score = hits[0].score
        else:
            intent = "other"
            reply = CANNED
            score = 0.0
        esc = decide_escalation(
            customer_text=customer_text,
            intent=intent,
            intent_confidence=0.5,
            retrieval_score=score,
            escalate_keywords=self.escalate_keywords,
            auto_handle_min_confidence=0.55,
        )
        return BaselineResult(
            intent=intent,
            draft_reply=reply,
            action=esc.action,
            escalation_reason=esc.reason,
            name="simple_retrieval",
        )


def majority_intent_from_pairs(pairs: pd.DataFrame) -> str:
    return pairs["weak_intent"].value_counts().idxmax()
