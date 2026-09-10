"""Escalation policy for Apple Support agent."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils import clean_text


HIGH_RISK_INTENTS = {"billing_purchase", "privacy_security"}


@dataclass
class EscalationDecision:
    action: str  # "auto_handle" | "escalate"
    reason: str
    risk_flags: list[str]


def decide_escalation(
    customer_text: str,
    intent: str,
    intent_confidence: float,
    retrieval_score: float,
    escalate_keywords: list[str],
    auto_handle_min_confidence: float,
) -> EscalationDecision:
    text = clean_text(customer_text).lower()
    flags: list[str] = []

    for kw in escalate_keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", text):
            flags.append(f"keyword:{kw}")

    # Strong emotion / repeated frustration often needs a human
    if re.search(r"(worst|hate|ridiculous|scam|fraud|angry|furious)", text):
        flags.append("high_frustration")
    if text.count("!") >= 3 or text.count("?") >= 3:
        flags.append("high_punctuation_stress")

    if intent in HIGH_RISK_INTENTS:
        flags.append(f"risk_intent:{intent}")

    if intent_confidence < auto_handle_min_confidence:
        flags.append("low_intent_confidence")

    if retrieval_score < 0.12:
        flags.append("weak_historical_match")

    # Account lockouts and device-lost situations
    if re.search(r"(locked out|lost (my )?iphone|stolen|can't (log|sign) in)", text):
        flags.append("account_lock_or_loss")

    if flags:
        # Soft cases: only weak match + low confidence together → escalate
        soft_only = set(flags) <= {
            "low_intent_confidence",
            "weak_historical_match",
            "high_punctuation_stress",
        }
        if soft_only and intent_confidence >= 0.45 and retrieval_score >= 0.08:
            return EscalationDecision(
                "auto_handle",
                "Borderline signals but enough confidence and retrieval support",
                flags,
            )
        return EscalationDecision(
            "escalate",
            "Escalated due to: " + "; ".join(flags),
            flags,
        )

    return EscalationDecision(
        "auto_handle",
        "No high-risk signals; intent confident and similar historical resolutions exist",
        [],
    )
