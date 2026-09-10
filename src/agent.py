"""End-to-end Apple Support agent: intent → retrieve → draft → escalate."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.escalate import EscalationDecision, decide_escalation
from src.intents import IntentClassifier, IntentPrediction, llm_intent
from src.retrieval import ReplyRetriever, RetrievedExample
from src.utils import clean_text, ensure_openai_client


@dataclass
class AgentResult:
    customer_text: str
    intent: str
    intent_confidence: float
    intent_method: str
    intent_rationale: str
    draft_reply: str
    action: str
    escalation_reason: str
    risk_flags: list[str]
    retrieved: list[dict[str, Any]]
    reply_method: str


def _sanitize_brand_reply(text: str) -> str:
    # Remove anonymized customer handles like @115854
    text = re.sub(r"@\d+\b", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def draft_reply_from_retrieval(
    customer_text: str,
    intent: str,
    examples: list[RetrievedExample],
) -> tuple[str, str]:
    """Deterministic grounded draft: adapt top historical reply patterns."""
    if not examples:
        return (
            "Thanks for reaching out. Could you share your device model and iOS version "
            "(Settings > General > About) so we can look into this?",
            "template_fallback",
        )

    top = _sanitize_brand_reply(examples[0].agent_text)
    cust = clean_text(customer_text).lower()
    already_has_device = bool(re.search(r"(iphone|ipad|ios\s*\d|iphone\s*\d)", cust))
    sensitive = intent in {"billing_purchase", "privacy_security", "account_icloud"}
    wants_dm = ("DM" in top or "direct message" in top.lower()) and (
        sensitive or len(cust) > 180 or not already_has_device
    )

    if wants_dm:
        reply = (
            "Thanks for the details — we'd like to look at this more closely in a private channel. "
            "Please DM us with your Apple ID email domain (not the password) and a short summary "
            "of what you've already tried."
        )
        return reply, "retrieval_adapted_dm"

    parts = ["Thanks for contacting Apple Support."]
    if intent == "battery_charging":
        parts.append(
            "Sorry you're seeing battery/charging trouble. "
            + (
                "Since you've shared device details, have you tried a force restart, "
                "and does this happen only with a specific cable/charger?"
                if already_has_device
                else "Which iPhone/iPad model are you using, and what iOS version is installed?"
            )
        )
    elif intent == "ios_update_bugs":
        parts.append(
            "Sorry the update isn't behaving as expected. "
            + (
                "When did the issue start after updating, and does a restart change anything?"
                if already_has_device
                else "What exact iOS version are you on now, and when did the issue start?"
            )
        )
    elif intent == "connectivity":
        parts.append(
            "Sorry you're having connection issues. "
            "Is this Wi-Fi, cellular, or Bluetooth — and does it happen with more than one accessory/network?"
        )
    elif intent == "billing_purchase":
        parts.append(
            "We can help review purchase/billing questions. "
            "Please share the approximate charge date and whether it shows under Purchases in your Apple ID settings."
        )
    elif intent == "privacy_security":
        parts.append(
            "Sorry you're dealing with a security concern — we take this seriously. "
            "Please DM us so we can help secure the Apple ID and review next steps privately."
        )
    elif intent == "account_icloud":
        parts.append(
            "We can help with Apple ID / iCloud access. "
            "If you're locked out or in recovery, the safest next step is a DM so we can verify details privately."
        )
    elif intent == "app_crash_freeze":
        parts.append(
            "Sorry things are crashing/freezing. Which app or screen is affected, "
            "and does force-closing the app or restarting the device change it?"
        )
    elif intent == "hardware_device":
        parts.append(
            "Sorry you're seeing a hardware/display issue. "
            "Which model is this, and did it start after a drop, liquid contact, or out of nowhere?"
        )
    elif intent == "music_media":
        parts.append(
            "Sorry Apple Music/media isn't cooperating. "
            "Are you signed in with the correct Apple ID, and does the issue happen on Wi-Fi and cellular?"
        )
    else:
        parts.append(
            "We're here to help. Could you share your device model and iOS version "
            "(Settings > General > About), plus what you expected vs what happened?"
        )

    # Light grounding from historical tone without pasting long anonymized replies
    if top and len(top) < 120 and "http" not in top.lower():
        parts.append(f"Similar past cases usually start by clarifying: {top}")

    return " ".join(parts), "retrieval_adapted"


def draft_reply_llm(
    customer_text: str,
    intent: str,
    examples: list[RetrievedExample],
    brand: str,
) -> tuple[str, str] | None:
    client = ensure_openai_client()
    if client is None:
        return None
    evidence = [
        {
            "customer": clean_text(e.customer_text)[:240],
            "brand_reply": _sanitize_brand_reply(e.agent_text)[:240],
            "similarity": round(e.score, 3),
        }
        for e in examples[:4]
    ]
    prompt = f"""You are drafting a first reply for {brand} on Twitter/X.
Rules:
- Ground the reply in the historical brand replies below (tone + troubleshooting asks).
- Do NOT invent policies, refunds, or guarantees not implied by evidence.
- Do NOT include links unless asking the customer to DM.
- Keep under 280 characters when possible; max 450 characters.
- Be empathetic, concise, and ask for missing diagnostic details.

Intent: {intent}
Customer message: {customer_text}

Historical similar resolutions (evidence):
{json.dumps(evidence, ensure_ascii=False)}

Return JSON: {{"reply": "...", "grounding_note": "..."}}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        reply = str(data.get("reply", "")).strip()
        if not reply:
            return None
        return reply, "llm_grounded"
    except Exception:  # noqa: BLE001
        return None


class SupportAgent:
    def __init__(
        self,
        cfg: dict,
        intent_clf: IntentClassifier,
        retriever: ReplyRetriever,
        use_llm: bool = False,
    ):
        self.cfg = cfg
        self.intent_clf = intent_clf
        self.retriever = retriever
        self.use_llm = use_llm

    def handle(self, customer_text: str) -> AgentResult:
        pred = self.intent_clf.predict_one(customer_text)
        if self.use_llm:
            pred = llm_intent(customer_text, self.cfg["intents"], pred)

        top_k = self.cfg["retrieval"]["top_k"]
        retrieved = self.retriever.retrieve(
            customer_text, top_k=top_k, intent=pred.intent
        )
        best_score = retrieved[0].score if retrieved else 0.0

        if self.use_llm:
            llm_out = draft_reply_llm(
                customer_text,
                pred.intent,
                retrieved,
                self.cfg["brand"],
            )
            if llm_out:
                draft, reply_method = llm_out
            else:
                draft, reply_method = draft_reply_from_retrieval(
                    customer_text, pred.intent, retrieved
                )
        else:
            draft, reply_method = draft_reply_from_retrieval(
                customer_text, pred.intent, retrieved
            )

        esc = decide_escalation(
            customer_text=customer_text,
            intent=pred.intent,
            intent_confidence=pred.confidence,
            retrieval_score=best_score,
            escalate_keywords=self.cfg["escalation"]["escalate_keywords"],
            auto_handle_min_confidence=self.cfg["escalation"][
                "auto_handle_min_confidence"
            ],
        )

        return AgentResult(
            customer_text=customer_text,
            intent=pred.intent,
            intent_confidence=pred.confidence,
            intent_method=pred.method,
            intent_rationale=pred.rationale,
            draft_reply=draft,
            action=esc.action,
            escalation_reason=esc.reason,
            risk_flags=esc.risk_flags,
            retrieved=[asdict(r) for r in retrieved],
            reply_method=reply_method,
        )
