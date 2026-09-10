"""Shared rubric for weak/gold intent and escalation labels."""
from __future__ import annotations

import re

from src.utils import clean_text

INTENT_RULES: list[tuple[str, list[str]]] = [
    (
        "battery_charging",
        [
            r"batter",
            r"charg",
            r"drain",
            r"overheat",
            r"hot\b",
            r"power.*(die|dead|off)",
            r"%\s*battery",
        ],
    ),
    (
        "ios_update_bugs",
        [
            r"\bios\b",
            r"update",
            r"upgrad",
            r"ios\s*1[01]",
            r"software update",
            r"after (the )?update",
        ],
    ),
    (
        "app_crash_freeze",
        [
            r"crash",
            r"freez",
            r"force.?clos",
            r"not responding",
            r"keeps closing",
            r"glitch",
            r"bug",
        ],
    ),
    (
        "connectivity",
        [
            r"wi-?fi",
            r"wifi",
            r"bluetooth",
            r"cellular",
            r"signal",
            r"airdrop",
            r"connect",
            r"network",
            r"lte",
            r"5g",
            r"hotspot",
        ],
    ),
    (
        "account_icloud",
        [
            r"icloud",
            r"apple\s*id",
            r"sign.?in",
            r"password",
            r"two.?factor",
            r"2fa",
            r"sync",
            r"backup",
            r"storage",
        ],
    ),
    (
        "hardware_device",
        [
            r"screen",
            r"display",
            r"speaker",
            r"microphone",
            r"button",
            r"camera",
            r"home button",
            r"earpiece",
            r"crack",
            r"broken",
            r"hardware",
        ],
    ),
    (
        "music_media",
        [
            r"apple music",
            r"itunes",
            r"podcast",
            r"song",
            r"playlist",
            r"streaming",
            r"video",
            r"playback",
        ],
    ),
    (
        "privacy_security",
        [
            r"privacy",
            r"hack",
            r"secur",
            r"spyware",
            r"tracking",
            r"permission",
            r"location",
            r"stolen",
            r"phish",
        ],
    ),
    (
        "billing_purchase",
        [
            r"charg(e|ed|ing)\b.*\$",
            r"\$\d",
            r"refund",
            r"subscription",
            r"purchase",
            r"billing",
            r"payment",
            r"app store",
            r"receipt",
            r"invoice",
        ],
    ),
    (
        "general_howto",
        [
            r"how (do|can|to)",
            r"where (do|can|is)",
            r"help me",
            r"settings",
            r"feature",
            r"what does",
        ],
    ),
]


def weak_label_intent(text: str) -> str:
    t = clean_text(text).lower()
    scores: dict[str, int] = {}
    for intent, patterns in INTENT_RULES:
        score = sum(1 for p in patterns if re.search(p, t))
        if score:
            scores[intent] = score
    if not scores:
        return "other"
    return max(scores, key=scores.get)


def refine_intent(text: str) -> str:
    """Priority rubric used for training labels and golden intents."""
    t = clean_text(text).lower()
    if re.search(
        r"(\$\d|\brefund\b|\bsubscription\b|\bbilled\b|\bpurchase\b|app store.*(charg|buy))",
        t,
    ):
        return "billing_purchase"
    if re.search(
        r"(\bhack(?:ed|ing)?\b|\bstolen\b|\bprivacy\b|\bspyware\b|\bphish|"
        r"\bidentity theft\b|\bmalware\b)",
        t,
    ):
        return "privacy_security"
    if re.search(r"(batter|charg(e|ing)|drain|overheat)", t) and not re.search(
        r"(\$\d|\brefund\b|\bsubscription\b)", t
    ):
        return "battery_charging"
    if re.search(r"(wi-?fi|wifi|bluetooth|cellular|airdrop|signal|hotspot)", t):
        return "connectivity"
    if re.search(r"(icloud|apple id|sign.?in|password|2fa|two.?factor|backup)", t):
        return "account_icloud"
    if re.search(r"(crash|freez|force.?close|not responding)", t):
        return "app_crash_freeze"
    if re.search(r"(after (the )?update|ios\s*\d|software update|upgrad)", t):
        return "ios_update_bugs"
    if re.search(r"(screen|speaker|camera|button|crack|display|microphone)", t):
        return "hardware_device"
    if re.search(r"(apple music|itunes|podcast|playlist|song)", t):
        return "music_media"
    if re.search(r"(how (do|can|to)|where (do|can)|settings >)", t):
        return "general_howto"
    return weak_label_intent(text)


def gold_escalate(text: str, intent: str) -> tuple[str, str]:
    t = clean_text(text).lower()
    reasons = []
    if re.search(
        r"(lawyer|lawsuit|sue|legal|police|fraud|chargeback|stolen|hacked|threaten|suicide|child)",
        t,
    ):
        reasons.append("legal_safety_or_fraud_language")
    if intent in {"billing_purchase", "privacy_security"}:
        reasons.append("sensitive_intent_needs_human")
    if re.search(r"(lost (my )?iphone|locked out|can't (log|sign) in|identity theft)", t):
        reasons.append("account_takeover_or_device_loss")
    if re.search(r"(worst company|scam|hate apple|ridiculous)", t) and intent != "general_howto":
        reasons.append("severe_complaint_tone")
    if len(t) < 25:
        reasons.append("too_little_context")
    if reasons:
        return "escalate", "; ".join(reasons)
    return "auto_handle", "Routine diagnostic issue suitable for scripted first reply"


def intrinsic_reply_quality(text: str) -> tuple[str, float, list[str]]:
    """Shared rubric for human-style bands and automated judge proxy."""
    t = clean_text(text).lower()
    score = 0.0
    notes: list[str] = []
    if 40 <= len(t) <= 220:
        score += 0.34
        notes.append("length_ok")
    elif 20 <= len(t) < 40 or 220 < len(t) <= 400:
        score += 0.15
        notes.append("length_borderline")
    else:
        notes.append("length_off")
    if any(x in t for x in ("sorry", "here for you", "happy to", "thanks", "help")):
        score += 0.33
        notes.append("empathy_or_ack")
    if any(x in t for x in ("ios", "settings", "version", "dm", "steps", "model", "device")):
        score += 0.33
        notes.append("asks_diagnostics")
    score = float(max(0.0, min(1.0, score)))
    label = "good" if score >= 0.75 else ("ok" if score >= 0.45 else "weak")
    return label, score, notes
