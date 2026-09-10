# Golden set labeling notes

## Sampling
- Source: Apple-filtered pairs from Customer Support on Twitter (via `OpenArchive/AppleConvos`).
- After cleaning (drop short / URL-only messages), we hold out a stratified candidate pool (`data/processed/holdout_candidates.csv`) that is **excluded from the retrieval index**.
- Golden set: **200 examples**, stratified across the 11-intent taxonomy so rare intents (billing, privacy) are not drowned out by iOS-update chatter.

## Label fields
| Field | Meaning |
|---|---|
| `gold_intent` | Single primary intent for the customer message |
| `gold_action` | `auto_handle` or `escalate` for a first-response agent |
| `gold_escalation_reason` | Short human-readable justification |
| `reference_agent_text` | Historical AppleSupport reply (style/evidence reference, **not** a gold draft target for exact match) |
| `reference_reply_quality` | `good` / `ok` / `weak` band for judge calibration sanity checks |

## Intent criteria (summary)
Applied in priority order (`src/labeling.py::refine_intent`):
1. Clear money/refund/App Store purchase language → `billing_purchase`
2. Hack/stolen/privacy/malware language (word-boundary safe) → `privacy_security`
3. Battery/charging/drain (without money cues) → `battery_charging`
4. Wi-Fi/cellular/Bluetooth/AirDrop → `connectivity`
5. Apple ID / iCloud / sign-in → `account_icloud`
6. Crash/freeze → `app_crash_freeze`
7. Post-update / iOS version regressions → `ios_update_bugs`
8. Physical screen/speaker/camera defects → `hardware_device`
9. Apple Music / iTunes media → `music_media`
10. How-to / settings questions → `general_howto`
11. Else → `other` / weak rules

## Escalation criteria
Escalate if any of:
- Legal / fraud / safety language
- Sensitive intents: billing or privacy/security
- Device loss / lockout / account takeover cues
- Severe complaint tone on non-howto issues
- Too little context to safely auto-reply

Otherwise `auto_handle` for routine diagnostic first replies.

## Judge agreement panel
`judge_agreement.jsonl` holds **36** agent drafts with **independent** human labels (`good`/`ok`/`weak`) using issue-specific helpfulness — not the keyword checklist. Auto labels are refreshed with `scripts/refresh_judge_agreement.py`.

## Process honesty
Labels follow an explicit, code-backed rubric (script-assisted adjudication) with spot-fixes for known false friends (`whack`≠`hack`, “Call identity”≠identity theft). This is closer to **rubric-based adjudication** than free-form crowdsourcing. See `DECISION_LOG.md`.
