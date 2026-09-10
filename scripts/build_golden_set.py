"""Build a 200-example golden evaluation set with rubric-based labels."""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from src.labeling import gold_escalate, intrinsic_reply_quality, refine_intent
from src.utils import load_config, project_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config()

    holdout_path = project_path("data", "processed", "holdout_candidates.csv")
    if not holdout_path.exists():
        raise SystemExit("Run python -m src.prepare_data first")

    df = pd.read_csv(holdout_path)
    df["gold_intent"] = df["customer_text"].map(refine_intent)
    intents = sorted(df["gold_intent"].unique())
    per = max(1, args.n // len(intents))
    parts = []
    for intent in intents:
        subset = df[df["gold_intent"] == intent]
        take = min(len(subset), per)
        if take:
            parts.append(subset.sample(n=take, random_state=args.seed))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < args.n:
        extra = df[~df["pair_id"].isin(sampled["pair_id"])].sample(
            n=min(args.n - len(sampled), len(df)),
            random_state=args.seed + 3,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    sampled = sampled.drop_duplicates("pair_id").head(args.n).reset_index(drop=True)

    rows = []
    for _, row in sampled.iterrows():
        intent = refine_intent(row["customer_text"])
        action, esc_reason = gold_escalate(row["customer_text"], intent)
        band, _, _ = intrinsic_reply_quality(row["agent_text"])
        rows.append(
            {
                "id": f"gold_{row['pair_id']}",
                "customer_text": row["customer_text"],
                "reference_agent_text": row["agent_text"],
                "gold_intent": intent,
                "gold_action": action,
                "gold_escalation_reason": esc_reason,
                "reference_reply_quality": band,
                "notes": "Labelled with documented criteria in LABELING_NOTES.md",
            }
        )

    out = project_path(cfg["paths"]["golden_jsonl"])
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} golden examples -> {out}")
    print("Intent counts:", Counter(r["gold_intent"] for r in rows))
    print("Action counts:", Counter(r["gold_action"] for r in rows))


if __name__ == "__main__":
    main()
