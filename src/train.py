"""Train retrieval index + intent classifier from processed pairs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.intents import train_intent_classifier
from src.retrieval import ReplyRetriever
from src.utils import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    pairs_path = project_path(cfg["paths"]["pairs_csv"])
    if not pairs_path.exists():
        raise SystemExit("Run prepare_data first: python -m src.prepare_data")

    pairs = pd.read_csv(pairs_path)
    intent_ids = [i["id"] for i in cfg["intents"]]

    print(f"Training intent classifier on {len(pairs)} pairs...")
    clf = train_intent_classifier(pairs, intent_ids)
    model_dir = project_path("data", "processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    clf_path = model_dir / "intent_model.joblib"
    clf.save(clf_path)

    print("Building retrieval index...")
    retriever = ReplyRetriever(
        max_features=cfg["retrieval"]["max_features"],
        ngram_range=tuple(cfg["retrieval"]["ngram_range"]),
    )
    retriever.fit(pairs)
    index_path = project_path(cfg["paths"]["retrieval_index"])
    retriever.save(index_path)

    print(f"Saved {clf_path}")
    print(f"Saved {index_path}")


if __name__ == "__main__":
    main()
