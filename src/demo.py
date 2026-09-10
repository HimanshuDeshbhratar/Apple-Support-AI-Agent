"""Interactive demo: classify, draft, and escalate a single customer message."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.agent import SupportAgent
from src.intents import IntentClassifier
from src.retrieval import ReplyRetriever
from src.utils import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Customer message text")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    clf = IntentClassifier.load(project_path("data", "processed", "intent_model.joblib"))
    retriever = ReplyRetriever.load(project_path(cfg["paths"]["retrieval_index"]))
    agent = SupportAgent(cfg, clf, retriever, use_llm=args.use_llm)
    result = agent.handle(args.text)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
