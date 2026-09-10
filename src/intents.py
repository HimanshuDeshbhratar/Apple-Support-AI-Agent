"""Intent classification: TF-IDF + LinearSVC over weak labels, optional LLM refine."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.labeling import weak_label_intent
from src.utils import clean_text, ensure_openai_client


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    method: str
    rationale: str


class IntentClassifier:
    def __init__(self, intent_ids: list[str]):
        self.intent_ids = intent_ids
        self.pipeline: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "IntentClassifier":
        base = LinearSVC(class_weight="balanced", random_state=42)
        # CalibratedClassifierCV gives predict_proba for confidence
        clf = CalibratedClassifierCV(base, cv=3)
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=40000,
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words="english",
                    ),
                ),
                ("clf", clf),
            ]
        )
        self.pipeline.fit(texts, labels)
        return self

    def predict_one(self, text: str) -> IntentPrediction:
        cleaned = clean_text(text)
        if self.pipeline is None:
            intent = weak_label_intent(cleaned)
            return IntentPrediction(intent, 0.4, "rules", "rule fallback")

        proba = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)
        idx = int(np.argmax(proba))
        intent = classes[idx]
        conf = float(proba[idx])
        # If model is unsure, fall back to rules when they fire strongly
        rule_intent = weak_label_intent(cleaned)
        if conf < 0.35 and rule_intent != "other":
            return IntentPrediction(
                rule_intent, max(conf, 0.45), "rules+model", "low model confidence"
            )
        return IntentPrediction(intent, conf, "sklearn", "calibrated LinearSVC")

    def save(self, path) -> None:
        joblib.dump({"pipeline": self.pipeline, "intent_ids": self.intent_ids}, path)

    @classmethod
    def load(cls, path) -> "IntentClassifier":
        obj = joblib.load(path)
        inst = cls(obj["intent_ids"])
        inst.pipeline = obj["pipeline"]
        return inst


def llm_intent(
    text: str, intents: list[dict[str, str]], fallback: IntentPrediction
) -> IntentPrediction:
    client = ensure_openai_client()
    if client is None:
        return fallback
    catalog = "\n".join(f"- {i['id']}: {i['description']}" for i in intents)
    prompt = (
        "Classify the customer support tweet into exactly one intent id.\n"
        f"Intents:\n{catalog}\n\n"
        f"Tweet: {text}\n\n"
        'Return JSON: {"intent": "...", "confidence": 0-1, "rationale": "..."}'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        intent = data.get("intent", fallback.intent)
        valid = {i["id"] for i in intents}
        if intent not in valid:
            intent = fallback.intent
        return IntentPrediction(
            intent=intent,
            confidence=float(data.get("confidence", fallback.confidence)),
            method="llm",
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:  # noqa: BLE001
        return IntentPrediction(
            fallback.intent,
            fallback.confidence,
            fallback.method,
            f"llm failed: {exc}",
        )


def train_intent_classifier(pairs: pd.DataFrame, intent_ids: list[str]) -> IntentClassifier:
    clf = IntentClassifier(intent_ids)
    clf.fit(pairs["customer_clean"].tolist(), pairs["weak_intent"].tolist())
    return clf
