"""TF-IDF retrieval over historical AppleSupport resolutions."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import clean_text


@dataclass
class RetrievedExample:
    pair_id: str
    customer_text: str
    agent_text: str
    intent: str
    score: float


class ReplyRetriever:
    def __init__(self, max_features: int = 50000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            min_df=2,
            stop_words="english",
        )
        self.matrix = None
        self.df: pd.DataFrame | None = None

    def fit(self, pairs: pd.DataFrame) -> "ReplyRetriever":
        self.df = pairs.reset_index(drop=True).copy()
        docs = self.df["customer_clean"].tolist()
        self.matrix = self.vectorizer.fit_transform(docs)
        return self

    def retrieve(self, query: str, top_k: int = 5, intent: str | None = None) -> list[RetrievedExample]:
        assert self.df is not None and self.matrix is not None
        q = clean_text(query)
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).ravel()

        if intent and intent != "other":
            # Soft boost same-intent historical pairs
            same = (self.df["weak_intent"] == intent).to_numpy()
            sims = sims + (0.08 * same.astype(float))

        order = np.argsort(-sims)[:top_k]
        out: list[RetrievedExample] = []
        for i in order:
            row = self.df.iloc[int(i)]
            out.append(
                RetrievedExample(
                    pair_id=str(row["pair_id"]),
                    customer_text=str(row["customer_text"]),
                    agent_text=str(row["agent_text"]),
                    intent=str(row["weak_intent"]),
                    score=float(sims[int(i)]),
                )
            )
        return out

    def save(self, path) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "df": self.df},
            path,
        )

    @classmethod
    def load(cls, path) -> "ReplyRetriever":
        obj = joblib.load(path)
        inst = cls()
        inst.vectorizer = obj["vectorizer"]
        inst.matrix = obj["matrix"]
        inst.df = obj["df"]
        return inst
