"""Prepare Apple Support conversation pairs and stratified subsample."""
from __future__ import annotations

import argparse

import pandas as pd

from src.labeling import refine_intent
from src.utils import clean_text, load_config, project_path


def prepare_pairs(cfg: dict) -> pd.DataFrame:
    raw = project_path(cfg["paths"]["raw_csv"])
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing {raw}. Run: python scripts/download_data.py"
        )
    df = pd.read_csv(raw)
    df = df.rename(
        columns={"text_input": "customer_text", "text_response": "agent_text"}
    )
    df["customer_clean"] = df["customer_text"].map(clean_text)
    df["agent_clean"] = df["agent_text"].map(clean_text)
    df = df[df["customer_clean"].str.len() >= 20].copy()
    df = df[df["agent_clean"].str.len() >= 20].copy()
    df["weak_intent"] = df["customer_clean"].map(refine_intent)
    df = df.reset_index(drop=True)
    df["pair_id"] = df.index.astype(str)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    intents = df["weak_intent"].unique().tolist()
    per = max(1, n // max(1, len(intents)))
    parts = []
    for intent in intents:
        subset = df[df["weak_intent"] == intent]
        take = min(len(subset), per)
        parts.append(subset.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        need = min(n - len(out), len(df))
        extra = df.sample(n=need, random_state=seed + 1)
        out = pd.concat([out, extra], ignore_index=True).drop_duplicates(
            subset=["pair_id"]
        )
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = prepare_pairs(cfg)
    seed = cfg["subsample"]["seed"]
    n_ret = cfg["subsample"]["retrieval_pairs"]
    n_hold = cfg["subsample"]["eval_holdout"]

    holdout = stratified_sample(df, n_hold, seed)
    rest = df[~df["pair_id"].isin(set(holdout["pair_id"]))]
    retrieval = stratified_sample(rest, n_ret, seed + 7)

    out_dir = project_path("data", "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = project_path(cfg["paths"]["pairs_csv"])
    retrieval.to_csv(pairs_path, index=False)
    holdout.to_csv(out_dir / "holdout_candidates.csv", index=False)

    print(f"Wrote retrieval pairs: {len(retrieval)} -> {pairs_path}")
    print(f"Wrote holdout candidates: {len(holdout)}")
    print("Intent distribution (retrieval):")
    print(retrieval["weak_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
