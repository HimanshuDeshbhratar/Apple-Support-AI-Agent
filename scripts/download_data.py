"""Download Apple Support conversation pairs (HF mirror of Kaggle Twitter CS dataset).

Source: OpenArchive/AppleConvos on Hugging Face — filtered subset of
thoughtvector/customer-support-on-twitter (AppleSupport only).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="OpenArchive/AppleConvos",
        repo_type="dataset",
        filename="train.csv",
        local_dir=str(OUT),
    )
    print(f"Downloaded: {path}")
    print("Citation: Thought Vector / Stuart Axelbrooke — Customer Support on Twitter (Kaggle).")
    print("Mirror used: OpenArchive/AppleConvos (Apple-filtered pairs).")


if __name__ == "__main__":
    main()
