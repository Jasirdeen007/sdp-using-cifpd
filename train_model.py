from __future__ import annotations

import argparse
import json

from src.sdp_pipeline import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SDP defect prediction model.")
    parser.add_argument("--csv", required=True, help="Path to the CIFPD/Eclipse CSV file.")
    parser.add_argument(
        "--artifacts",
        default="models",
        help="Directory where the trained classifier and metadata will be saved.",
    )
    parser.add_argument(
        "--reports",
        default="reports",
        help="Directory where evaluation outputs and feature rankings will be saved.",
    )
    parser.add_argument(
        "--feature-method",
        default="mutual_info",
        choices=["mutual_info", "chi_square", "cramers_v", "random_forest"],
        help="Feature grouping method used to build intent_text.",
    )
    parser.add_argument(
        "--embedding-model",
        default="roberta-base",
        help="Hugging Face transformer model used for text embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train_pipeline(
        csv_path=args.csv,
        artifact_dir=args.artifacts,
        reports_dir=args.reports,
        feature_method=args.feature_method,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
