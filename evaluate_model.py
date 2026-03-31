from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the saved evaluation summary for a trained SDP model."
    )
    parser.add_argument(
        "--artifacts",
        default="models",
        help="Directory containing metadata.json from training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.artifacts) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
