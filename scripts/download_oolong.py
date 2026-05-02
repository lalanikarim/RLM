#!/usr/bin/env python3
"""Download and cache OOLONG dataset locally.

Usage:
    python3 scripts/download_oolong.py              # 20 validation examples
    python3 scripts/download_oolong.py --split test  # 50 test examples
    python3 scripts/download_oolong.py --all         # full validation (1,300)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlm.datasets.oolong import OOLONGContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OOLONG dataset")
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Max examples to download (default: 20)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download full split (ignores --max)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    max_examples = None if args.all else args.max
    print(f"OOLONG download: split={args.split}, max={max_examples}, seed={args.seed}")
    print("=" * 60)

    # Load and show info
    adapter = OOLONGContext(
        split=args.split, max_examples=max_examples, seed=args.seed, force_download=True
    )

    if args.all:
        # Show info before downloading full set
        temp_adapter = OOLONGContext(split=args.split, max_examples=None, seed=42)
        info = temp_adapter.info()
        print("\nDataset info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    cases = adapter.load_as_test_cases()
    print(f"\nLoaded {len(cases)} examples")

    # Show sample
    if cases:
        sample = cases[0]
        print("\nSample case:")
        print(f"  ID:     {sample.id}")
        print(f"  Name:   {sample.name}")
        print(f"  Task:   {sample.context_type}/{sample.query[:60]}")
        print(f"  Answer: {sample.ground_truth}")
        print(f"  Context: {len(sample.context)} chars ({sample.context_size} tokens)")

    print("\n✅ Dataset cached in:", ROOT / "data" / "oolong")


if __name__ == "__main__":
    main()
