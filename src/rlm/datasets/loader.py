"""Unified dataset loader — supports JSON and OOLONG formats."""

from __future__ import annotations

import json
from pathlib import Path

from .oolong import TestCase


# Default paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "synthetic_test_cases.json"


def load_dataset(
    dataset_path: str | Path | None = None,
    oolong_split: str = "validation",
    oolong_max: int | None = None,
) -> list[TestCase]:
    """Load test cases from either JSON or OOLONG dataset.

    Args:
        dataset_path: Path to JSON dataset file. If None, uses default.
        oolong_split: If dataset_path="oolong", which split to use.
        oolong_max: If loading OOLONG, limit to N examples.

    Returns:
        List of TestCase objects
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET

    path = str(dataset_path).lower()

    # OOLONG shortcut
    if path in ("oolong", "oolong-validation", "oolong-test"):
        split_map = {
            "oolong": oolong_split,
            "oolong-validation": "validation",
            "oolong-test": "test",
        }
        from .oolong import OOLONGContext

        split = split_map.get(path, oolong_split)
        adapter = OOLONGContext(
            split=split, max_examples=oolong_max, force_download=False
        )
        return adapter.load_as_test_cases()

    # JSON file
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        dataset = json.load(f)

    cases = []
    for case_data in dataset["cases"]:
        cases.append(
            TestCase(
                id=case_data["id"],
                name=case_data["name"],
                query=case_data["query"],
                ground_truth=case_data["ground_truth"],
                context_type=case_data["context_type"],
                context_size=case_data["context_size"],
                difficulty=case_data["difficulty"],
                context=case_data["context"],
                answer_type=case_data.get("answer_type", "categorical"),
            )
        )

    return cases
