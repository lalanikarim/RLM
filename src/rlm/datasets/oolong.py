"""OOLONG dataset adapter — loads from HuggingFace, caches locally."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cached dataset path
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
OOLONG_CACHE_DIR = DATA_DIR / "oolong"
OOLONG_CACHE_JSON = OOLONG_CACHE_DIR / "oolong-synth.json"


# ---------------------------------------------------------------------------
# Model context capacity map
# ---------------------------------------------------------------------------

MODEL_CONTEXT_CAPACITY: dict[str, int] = {
    "qwen3:0.6b": 32_000,
    "qwen3:1.7b": 32_000,
    "qwen3:4b": 32_000,
    "qwen3:8b": 128_000,
    "qwen3:14b": 128_000,
    "qwen3:30b-a3b": 128_000,
    "deepseek-r1:1.5b": 32_000,
    "deepseek-r1:7b": 128_000,
    "deepseek-r1:8b": 128_000,
    "deepseek-r1:671b": 128_000,
    "gemma3:4b": 32_000,
    "gemma3:12b": 128_000,
    "gemma3:27b": 128_000,
    "llama3:8b": 32_000,
    "llama3:70b": 128_000,
    "llama3.1:8b": 32_000,
    "llama3.1:70b": 128_000,
    "llama3.2:1b": 12_800,
    "llama3.2:3b": 32_000,
    "llama3.2:11b": 32_000,
    "llama3.2:90b": 128_000,
    "gpt-oss:20.9b": 128_000,
}


# ---------------------------------------------------------------------------
# OOLONG test case
# ---------------------------------------------------------------------------


@dataclass
class OOLONGTestCase:
    """A test case loaded from OOLONG."""

    id: str
    context_window_text: str
    question: str
    answer: str
    answer_type: str
    task: str
    task_group: str
    dataset: str
    context_len: int
    input_subset: str
    num_labels: int

    @property
    def query(self) -> str:
        return self.question

    def to_test_case(self, context_window: int | None = None) -> TestCase:
        """Convert to the standard TestCase format."""
        context = self.context_window_text
        if context_window and len(context) > context_window * 3:
            context = context[: context_window * 3]
            while context and context[-1] not in ("\n", " "):
                context = context[:-1]
            context += "\n...(truncated)"

        return TestCase(
            id=self.id,
            name=f"{self.task_group}:{self.task} ({self.id[:12]})",
            query=self.query,
            ground_truth=self.answer,
            context_type=self.task_group,
            context_size=self.context_len,
            difficulty=self._difficulty(),
            context=context,
            answer_type=self.answer_type,
        )

    def _difficulty(self) -> str:
        if self.context_len <= 16_000:
            return "easy"
        elif self.context_len <= 64_000:
            return "medium"
        else:
            return "hard"


@dataclass
class TestCase:
    """Standard test case used across the benchmark."""

    id: str
    name: str
    query: str
    ground_truth: str
    context_type: str
    context_size: int
    difficulty: str
    context: str
    answer_type: str = "categorical"


# ---------------------------------------------------------------------------
# Download and cache OOLONG dataset
# ---------------------------------------------------------------------------


def _download_oolong_cache(
    split: str = "validation",
    max_examples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Download OOLONG dataset parquet shards and cache as JSON.

    Downloads only the parquet files for the requested split instead of
    loading the full repo via load_dataset (which downloads all 49 shards).
    """
    print(f"Downloading OOLONG dataset ({split} split) from HuggingFace...")
    t0 = time.time()

    if split == "validation":
        parquet_files = [f"data/validation-{i:05d}-of-00007.parquet" for i in range(7)]
    else:
        parquet_files = [f"data/test-{i:05d}-of-00042.parquet" for i in range(42)]

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    all_records: list[dict[str, Any]] = []

    for rel_path in parquet_files:
        print(f"  Fetching {rel_path.split('/')[-1]}...", end=" ")
        try:
            local_path = hf_hub_download(
                "oolongbench/oolong-synth",
                repo_type="dataset",
                filename=rel_path,
            )
            table = pq.read_table(local_path)
            records = table.to_pylist()
            all_records.extend(records)
            print(f"{len(records)} records")
        except Exception as e:
            print(f"skipped ({e})")

    total = len(all_records)
    print(f"  Loaded {total} total examples ({split} split)")

    if max_examples and max_examples < total:
        print(f"  Sampling {max_examples} examples (seed={seed})...")
        indices = sorted(random.Random(seed).sample(range(total), max_examples))
        all_records = [all_records[i] for i in indices]
        total = max_examples

    OOLONG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(OOLONG_CACHE_JSON, "w") as f:
        json.dump(all_records, f)

    elapsed = time.time() - t0
    print(
        f"  Cached to {OOLONG_CACHE_JSON} ({len(all_records)} examples, {elapsed:.1f}s)"
    )
    return all_records


def _load_cached_oolong(
    split: str = "validation",
    max_examples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load OOLONG dataset from cache, or download if not cached."""
    if not OOLONG_CACHE_JSON.exists():
        return _download_oolong_cache(split=split, max_examples=max_examples, seed=seed)
    print(f"Loading OOLONG from cache: {OOLONG_CACHE_JSON}")
    with open(OOLONG_CACHE_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# OOLONG adapter
# ---------------------------------------------------------------------------


class OOLONGContext:
    """Adapter for OOLONG dataset to our benchmark framework."""

    def __init__(
        self,
        split: str = "validation",
        max_examples: int | None = None,
        seed: int = 42,
        force_download: bool = False,
    ):
        self.split = split
        self.max_examples = max_examples
        self.seed = seed
        self.force_download = force_download

    def load(self) -> list[OOLONGTestCase]:
        """Load OOLONG examples."""
        raw_data = _load_cached_oolong(
            split=self.split, max_examples=self.max_examples, seed=self.seed
        )
        cases = []
        for item in raw_data:
            cases.append(
                OOLONGTestCase(
                    id=str(item["id"]),
                    context_window_text=item["context_window_text"],
                    question=item["question"],
                    answer=item["answer"],
                    answer_type=item["answer_type"],
                    task=item["task"],
                    task_group=item["task_group"],
                    dataset=item["dataset"],
                    context_len=item["context_len"],
                    input_subset=item["input_subset"],
                    num_labels=item["num_labels"],
                )
            )
        return cases

    def load_as_test_cases(self, context_window: int | None = None) -> list[TestCase]:
        """Load and convert to standard TestCase format."""
        oolong_cases = self.load()
        return [c.to_test_case(context_window) for c in oolong_cases]

    def info(self) -> dict[str, Any]:
        """Return dataset statistics."""
        raw_data = _load_cached_oolong(split=self.split)

        task_groups: dict[str, int] = {}
        datasets: dict[str, int] = {}
        answer_types: dict[str, int] = {}

        for item in raw_data:
            task_groups[item["task_group"]] = task_groups.get(item["task_group"], 0) + 1
            datasets[item["dataset"]] = datasets.get(item["dataset"], 0) + 1
            answer_types[item["answer_type"]] = (
                answer_types.get(item["answer_type"], 0) + 1
            )

        return {
            "split": self.split,
            "total_examples": len(raw_data),
            "task_groups": dict(sorted(task_groups.items())),
            "datasets": dict(sorted(datasets.items())),
            "answer_types": dict(sorted(answer_types.items())),
        }


# ---------------------------------------------------------------------------
# Context window validation
# ---------------------------------------------------------------------------


def check_model_context_capacity(model_name: str) -> dict[str, Any]:
    """Check if a model can handle OOLONG context sizes."""
    capacity = MODEL_CONTEXT_CAPACITY.get(model_name, 32_000)
    oolong_sizes = [4_000, 16_000, 64_000, 128_000, 256_000]
    allowed = [s for s in oolong_sizes if s <= capacity * 0.9]

    return {
        "model": model_name,
        "estimated_capacity": capacity,
        "oolong_sizes": oolong_sizes,
        "allowed_sizes": allowed,
        "skipped_sizes": [s for s in oolong_sizes if s not in allowed],
        "warning": None,
    }
