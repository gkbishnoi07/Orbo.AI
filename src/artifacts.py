"""Loading the precomputed files the app and the evaluation both read.

Everything here is committed to the repo, because the deployed app has no
Kaggle credentials and cannot parse 530MB of CSV on a cold start. The rule is
that `artifacts/` holds only what is small, derived, and needed at request time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path("artifacts")

EMBEDDING_IDS = "embedding_ids.json"
PRODUCTS = "products.parquet"
INTERACTIONS = "interactions.parquet"
COHORT_STATS = "cohort_stats.parquet"


def embedding_path(method: str, root: Path = ARTIFACTS) -> Path:
    return Path(root) / f"embeddings_{method}.npy"


def available_methods(root: Path = ARTIFACTS) -> list[str]:
    return sorted(
        p.stem.removeprefix("embeddings_") for p in Path(root).glob("embeddings_*.npy")
    )


def load_embeddings(
    method: str, root: Path = ARTIFACTS
) -> tuple[np.ndarray, list[str]]:
    """Return (matrix, ids). The id list is the row-order contract."""
    root = Path(root)
    matrix = np.load(embedding_path(method, root))
    ids = json.loads((root / EMBEDDING_IDS).read_text())
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"{method}: matrix has {matrix.shape[0]} rows but "
            f"{EMBEDDING_IDS} lists {len(ids)} products — rebuild artifacts"
        )
    return matrix, [str(i) for i in ids]


def as_lookup(matrix: np.ndarray, ids: list[str]) -> dict[str, np.ndarray]:
    """Row-per-product dict, which is what the diversity metric expects."""
    return {pid: matrix[row] for row, pid in enumerate(ids)}


def load_products(root: Path = ARTIFACTS) -> pd.DataFrame:
    frame = pd.read_parquet(Path(root) / PRODUCTS)
    for column in ("highlights", "suits_skin_types", "addresses_concerns"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda v: [str(x) for x in v] if v is not None else []
            )
    return frame


def load_interactions(root: Path = ARTIFACTS) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / INTERACTIONS)


def load_cohort_stats(root: Path = ARTIFACTS) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / COHORT_STATS)
