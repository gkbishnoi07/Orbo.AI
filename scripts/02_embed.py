"""Precompute product vectors offline and commit them.

The deployed app must not import sentence-transformers: torch would blow the
free tier's disk budget and add seconds to cold start for a matrix that never
changes. So the transformer runs here, once, and the vectors ship as .npy.

Two methods are supported, and which one wins is an empirical question rather
than an obvious one. The text these vectors describe is not prose — there is no
description field in this dataset, so the blob is brand, name, category,
highlight tokens and an INCI ingredient list. That is keyword soup, which is
TF-IDF's home ground; a sentence transformer earns its place only if semantic
generalisation ("Hydrating" ~ "moisture") beats exact term overlap. Both are
built here and compared in scripts/04_evaluate.py.

Run:  python scripts/02_embed.py              # both methods
      python scripts/02_embed.py --method tfidf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load  # noqa: E402
from src.schema import ProductCols  # noqa: E402

ARTIFACTS = Path("artifacts")
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_DIMS = 256


def build_tfidf(texts: list[str]) -> np.ndarray:
    """TF-IDF over word and character n-grams, reduced by SVD.

    Character n-grams matter here: ingredient names are long, inconsistently
    spelled and often hyphenated, so word tokens alone miss that "Sodium
    Hyaluronate" and "Hyaluronic Acid" are related.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import Normalizer

    # Both vocabularies are capped. Uncapped, the character analyser generates
    # hundreds of thousands of features from 8,494 INCI lists and the fit stops
    # being worth the wait for a matrix that barely changes.
    words = TfidfVectorizer(
        sublinear_tf=True,
        min_df=3,
        max_df=0.6,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=60_000,
    )
    chars = TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        analyzer="char_wb",
        ngram_range=(4, 5),
        max_features=30_000,
    )
    pipeline = make_pipeline(
        make_union(words, chars),
        TruncatedSVD(n_components=TFIDF_DIMS, random_state=0),
        Normalizer(copy=False),
    )
    return pipeline.fit_transform(texts).astype(np.float32)


def build_minilm(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_MODEL)
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


BUILDERS = {"tfidf": build_tfidf, "minilm": build_minilm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*BUILDERS, "both"], default="both")
    args = parser.parse_args()

    products, _ = load()
    texts = products[ProductCols.TEXT].fillna("").astype(str).tolist()
    ids = products[ProductCols.ID].astype(str).tolist()
    print(f"{len(texts):,} products to embed")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Row order is the contract between the matrix and the catalogue. Written
    # once, alongside, so a stale matrix can never be silently mis-indexed.
    (ARTIFACTS / "embedding_ids.json").write_text(json.dumps(ids))

    methods = list(BUILDERS) if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n--- {method} ---")
        try:
            matrix = BUILDERS[method](texts)
        except ImportError as exc:
            print(f"  skipped: {exc}")
            continue
        assert matrix.shape[0] == len(ids), "row count must match the catalogue"
        out = ARTIFACTS / f"embeddings_{method}.npy"
        np.save(out, matrix)
        print(f"  {matrix.shape} -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
