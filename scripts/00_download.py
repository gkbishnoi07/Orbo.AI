"""Fetch the raw dataset from Kaggle into data/raw/.

The dataset is not redistributed in this repository — it is ~1.4GB and carries
Kaggle's licence terms — so this script is how another engineer reproduces the
input. It needs credentials in ~/.kaggle/kaggle.json (Kaggle -> Settings ->
Create New Token).

Run:  python scripts/00_download.py
      python scripts/00_download.py --force    # re-download over existing files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET = "nadyinky/sephora-products-and-skincare-reviews"
DEFAULT_DEST = Path("data/raw")


def already_downloaded(dest: Path) -> list[Path]:
    return sorted(dest.glob("*.csv"))


def download(dest: Path, *, force: bool) -> int:
    existing = already_downloaded(dest)
    if existing and not force:
        print(f"{len(existing)} CSV(s) already in {dest}; nothing to do.")
        for path in existing:
            print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        print("Pass --force to re-download.")
        return 0

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The kaggle client is missing. Install the dev requirements:\n"
            "  pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        return 1

    # Imported this way rather than `import kaggle`, which authenticates as a
    # side effect of import and dies with a bare stack trace if the token is
    # absent. Here we control the error message.
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - credentials are the usual cause
        print(
            f"Kaggle authentication failed: {exc}\n"
            "Expected credentials at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} -> {dest} (this takes a few minutes)")
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)

    files = already_downloaded(dest)
    if not files:
        print("Download reported success but no CSVs landed.", file=sys.stderr)
        return 1

    print(f"\n{len(files)} CSV(s) in {dest}:")
    for path in files:
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext: python scripts/01_inspect.py data/raw")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", nargs="?", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if CSVs exist"
    )
    args = parser.parse_args()
    return download(args.dest, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
