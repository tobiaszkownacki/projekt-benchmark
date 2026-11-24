"""Preprocess the raw datasets into the files the benchmark discovers.

The raw data lives in a directory that is never committed (``data/private/`` by
default):

    csv/            raw CSV files
    prepare/<id>.py ``prepare(df: pandas.DataFrame) -> (X, y)``, X float32 2-D, y int64 1-D
    manifest.toml   which datasets exist, see below

Each dataset is written to ``src/benchmark_core/datasets/data/<id>.npz``, which
is all it takes to register it.

manifest.toml::

    [datasets.ds01]
    csv = "some_file.csv"
    # optional: prepare = "ds01" (module in prepare/, defaults to the id)
    # optional: read_csv = { sep = ";" }

Usage:
    uv run python scripts/prepare_datasets.py
    uv run python scripts/prepare_datasets.py --only ds03
"""

import argparse
import importlib.util
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark_core import dataset_files  # noqa: E402

OUT_DIR = ROOT / "src" / "benchmark_core" / "datasets" / "data"


def load_prepare(path: Path):
    spec = importlib.util.spec_from_file_location(f"prepare_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "prepare"):
        raise SystemExit(f"{path} defines no prepare(df) function")
    return module.prepare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--private",
        type=Path,
        default=ROOT / "data" / "private",
        help="folder with csv/, prepare/ and manifest.toml (default: data/private)",
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="prepare only these dataset ids")
    args = parser.parse_args()

    private: Path = args.private
    manifest = tomllib.loads((private / "manifest.toml").read_text())["datasets"]
    if args.only:
        unknown = set(args.only) - set(manifest)
        if unknown:
            raise SystemExit(f"not in manifest: {sorted(unknown)}")
        manifest = {k: v for k, v in manifest.items() if k in args.only}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, entry in manifest.items():
        df = pd.read_csv(private / "csv" / entry["csv"], **entry.get("read_csv", {}))
        prepare = load_prepare(private / "prepare" / f"{entry.get('prepare', dataset_id)}.py")
        X, y = prepare(df)
        X, y = np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(y, dtype=np.int64)

        out_path = OUT_DIR / f"{dataset_id}.npz"
        dataset_files.save_arrays(out_path, X, y)
        print(f"{dataset_id}: X {X.shape}, {int(y.max()) + 1} classes, {out_path.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
