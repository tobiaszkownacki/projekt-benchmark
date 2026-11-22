"""Preprocess the competition datasets and encrypt them.

Everything dataset-specific lives in a private directory that is never
committed (``data/private/`` by default):

    csv/            raw CSV files
    prepare/<id>.py ``prepare(df: pandas.DataFrame) -> (X, y)``, X float32 2-D, y int64 1-D
    manifest.toml   which datasets exist, see below
    key             base64 AES-256 key (created with --gen-key)
    encrypted/      output: <id>.bin

manifest.toml::

    [datasets.ds01]
    csv = "some_file.csv"
    # optional: prepare = "ds01" (module in prepare/, defaults to the id)
    # optional: read_csv = { sep = ";" }

Usage:
    uv run python scripts/encrypt_datasets.py --gen-key --verify
    uv run python scripts/encrypt_datasets.py --only ds03
"""

import argparse
import importlib.util
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark_core import dataset_crypto  # noqa: E402


def load_prepare(path: Path):
    spec = importlib.util.spec_from_file_location(f"prepare_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "prepare"):
        raise SystemExit(f"{path} defines no prepare(df) function")
    return module.prepare


def read_key(private: Path, generate: bool, force: bool) -> bytes:
    key_path = private / "key"
    if generate:
        if key_path.exists() and not force:
            raise SystemExit(f"{key_path} already exists; a new key makes every existing .bin unreadable (--force)")
        key_path.write_text(dataset_crypto.generate_key() + "\n")
        key_path.chmod(0o600)
        print(f"new key written to {key_path}")
    if not key_path.exists():
        raise SystemExit(f"no key at {key_path}; create one with --gen-key")
    return dataset_crypto.decode_key(key_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--private",
        type=Path,
        default=Path("data/private"),
        help="folder with csv/, prepare/, manifest.toml and key (default: data/private)",
    )
    parser.add_argument("--gen-key", action="store_true", help="create a new key before encrypting")
    parser.add_argument("--force", action="store_true", help="allow --gen-key to replace an existing key")
    parser.add_argument("--verify", action="store_true", help="decrypt every output and compare it to the input")
    parser.add_argument("--only", nargs="+", metavar="ID", help="encrypt only these dataset ids")
    args = parser.parse_args()

    private: Path = args.private
    manifest = tomllib.loads((private / "manifest.toml").read_text())["datasets"]
    if args.only:
        unknown = set(args.only) - set(manifest)
        if unknown:
            raise SystemExit(f"not in manifest: {sorted(unknown)}")
        manifest = {k: v for k, v in manifest.items() if k in args.only}

    key = read_key(private, args.gen_key, args.force)
    out_dir = private / "encrypted"
    out_dir.mkdir(exist_ok=True)

    for dataset_id, entry in manifest.items():
        df = pd.read_csv(private / "csv" / entry["csv"], **entry.get("read_csv", {}))
        prepare = load_prepare(private / "prepare" / f"{entry.get('prepare', dataset_id)}.py")
        X, y = prepare(df)
        X, y = np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(y, dtype=np.int64)

        blob = dataset_crypto.encrypt_arrays(X, y, dataset_id, key)
        out_path = out_dir / f"{dataset_id}.bin"
        out_path.write_bytes(blob)

        line = f"{dataset_id}: X {X.shape}, {int(y.max()) + 1} classes, {len(blob) / 1024:.0f} KiB -> {out_path}"
        if args.verify:
            X2, y2, _ = dataset_crypto.load_encrypted(out_path, dataset_id, key)
            if not (np.array_equal(X, X2) and np.array_equal(y, y2)):
                raise SystemExit(f"{dataset_id}: decrypted data differs from the input")
            line += " [verified]"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
