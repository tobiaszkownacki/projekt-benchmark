"""
Code to create features for modeling
"""

from pathlib import Path

from src.config import PROCESSED_DATA_DIR


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):

    pass


if __name__ == "__main__":
    main()
