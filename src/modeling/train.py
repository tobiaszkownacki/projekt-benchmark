"""
Code to train models
"""

from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    pass


if __name__ == "__main__":
    main()
