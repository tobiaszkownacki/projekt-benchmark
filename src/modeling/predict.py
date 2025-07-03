"""
Code to run model inference with trained models
"""

from pathlib import Path
import torch

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    pass


def cma_predict(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    best_accuracy = correct / total
    return best_accuracy


if __name__ == "__main__":
    main()
