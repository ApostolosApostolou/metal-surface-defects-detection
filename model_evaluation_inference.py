from pathlib import Path
from ultralytics import YOLO
import torch

# Path to the YOLO dataset YAML 
DATASET_YAML = Path("dataset_yolo") / "neu_det.yaml"

# Path to the directory where training runs are saved
RUNS_DIR = Path("runs_yolo11")

# Name of the run to evaluate
RUN_NAME = "yolo11n_neu_det_e100_b4_img640"

# Paths to weights
RUN_DIR = RUNS_DIR / RUN_NAME
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
LAST_WEIGHTS = RUN_DIR / "weights" / "last.pt"

# Helper function to print metrics
def print_metrics(label: str, metrics):
    box = metrics.box  # this holds box-level metrics (precision, recall, mAP, etc.)

    print(f"\n=== {label} ===")
    print(f"Mean Precision (mp):     {box.mp:.4f}")
    print(f"Mean Recall (mr):        {box.mr:.4f}")
    print(f"mAP@0.50 (map50):        {box.map50:.4f}")
    print(f"mAP@0.50–0.95 (map):     {box.map:.4f}")


def main():
    # Check device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if weights exist
    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"best.pt not found at: {BEST_WEIGHTS}")
    if not LAST_WEIGHTS.exists():
        raise FileNotFoundError(f"last.pt not found at: {LAST_WEIGHTS}")

    # Validation on best weights
    print("\nLoading best model for VALIDATION...")
    model_best = YOLO(str(BEST_WEIGHTS))

    val_metrics_best = model_best.val(
        data=str(DATASET_YAML),
        split="val",   # explicitly use validation split
        device=device,
        verbose=False,  # set True if you want the full Ultralytics printout
        workers=0  # set to 0 for Windows compatibility
    )
    print_metrics("Validation (best.pt)", val_metrics_best)

    # Validation on last weights
    print("\nLoading last model for VALIDATION...")
    model_last = YOLO(str(LAST_WEIGHTS))

    val_metrics_last = model_last.val(
        data=str(DATASET_YAML),
        split="val",
        device=device,
        verbose=False,
        workers=0  
    )
    print_metrics("Validation (last.pt)", val_metrics_last)

    # Test set on best weights
    print("\nLoading best model for TEST evaluation...")
    test_metrics = model_best.val(
        data=str(DATASET_YAML),
        split="test",   # use test split defined in neu_det.yaml
        device=device,
        verbose=False,
        workers=0  
    )
    print_metrics("Test Set (best.pt)", test_metrics)

    print("\nDone.")

if __name__ == "__main__":
    main()
