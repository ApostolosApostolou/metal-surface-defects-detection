from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import cv2
import random

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

# Path to label files for test images
TEST_IMAGES_DIR = Path("dataset_yolo") / "test" / "images"
TEST_LABELS_DIR = Path("dataset_yolo") / "test" / "labels"

# Helper function to print metrics
def print_metrics(label: str, metrics):
    box = metrics.box

    print(f"\n=== {label} ===")
    print(f"Mean Precision (mp):     {box.mp:.4f}")
    print(f"Mean Recall (mr):        {box.mr:.4f}")
    print(f"mAP@0.50 (map50):        {box.map50:.4f}")
    print(f"mAP@0.50–0.95 (map):     {box.map:.4f}")


# Load YOLO labels (.txt)
def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])

            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


# Plot test predictions with labels
def plot_test_predictions_grid(model: YOLO, num_images: int = 20, rows: int = 4, cols: int = 5):

    image_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print("No test images found to visualize.")
        return

    num_images = min(num_images, len(image_paths))
    sampled_paths = random.sample(image_paths, num_images)

    plt.figure(figsize=(cols * 3, rows * 3))

    for idx, img_path in enumerate(sampled_paths):

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Ground truth
        label_path = TEST_LABELS_DIR / f"{img_path.stem}.txt"
        gt_boxes = load_yolo_labels(label_path, w, h)

        # Predictions
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )

        pred_boxes = []
        if results and len(results) > 0:
            r = results[0]
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                pred_boxes.append((cls_id, x1, y1, x2, y2))

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img_rgb)
        ax.axis("off")

        # Draw GT boxes (green)
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, linewidth=2, edgecolor="g",
                )
            )

        ax.text(
            x2 - 3,            # x2 = right side of box
            y2 - 3,            # y2 = bottom of box
            f"GT:{cls_id}",
            color="g",
            fontsize=8,
            ha="right",        # align text to the right
            va="bottom",       # align text to bottom
            backgroundcolor="black"
        )
        
        # Draw Predicted boxes (red)
        for cls_id, x1, y1, x2, y2 in pred_boxes:
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, linewidth=2, edgecolor="r",
                )
            )
            ax.text(
                x1 + 3, # x1 = left side of box
                y1 + 12, # y1 = top of box
                f"PR:{cls_id}",
                color="r",
                fontsize=8,
                backgroundcolor="black"
            )

        ax.set_title(img_path.name, fontsize=8)

    plt.suptitle(
        "Test Set Predictions (PR) vs Ground Truth (GT)\nGreen = GT   |   Red = Pred",
        fontsize=15,
        y=1.02
    )
    plt.subplots_adjust(top=0.88)
    plt.tight_layout()
    plt.savefig("test_predictions_grid.png", dpi=200)
    plt.show()
    print("Saved visualization grid to test_predictions_grid.png")


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"best.pt not found at: {BEST_WEIGHTS}")
    if not LAST_WEIGHTS.exists():
        raise FileNotFoundError(f"last.pt not found at: {LAST_WEIGHTS}")

    print("\nLoading best model for VALIDATION...")
    model_best = YOLO(str(BEST_WEIGHTS))

    val_metrics_best = model_best.val(
        data=str(DATASET_YAML),
        split="val",
        device=device,
        verbose=False,
        workers=0
    )
    print_metrics("Validation (best.pt)", val_metrics_best)

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

    print("\nLoading best model for TEST evaluation...")
    test_metrics = model_best.val(
        data=str(DATASET_YAML),
        split="test",
        device=device,
        verbose=False,
        workers=0
    )
    print_metrics("Test Set (best.pt)", test_metrics)

    print("\nPlotting random test predictions grid...")
    plot_test_predictions_grid(model_best, num_images=20, rows=4, cols=5)

    print("\nDone.")


if __name__ == "__main__":
    main()
