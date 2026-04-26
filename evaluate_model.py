"""
Evaluate PCA-based drone detector predictions against YOLO ground truth labels.

Data formats:
- Ground truth (YOLO): class cx cy w h          (normalized [0, 1])
- Predictions:         score x y w h            (absolute pixels)

This script converts both formats to absolute XYXY boxes:
[x_min, y_min, x_max, y_max]

Then it computes:
- TP, FP, FN using IoU matching at a chosen threshold (default 0.5)
- Precision, Recall, F1-score
- AP@0.5 (single-class), reported as mAP@0.5 for convenience

Usage examples:
    python evaluate_model.py
    python evaluate_model.py --iou-thr 0.5 --max-images 100
    python evaluate_model.py --gt-labels-dir archive/drone_dataset_yolo/dataset_txt --pred-dir data/res_2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# =========================
# PATH CONFIGURATION
# =========================
# Update these three paths if your folder structure changes.
PROJECT_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_LABELS_DIR = PROJECT_ROOT / "archive" / "drone_dataset_yolo" / "dataset_txt"
GROUND_TRUTH_IMAGES_DIR = PROJECT_ROOT / "archive" / "drone_dataset_yolo" / "dataset_txt"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "res_2"


@dataclass
class BoxRecord:
    """One bounding box in absolute XYXY format."""

    image_id: str
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute IoU between two boxes in XYXY format.

    Args:
        box_a: [x_min, y_min, x_max, y_max]
        box_b: [x_min, y_min, x_max, y_max]

    Returns:
        IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized YOLO box (cx, cy, w, h) to absolute XYXY pixels."""
    x_min = (cx - w / 2.0) * img_w
    y_min = (cy - h / 2.0) * img_h
    x_max = (cx + w / 2.0) * img_w
    y_max = (cy + h / 2.0) * img_h

    # Clamp to image bounds to avoid invalid coordinates from noisy labels.
    x_min = float(np.clip(x_min, 0.0, float(img_w)))
    y_min = float(np.clip(y_min, 0.0, float(img_h)))
    x_max = float(np.clip(x_max, 0.0, float(img_w)))
    y_max = float(np.clip(y_max, 0.0, float(img_h)))
    return x_min, y_min, x_max, y_max


def pred_xywh_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert prediction [x, y, w, h] pixels to XYXY; return None if invalid."""
    if w <= 0.0 or h <= 0.0:
        return None
    return float(x), float(y), float(x + w), float(y + h)


def read_image_shape(image_path: Path) -> Tuple[int, int]:
    """Return (height, width) for an image file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    return h, w


def parse_yolo_file(
    yolo_path: Path,
    image_id: str,
    img_w: int,
    img_h: int,
) -> List[BoxRecord]:
    """
    Parse one YOLO label file into XYXY boxes.

    Expected line format:
        class cx cy w h
    """
    records: List[BoxRecord] = []
    if not yolo_path.exists():
        return records

    for raw_line in yolo_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            # class_id is parsed but not used because this is a single-class evaluation.
            _class_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue

        bbox = yolo_to_xyxy(cx, cy, w, h, img_w=img_w, img_h=img_h)
        records.append(BoxRecord(image_id=image_id, bbox_xyxy=bbox, confidence=1.0))

    return records


def parse_prediction_file(pred_path: Path, image_id: str) -> List[BoxRecord]:
    """
    Parse one prediction file into XYXY boxes.

    Expected line format:
        score x y w h

    Note:
        In your detector, lower distance score is better.
        To use a standard AP pipeline (descending confidence),
        confidence is defined as -score.
    """
    records: List[BoxRecord] = []
    if not pred_path.exists():
        return records

    for raw_line in pred_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            score, x, y, w, h = map(float, parts)
        except ValueError:
            continue

        bbox = pred_xywh_to_xyxy(x, y, w, h)
        if bbox is None:
            continue

        confidence = -score
        records.append(BoxRecord(image_id=image_id, bbox_xyxy=bbox, confidence=confidence))

    return records


def find_image_path(images_dir: Path, image_id: str) -> Optional[Path]:
    """Find the actual image file for an image id by trying common extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def match_image_predictions(
    pred_boxes: List[BoxRecord],
    gt_boxes: List[BoxRecord],
    iou_thr: float,
) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching for a single image.

    A prediction is TP if it matches one unmatched GT with IoU >= iou_thr.
    Remaining predictions are FP; unmatched GT boxes are FN.
    """
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0

    pred_sorted = sorted(pred_boxes, key=lambda r: r.confidence, reverse=True)
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)

    tp = 0
    fp = 0

    for pred in pred_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if matched_gt[gt_idx]:
                continue
            iou = calculate_iou(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr and best_gt_idx >= 0:
            tp += 1
            matched_gt[best_gt_idx] = True
        else:
            fp += 1

    fn = int((~matched_gt).sum())
    return tp, fp, fn


def compute_ap50(
    all_predictions: List[BoxRecord],
    gt_by_image: Dict[str, List[BoxRecord]],
    iou_thr: float,
) -> float:
    """
    Compute AP at one IoU threshold for a single class.

    Steps:
    1. Sort all predictions by confidence descending.
    2. Mark each prediction as TP/FP with one-to-one GT matching per image.
    3. Build precision-recall curve.
    4. Integrate using precision envelope (VOC-style continuous area).
    """
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    preds_sorted = sorted(all_predictions, key=lambda r: r.confidence, reverse=True)

    # Track which GT boxes are already claimed by earlier (higher-confidence) preds.
    matched_flags: Dict[str, np.ndarray] = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in gt_by_image.items()
    }

    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for pred in preds_sorted:
        gt_boxes = gt_by_image.get(pred.image_id, [])
        gt_matched = matched_flags.get(pred.image_id)

        if not gt_boxes:
            tp_flags.append(0)
            fp_flags.append(1)
            continue

        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if gt_matched is not None and gt_matched[idx]:
                continue
            iou = calculate_iou(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thr and best_idx >= 0:
            tp_flags.append(1)
            fp_flags.append(0)
            if gt_matched is not None:
                gt_matched[best_idx] = True
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum = np.cumsum(tp_flags, dtype=np.float64)
    fp_cum = np.cumsum(fp_flags, dtype=np.float64)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / max(float(total_gt), 1e-12)

    # Precision envelope integration.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))
    return ap


def evaluate_dataset(
    gt_labels_dir: Path,
    gt_images_dir: Path,
    pred_dir: Path,
    iou_thr: float = 0.5,
    max_images: Optional[int] = 100,
) -> None:
    """Run full dataset evaluation and print a clean metric summary."""
    if not gt_labels_dir.exists():
        raise FileNotFoundError(f"Ground-truth labels directory not found: {gt_labels_dir}")
    if not gt_images_dir.exists():
        raise FileNotFoundError(f"Ground-truth images directory not found: {gt_images_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    pred_files = sorted([p for p in pred_dir.glob("*.txt") if p.is_file()])
    if max_images is not None:
        pred_files = pred_files[:max_images]

    if not pred_files:
        raise RuntimeError(f"No prediction .txt files found in: {pred_dir}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    skipped_images = 0

    all_predictions: List[BoxRecord] = []
    gt_by_image: Dict[str, List[BoxRecord]] = {}

    for pred_path in pred_files:
        image_id = pred_path.stem

        image_path = find_image_path(gt_images_dir, image_id)
        if image_path is None:
            skipped_images += 1
            continue

        img_h, img_w = read_image_shape(image_path)

        gt_path = gt_labels_dir / f"{image_id}.txt"
        gt_boxes = parse_yolo_file(gt_path, image_id=image_id, img_w=img_w, img_h=img_h)
        pred_boxes = parse_prediction_file(pred_path, image_id=image_id)

        tp, fp, fn = match_image_predictions(pred_boxes, gt_boxes, iou_thr=iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        gt_by_image[image_id] = gt_boxes
        all_predictions.extend(pred_boxes)

    precision = total_tp / max((total_tp + total_fp), 1e-12)
    recall = total_tp / max((total_tp + total_fn), 1e-12)
    f1 = 2.0 * precision * recall / max((precision + recall), 1e-12)

    ap50 = compute_ap50(all_predictions, gt_by_image, iou_thr=iou_thr)

    evaluated_images = len(pred_files) - skipped_images

    print("\n" + "=" * 72)
    print("PCA Drone Detector Evaluation Summary")
    print("=" * 72)
    print(f"Ground-truth labels dir : {gt_labels_dir}")
    print(f"Ground-truth images dir : {gt_images_dir}")
    print(f"Predictions dir         : {pred_dir}")
    print(f"IoU threshold           : {iou_thr:.2f}")
    print(f"Images evaluated        : {evaluated_images}")
    print(f"Images skipped          : {skipped_images}")

    print("\nCounts")
    print(f"  TP: {total_tp}")
    print(f"  FP: {total_fp}")
    print(f"  FN: {total_fn}")

    print("\nMetrics")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AP@0.5    : {ap50:.4f}")
    print(f"  mAP@0.5   : {ap50:.4f}  (single-class)")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    """CLI for flexible path and threshold control."""
    parser = argparse.ArgumentParser(description="Evaluate drone detector predictions vs YOLO labels.")

    parser.add_argument("--gt-labels-dir", type=Path, default=GROUND_TRUTH_LABELS_DIR)
    parser.add_argument("--gt-images-dir", type=Path, default=GROUND_TRUTH_IMAGES_DIR)
    parser.add_argument("--pred-dir", type=Path, default=PREDICTIONS_DIR)

    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for TP matching.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Evaluate first N prediction files after sorting; set -1 for all.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_images = None if args.max_images is not None and args.max_images < 0 else args.max_images

    evaluate_dataset(
        gt_labels_dir=args.gt_labels_dir,
        gt_images_dir=args.gt_images_dir,
        pred_dir=args.pred_dir,
        iou_thr=float(args.iou_thr),
        max_images=max_images,
    )


if __name__ == "__main__":
    main()
