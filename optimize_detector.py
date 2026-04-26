from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from evaluate_model import calculate_iou


PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_DIR = PROJECT_ROOT / "archive" / "drone_dataset_yolo" / "dataset_txt"
BG_DIR = PROJECT_ROOT / "archive" / "Dataset" / "train" / "images"

IMG_SIZE = (64, 64)
DRONE_LIMIT = 700
SEED = 42


@dataclass
class Params:
    variance_thr: float
    nn_margin: float
    nms_iou: float
    std_threshold: float
    scales: Tuple[float, ...]
    stride: int = 16
    top_k: int = 8
    pad_ratio: float = 0.0
    hard_negative_mining: bool = False
    hnm_images: int = 25
    hnm_max_patches: int = 600


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap50: float


@dataclass
class TrialResult:
    phase: str
    trial_name: str
    params: Params
    metrics: Metrics
    images_evaluated: int


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def extract_features(image: np.ndarray, target_size: Tuple[int, int] = IMG_SIZE) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.float64) / 255.0


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    x1 = float(np.clip(x1, 0, w))
    y1 = float(np.clip(y1, 0, h))
    x2 = float(np.clip(x2, 0, w))
    y2 = float(np.clip(y2, 0, h))
    return x1, y1, x2, y2


def parse_gt_yolo(txt_path: Path, w: int, h: int) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _, cx, cy, bw, bh = parts
            boxes.append(yolo_to_xyxy(float(cx), float(cy), float(bw), float(bh), w, h))
        except ValueError:
            continue
    return boxes


def load_drone_patches(dataset_folder: Path, limit: int, pad_ratio: float) -> np.ndarray:
    patches: List[np.ndarray] = []
    for img_path in list_image_files(dataset_folder):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _, cx, cy, bw, bh = map(float, parts)
            except ValueError:
                continue

            w_pix, h_pix = bw * w, bh * h
            pad_w, pad_h = w_pix * pad_ratio, h_pix * pad_ratio

            x1 = int((cx * w) - (w_pix / 2) - pad_w)
            y1 = int((cy * h) - (h_pix / 2) - pad_h)
            x2 = int((cx * w) + (w_pix / 2) + pad_w)
            y2 = int((cy * h) + (h_pix / 2) + pad_h)

            crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            feat = extract_features(crop)
            if feat is not None:
                patches.append(feat)

            if len(patches) >= limit:
                return np.array(patches[:limit], dtype=np.float64)
    return np.array(patches[:limit], dtype=np.float64)


def load_background_patches(dataset_folder: Path, n_patches: int, seed: int = SEED) -> np.ndarray:
    patches: List[np.ndarray] = []
    rng = np.random.default_rng(seed)
    files = list_image_files(dataset_folder)
    if not files:
        return np.empty((0, IMG_SIZE[0] * IMG_SIZE[1]), dtype=np.float64)

    while len(patches) < n_patches:
        img_path = rng.choice(files)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < IMG_SIZE[1] or w < IMG_SIZE[0]:
            continue
        x = int(rng.integers(0, w - IMG_SIZE[0] + 1))
        y = int(rng.integers(0, h - IMG_SIZE[1] + 1))
        feat = extract_features(img[y : y + IMG_SIZE[1], x : x + IMG_SIZE[0]])
        if feat is not None:
            patches.append(feat)

    return np.array(patches[:n_patches], dtype=np.float64)


def build_pca(X: np.ndarray, variance_threshold: float, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    n, _d = X.shape
    mean_vec = X.mean(axis=0)
    Xc = X - mean_vec
    C = Xc @ Xc.T / max(n, 1)

    total_var = float(np.trace(C))
    if total_var <= 1e-12:
        return mean_vec, np.eye(X.shape[1], 1, dtype=np.float64)

    rng = np.random.default_rng(seed)
    evecs_n: List[np.ndarray] = []
    evecs_d: List[np.ndarray] = []
    cumul = 0.0

    for _ in range(n):
        if cumul / total_var >= variance_threshold:
            break

        v = rng.standard_normal(n)
        v /= np.linalg.norm(v) + 1e-12
        for u in evecs_n:
            v -= (v @ u) * u
        v /= np.linalg.norm(v) + 1e-12

        for _it in range(60):
            vn = C @ v
            for u in evecs_n:
                vn -= (vn @ u) * u
            nrm = np.linalg.norm(vn)
            if nrm < 1e-12:
                break
            vn /= nrm
            if min(np.linalg.norm(vn - v), np.linalg.norm(vn + v)) < 1e-7:
                v = vn
                break
            v = vn

        eigval = float(v @ C @ v)
        if eigval <= 1e-12:
            continue
        evecs_n.append(v)
        cumul += eigval

        y = Xc.T @ v
        yn = np.linalg.norm(y)
        if yn > 1e-12:
            evecs_d.append(y / yn)

    if not evecs_d:
        return mean_vec, np.eye(X.shape[1], 1, dtype=np.float64)
    return mean_vec, np.column_stack(evecs_d)


def min_dist(A: np.ndarray, B: np.ndarray, chunk: int = 512) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if len(B) == 0:
        return np.full(len(A), np.inf, dtype=np.float64)
    A_sq = np.einsum("ij,ij->i", A, A)[:, None]
    B_sq = np.einsum("ij,ij->i", B, B)[None, :]
    out = np.full(len(A), np.inf, dtype=np.float64)
    for start in range(0, len(B), chunk):
        Bi = B[start : start + chunk]
        d2 = np.maximum(A_sq + B_sq[:, start : start + len(Bi)] - 2.0 * (A @ Bi.T), 0.0)
        out = np.minimum(out, d2.min(axis=1))
    return np.sqrt(out)


def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    return calculate_iou((ax1, ay1, ax1 + aw, ay1 + ah), (bx1, by1, bx1 + bw, by1 + bh))


def nms(dets: List[Tuple[float, int, int, int, int]], iou_thr: float) -> List[Tuple[float, int, int, int, int]]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d[0])
    keep: List[Tuple[float, int, int, int, int]] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou_xywh(best[1:], d[1:]) < iou_thr]
    return keep


def detect_single(
    gray: np.ndarray,
    mean_vec: np.ndarray,
    evecs: np.ndarray,
    Z_drone: np.ndarray,
    Z_bg: np.ndarray,
    params: Params,
) -> List[Tuple[float, int, int, int, int]]:
    ph, pw = IMG_SIZE
    dets: List[Tuple[float, int, int, int, int]] = []

    for s in params.scales:
        sh = int(gray.shape[0] * s)
        sw = int(gray.shape[1] * s)
        if sh < ph or sw < pw:
            continue

        simg = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        feats: List[np.ndarray] = []
        pos: List[Tuple[int, int]] = []

        for y in range(0, sh - ph + 1, params.stride):
            for x in range(0, sw - pw + 1, params.stride):
                raw = simg[y : y + ph, x : x + pw]
                if float(np.std(raw)) < params.std_threshold:
                    continue
                feat = extract_features(raw)
                if feat is not None:
                    feats.append(feat)
                    pos.append((x, y))

        if not feats:
            continue

        Z = (np.array(feats, dtype=np.float64) - mean_vec) @ evecs
        d_drone = min_dist(Z, Z_drone)
        d_bg = min_dist(Z, Z_bg)

        mask = (d_bg - d_drone) > params.nn_margin
        for idx in np.where(mask)[0]:
            x, y = pos[int(idx)]
            dets.append((
                float(d_drone[int(idx)]),
                int(x / s),
                int(y / s),
                int(pw / s),
                int(ph / s),
            ))

    return nms(dets, params.nms_iou)[: params.top_k]


def match_image(
    preds_xywh: List[Tuple[float, int, int, int, int]],
    gt_xyxy: List[Tuple[float, float, float, float]],
    iou_thr: float,
) -> Tuple[int, int, int, List[int], List[int]]:
    pred_order = sorted(range(len(preds_xywh)), key=lambda i: preds_xywh[i][0])
    gt_used = np.zeros(len(gt_xyxy), dtype=bool)
    tp = 0
    fp = 0
    pred_tp_flags = [0] * len(pred_order)
    pred_fp_flags = [0] * len(pred_order)

    for rank, p_idx in enumerate(pred_order):
        _, x, y, w, h = preds_xywh[p_idx]
        pred_xyxy = (x, y, x + w, y + h)
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_xyxy):
            if gt_used[j]:
                continue
            iou = calculate_iou(pred_xyxy, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            gt_used[best_j] = True
            pred_tp_flags[rank] = 1
        else:
            fp += 1
            pred_fp_flags[rank] = 1

    fn = int((~gt_used).sum())
    return tp, fp, fn, pred_tp_flags, pred_fp_flags


def compute_ap_from_ranked(tp_flags: List[int], fp_flags: List[int], total_gt: int) -> float:
    if total_gt <= 0:
        return 0.0
    tp = np.cumsum(tp_flags, dtype=np.float64)
    fp = np.cumsum(fp_flags, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(total_gt)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mine_hard_negatives(
    image_files: List[Path],
    params: Params,
    mean_vec: np.ndarray,
    evecs: np.ndarray,
    Z_drone: np.ndarray,
    Z_bg: np.ndarray,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    ph, pw = IMG_SIZE

    for img_path in image_files[: params.hnm_images]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gt = parse_gt_yolo(img_path.with_suffix(".txt"), w=w, h=h)
        preds = detect_single(gray, mean_vec, evecs, Z_drone, Z_bg, params)

        for _score, x, y, bw, bh in preds:
            pxyxy = (x, y, x + bw, y + bh)
            best = 0.0
            for g in gt:
                best = max(best, calculate_iou(pxyxy, g))
            if best >= 0.2:
                continue
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = gray[y1:y2, x1:x2]
            feat = extract_features(patch)
            if feat is not None:
                feats.append(feat)
            if len(feats) >= params.hnm_max_patches:
                return np.array(feats, dtype=np.float64)

    return np.array(feats, dtype=np.float64)


def evaluate_with_params(
    image_files: List[Path],
    params: Params,
    limit_images: int,
    drone_limit: int,
) -> Metrics:
    train_images = list_image_files(YOLO_DIR)

    X_drone = load_drone_patches(YOLO_DIR, limit=drone_limit, pad_ratio=params.pad_ratio)
    if len(X_drone) == 0:
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    X_bg = load_background_patches(BG_DIR, n_patches=len(X_drone), seed=SEED)

    mean_vec, evecs = build_pca(X_drone, variance_threshold=params.variance_thr, seed=SEED)
    Z_drone = ((X_drone - mean_vec) @ evecs).astype(np.float64)
    Z_bg = ((X_bg - mean_vec) @ evecs).astype(np.float64)

    if params.hard_negative_mining:
        hnm = mine_hard_negatives(train_images, params, mean_vec, evecs, Z_drone, Z_bg)
        if len(hnm) > 0:
            X_bg_aug = np.vstack([X_bg, hnm])
            Z_bg = ((X_bg_aug - mean_vec) @ evecs).astype(np.float64)

    eval_files = image_files[:limit_images]
    total_tp = 0
    total_fp = 0
    total_fn = 0

    ranked_tp: List[int] = []
    ranked_fp: List[int] = []
    total_gt = 0

    for img_path in eval_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gt = parse_gt_yolo(img_path.with_suffix(".txt"), w=w, h=h)
        preds = detect_single(gray, mean_vec, evecs, Z_drone, Z_bg, params)

        tp, fp, fn, pred_tp_flags, pred_fp_flags = match_image(preds, gt, iou_thr=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt)

        ranked_tp.extend(pred_tp_flags)
        ranked_fp.extend(pred_fp_flags)

    precision = total_tp / max(total_tp + total_fp, 1e-12)
    recall = total_tp / max(total_tp + total_fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    ap50 = compute_ap_from_ranked(ranked_tp, ranked_fp, total_gt)

    return Metrics(
        tp=int(total_tp),
        fp=int(total_fp),
        fn=int(total_fn),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        ap50=float(ap50),
    )


def run_optimization(
    output_json: Path,
    output_csv: Path,
    full_images: int,
    quick_images: int,
    max_combos: int,
    top_k: int,
    drone_limit: int,
) -> None:
    image_files = list_image_files(YOLO_DIR)

    baseline = Params(
        variance_thr=0.97,
        nn_margin=0.25,
        nms_iou=0.15,
        std_threshold=15.0,
        scales=(1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10),
        pad_ratio=0.0,
    )

    padded = Params(**{**asdict(baseline), "pad_ratio": 0.20})

    results: List[TrialResult] = []
    print(f"Starting optimization: full_images={full_images}, quick_images={quick_images}, max_combos={max_combos}, top_k={top_k}")

    for name, p in [("baseline_no_padding", baseline), ("phase1_padding_20", padded)]:
        m = evaluate_with_params(image_files, p, limit_images=full_images, drone_limit=drone_limit)
        results.append(TrialResult("phase1", name, p, m, full_images))
        print(f"[Phase1] {name}: TP={m.tp} FP={m.fp} FN={m.fn} F1={m.f1:.4f} AP50={m.ap50:.4f}")

    # Phase 2: full Cartesian on quick subset, then re-test top-k on full set.
    search_space = {
        "nn_margin": [0.25, 0.35, 0.45, 0.50],
        "nms_iou": [0.10, 0.15, 0.25, 0.40],
        "std_threshold": [15.0, 20.0, 25.0],
        "variance_thr": [0.90, 0.95, 0.97],
    }
    expanded_scales = (1.5, 1.2, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05)

    phase2_trials: List[TrialResult] = []

    combos_all = list(
        itertools.product(
            search_space["nn_margin"],
            search_space["nms_iou"],
            search_space["std_threshold"],
            search_space["variance_thr"],
        )
    )
    combos = combos_all[: max(1, min(max_combos, len(combos_all)))]
    print(f"Phase2 quick grid size: {len(combos)} / {len(combos_all)}")

    for i, (nn_margin, nms_iou, std_thr, var_thr) in enumerate(combos, start=1):
        p = Params(
            variance_thr=var_thr,
            nn_margin=nn_margin,
            nms_iou=nms_iou,
            std_threshold=std_thr,
            scales=expanded_scales,
            pad_ratio=0.20,
        )
        m = evaluate_with_params(image_files, p, limit_images=quick_images, drone_limit=drone_limit)
        tr = TrialResult("phase2_quick", f"grid_{i:03d}", p, m, quick_images)
        phase2_trials.append(tr)
        if i % 25 == 0:
            print(f"[Phase2 quick] {i}/{len(combos)} complete")

    phase2_trials_sorted = sorted(phase2_trials, key=lambda r: (r.metrics.ap50, r.metrics.f1, -r.metrics.fp), reverse=True)
    top_candidates = phase2_trials_sorted[: max(1, top_k)]

    for i, tr in enumerate(top_candidates, start=1):
        m = evaluate_with_params(image_files, tr.params, limit_images=full_images, drone_limit=drone_limit)
        results.append(TrialResult("phase2_full", f"top{i}_{tr.trial_name}", tr.params, m, full_images))
        print(f"[Phase2 full] top{i}: TP={m.tp} FP={m.fp} FN={m.fn} F1={m.f1:.4f} AP50={m.ap50:.4f}")

    # Phase 3: hard negative mining over best phase2/full configuration.
    best_so_far = max(results, key=lambda r: (r.metrics.ap50, r.metrics.f1, -r.metrics.fp))
    p_hnm = Params(**{**asdict(best_so_far.params), "hard_negative_mining": True, "hnm_images": 30, "hnm_max_patches": 800})
    m_hnm = evaluate_with_params(image_files, p_hnm, limit_images=full_images, drone_limit=drone_limit)
    results.append(TrialResult("phase3_hnm", "hard_negative_mining", p_hnm, m_hnm, full_images))
    print(f"[Phase3 HNM] TP={m_hnm.tp} FP={m_hnm.fp} FN={m_hnm.fn} F1={m_hnm.f1:.4f} AP50={m_hnm.ap50:.4f}")

    # Persist logs.
    output_rows: List[Dict[str, object]] = []
    for tr in results + phase2_trials:
        row: Dict[str, object] = {
            "phase": tr.phase,
            "trial_name": tr.trial_name,
            "images_evaluated": tr.images_evaluated,
            "tp": tr.metrics.tp,
            "fp": tr.metrics.fp,
            "fn": tr.metrics.fn,
            "precision": tr.metrics.precision,
            "recall": tr.metrics.recall,
            "f1": tr.metrics.f1,
            "ap50": tr.metrics.ap50,
        }
        row.update({f"param_{k}": v for k, v in asdict(tr.params).items()})
        output_rows.append(row)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)

    if output_rows:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)

    best_final = max(results, key=lambda r: (r.metrics.ap50, r.metrics.f1, -r.metrics.fp))
    print("\n=== BEST FINAL TRIAL ===")
    print(best_final.trial_name)
    print(json.dumps(asdict(best_final), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous PCA detector optimization")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "optimization_results.json")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "optimization_results.csv")
    parser.add_argument("--full-images", type=int, default=100)
    parser.add_argument("--quick-images", type=int, default=25)
    parser.add_argument("--max-combos", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--drone-limit", type=int, default=700)
    args = parser.parse_args()

    run_optimization(
        args.output_json,
        args.output_csv,
        args.full_images,
        args.quick_images,
        args.max_combos,
        args.top_k,
        args.drone_limit,
    )


if __name__ == "__main__":
    main()
