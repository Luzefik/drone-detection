import os
import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE         = (64, 64)
FLAT_SIZE        = IMG_SIZE[0] * IMG_SIZE[1]
YOLO_DIR         = './archive/drone_dataset_yolo/dataset_txt'
OUTPUT_DIR       = 'data/res_1'
PARAMS_DIR       = 'model_params'
NUM_TEST_IMAGES  = 100

# Sliding-window settings
SCALES           = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10)
STRIDE           = 16
STD_THRESHOLD    = 15.0   # skip flat sky/uniform patches  (Idea 1)
NN_MARGIN        = 0.25   # (d_none − d_drone) must exceed this  
NMS_IOU          = 0.5
TOP_K            = 8

# =============================================================================
# Feature Extraction  (raw-pixel — must match generate_params.py exactly)
# =============================================================================
def extract_features(image, target_size=IMG_SIZE):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return resized.flatten() / 255.0

# =============================================================================
# Vectorised minimum distance  (pure NumPy, no scipy)
# =============================================================================
def _min_dist(A, B):
    """
    A : (M, K)  —  query projections
    B : (N, K)  —  reference projections
    Returns (M,) minimum L2 distance from each row of A to any row of B.
    Uses chunked computation to avoid float32 overflow on large matrices.
    """
    A  = np.asarray(A,  dtype=np.float64)
    B  = np.asarray(B,  dtype=np.float64)
    A_sq = np.einsum('ij,ij->i', A, A)[:, None]   # (M, 1)
    B_sq = np.einsum('ij,ij->i', B, B)[None, :]   # (1, N)
    # Chunk dot-product to keep memory under control
    chunk = 512
    min_d = np.full(len(A), np.inf)
    for start in range(0, len(B), chunk):
        AB = A @ B[start:start+chunk].T            # (M, chunk)
        d2 = np.maximum(A_sq + B_sq[:, start:start+chunk] - 2.0 * AB, 0.0)
        min_d = np.minimum(min_d, d2.min(axis=1))
    return np.sqrt(min_d)

# =============================================================================
# Geometry helpers
# =============================================================================
def _iou(b1, b2):
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = ax1 + b1[2], ay1 + b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = bx1 + b2[2], by1 + b2[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def nms(detections, iou_thresh=NMS_IOU):
    """Greedy NMS. Score = d_drone (lower is better match)."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[0])
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if _iou(best[1:], d[1:]) < iou_thresh]
    return kept

# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model parameters...")
    try:
        mean_vec = np.load(os.path.join(PARAMS_DIR, 'mean_vector.npy')).astype(np.float64)
        evecs    = np.load(os.path.join(PARAMS_DIR, 'eigenvectors.npy')).astype(np.float64)
        Z_drone  = np.load(os.path.join(PARAMS_DIR, 'Z_train_drone.npy')).astype(np.float64)
        Z_bg     = np.load(os.path.join(PARAMS_DIR, 'Z_train_non_drone.npy')).astype(np.float64)
    except FileNotFoundError as e:
        print(f"Error: {e}. Run generate_params.py first.")
        return

    all_files  = sorted([f for f in os.listdir(YOLO_DIR) if f.endswith(('.jpg', '.png'))])
    test_files = all_files[:NUM_TEST_IMAGES]
    print(f"Running on {len(test_files)} images...")

    ph, pw = IMG_SIZE

    for i, filename in enumerate(test_files):
        img = cv2.imread(os.path.join(YOLO_DIR, filename))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = []

        for s in SCALES:
            sh = int(gray.shape[0] * s)
            sw = int(gray.shape[1] * s)
            if sh < ph or sw < pw:
                continue
            simg = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

            feats, pos = [], []
            for y in range(0, sh - ph, STRIDE):
                for x in range(0, sw - pw, STRIDE):
                    raw = simg[y:y+ph, x:x+pw]
                    # Idea 1: skip flat patches (uniform sky / plain wall)
                    if np.std(raw) < STD_THRESHOLD:
                        continue
                    feat = extract_features(raw)
                    if feat is not None:
                        feats.append(feat)
                        pos.append((x, y))

            if not feats:
                continue

            # Project batch to PCA space
            P = np.array(feats, dtype=np.float64)
            Z = (P - mean_vec) @ evecs

            # Two-class nearest-neighbour in PCA space
            d_drone = _min_dist(Z, Z_drone)
            d_bg    = _min_dist(Z, Z_bg)

            # A patch is a drone when it is closer to the drone set
            # than to the background set by at least NN_MARGIN
            mask = (d_bg - d_drone) > NN_MARGIN
            for idx in np.where(mask)[0]:
                x, y = pos[idx]
                ox, oy = int(x / s), int(y / s)
                ow, oh = int(pw / s), int(ph / s)
                detections.append((d_drone[idx], ox, oy, ow, oh))

        kept = nms(detections)[:TOP_K]
        base = os.path.splitext(filename)[0]

        # Draw and save annotated image
        img_out = img.copy()
        for score, x, y, w, h in kept:
            cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_out, f"drone {score:.2f}", (x, max(0, y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}.jpg"), img_out)

        # Save bounding-box results  — score x y w h
        with open(os.path.join(OUTPUT_DIR, f"{base}.txt"), 'w') as f:
            for score, x, y, w, h in kept:
                f.write(f"{score:.4f} {x} {y} {w} {h}\n")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_files)}")

    print(f"Done — results in {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
