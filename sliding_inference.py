"""Adaptive multi-scale sliding-window ONNX inference for the container model
(yolo26m, opset 17).

Usage: python sliding_inference.py <image_path> <output_path>
       [--scales 1280,512] [--overlap 0.4] [--conf 0.25] [--model models/best.onnx]

Single-pass inference misses containers that are edge-truncated, too small in
short panoramas, or adjacent twins merged into one box (README §6 + local
tests). Instead of running every tile scale everywhere, this script works as a
coarse-to-fine cascade that adapts to the image content:

  1. run tiles at the coarsest scale (1280 = native training width)
  2. merge detections (IOS), paint them into a coverage mask
  3. next scale (512 = ~2.5x zoom) runs only on tiles still >=50% uncovered —
     this rescues tiny containers, isolates adjacent twins and catches
     edge-truncated vehicles without paying for covered regions
  4. stop early as soon as no uncovered tiles remain

Easy panoramas finish after the first scale; hard ones get progressively more
zoom exactly where single-pass failed.

Per tile (ONNX contract, AGENTS.md): preprocess (BGR->RGB, longest side ->
1280, /32 pad 114-gray, /255) -> decode output0 (1, 5, anchors) -> conf >=
0.25 -> per-tile NMS IoU 0.45 -> map boxes to panorama coords.
Merge: IOS (intersection-over-smaller) 0.5 — edge fragments nest inside the
full-contour box (IoU < 0.45 but IoS ~ 1), which plain NMS-IoU cannot
suppress; genuine twin containers never overlap (IoU = 0 in training).
FP filter: zoomed tiles can hallucinate containers on smeared background;
those regions are horizontal line-scan streaks (vertical-edge ratio ~0.10)
while real corrugated containers measure 0.25+ — boxes below 0.15 are dropped.
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

CONF = 0.25         # score threshold (training spec)
IOU = 0.45          # per-tile NMS (training spec)
IOS = 0.5           # cross-tile/scale merge threshold
SCALES = (1280, 512)
OVERLAP = 0.4       # 30-40% overlap per README §6
MAX_DET = 10
VERT_MIN = 0.15     # min vertical-edge ratio; container corrugation measures
                    # 0.25+, line-scan smear FPs ~0.10 (see verticality())
UNCOVER = 0.5       # tile runs at a scale if >= this fraction is uncovered


def preprocess(img):
    h, w = img.shape[:2]
    scale = 1280 / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    nw32 = ((nw + 31) // 32) * 32
    nh32 = ((nh + 31) // 32) * 32
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((nh32, nw32, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], scale, (w, h)


def decode(out, scale, ow, oh, conf=CONF):
    """Output (1, 5, anchors) -> Nx5 array [x1, y1, x2, y2, score] in original px."""
    pred = out[0].T  # (anchors, 5)
    cxcywh, scores = pred[:, :4], pred[:, 4]
    xyxy = np.empty_like(cxcywh)
    xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
    keep = scores >= conf
    det = np.hstack([xyxy[keep], scores[keep, None]])
    det[:, :4] /= scale  # coords only — scores must stay unscaled
    det[:, [0, 2]] = det[:, [0, 2]].clip(0, ow - 1)
    det[:, [1, 3]] = det[:, [1, 3]].clip(0, oh - 1)
    return nms(det, score_thr=conf)


def nms(det, iou=IOU, score_thr=CONF):
    if len(det) == 0:
        return det
    rects = det[:, :4].copy()
    rects[:, 2:] -= rects[:, :2]
    idx = np.array(cv2.dnn.NMSBoxes(
        rects.tolist(), det[:, 4].tolist(), score_thr, iou)).reshape(-1).astype(int)
    return det[idx] if len(idx) else det[:0]


def merge_ios(det, thr=IOS):
    """Cross-tile/scale NMS on intersection-over-smaller (a box nested inside a
    duplicate of the same object has IOS ~1 but possibly small IoU)."""
    if len(det) == 0:
        return det
    det = det[np.argsort(-det[:, 4])]
    keep = []
    while len(det):
        best, rest = det[0], det[1:]
        keep.append(best)
        if len(rest) == 0:
            break
        ix1 = np.maximum(best[0], rest[:, 0])
        iy1 = np.maximum(best[1], rest[:, 1])
        ix2 = np.minimum(best[2], rest[:, 2])
        iy2 = np.minimum(best[3], rest[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        smaller = np.minimum((best[2] - best[0]) * (best[3] - best[1]),
                             (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1]))
        det = rest[inter / np.maximum(smaller, 1e-9) < thr]
    return np.array(keep)


def sliding_windows(w, h, tile, overlap=OVERLAP):
    stride = max(1, int(tile * (1 - overlap)))
    starts = lambda size: sorted({*range(0, max(size - tile, 0) + 1, stride),
                                  max(size - tile, 0)})
    return [(x, y) for y in starts(h) for x in starts(w)]


def run_tiles(sess, img, tiles, tile, conf):
    """Decode the given (tx, ty) tile origins of size `tile`; boxes in panorama coords."""
    dets = []
    for tx, ty in tiles:
        tile_img = img[ty:ty + tile, tx:tx + tile]
        x, scale, _ = preprocess(tile_img)
        input_name = sess.get_inputs()[0].name
        d = decode(sess.run(None, {input_name: x})[0], scale,
                   tile_img.shape[1], tile_img.shape[0], conf)
        d[:, [0, 2]] += tx
        d[:, [1, 3]] += ty
        dets.append(d)
    return dets


def verticality(det, img, vert_min=VERT_MIN):
    """Drop zoom-induced FPs on smeared background: container corrugation puts
    most gradient energy on vertical edges (gx/(gx+gy) ~ 0.25-0.38 measured),
    line-scan smear is horizontal streaks (~0.10)."""
    if len(det) == 0:
        return det
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ratios = []
    for x1, y1, x2, y2, _ in det:
        roi = gray[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        gx = np.abs(cv2.Sobel(roi, cv2.CV_64F, 1, 0)).mean()
        gy = np.abs(cv2.Sobel(roi, cv2.CV_64F, 0, 1)).mean()
        ratios.append(gx / max(gx + gy, 1e-9))
    keep = np.array(ratios) >= vert_min
    for r, k in zip(ratios, keep):
        if not k:
            print(f"  dropped FP: verticality {r:.3f} < {vert_min} (smeared background)")
    return det[keep]


def detect(sess, img, scales=SCALES, overlap=OVERLAP, conf=CONF,
           vert_min=VERT_MIN, uncover=UNCOVER):
    """Adaptive coarse-to-fine cascade. Each scale runs only on tiles still
    >= `uncover` uncovered by detections from coarser scales; the loop stops
    as soon as no uncovered tile remains. Returns (det, ms, stats)."""
    t0 = time.perf_counter()
    oh, ow = img.shape[:2]
    covered = np.zeros((oh, ow), np.uint8)
    det = np.empty((0, 5))
    stats = []
    for tile in scales:
        wins = sliding_windows(ow, oh, tile, overlap)
        ii = cv2.integral(covered)
        todo = []
        for tx, ty in wins:
            x2, y2 = min(tx + tile, ow), min(ty + tile, oh)
            area = (x2 - tx) * (y2 - ty)
            cov = ii[y2, x2] - ii[ty, x2] - ii[y2, tx] + ii[ty, tx]
            if 1 - cov / area >= uncover:
                todo.append((tx, ty))
        stats.append((tile, len(wins), len(todo)))
        if not todo:
            break  # panorama fully covered — skip remaining (zoomier) scales
        det = merge_ios(np.vstack([det, *run_tiles(sess, img, todo, tile, conf)]))
        for x1, y1, x2, y2, _ in det:
            covered[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 1
    det = verticality(det, img, vert_min)
    det = det[np.argsort(-det[:, 4])][:MAX_DET] if len(det) else det
    return det, (time.perf_counter() - t0) * 1000, stats


def draw(img, det, color, label="container"):
    vis = img.copy()
    for x1, y1, x2, y2, s in det:
        p1, q1, p2, q2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (p1, q1), (p2, q2), color, 3)
        cv2.putText(vis, f"{label} {s:.2f}", (p1, max(20, q1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("output")
    ap.add_argument("--model", default="models/best.onnx")
    ap.add_argument("--scales", default=",".join(map(str, SCALES)),
                    help="tile widths, comma-separated (default: 1280,512)")
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--ios", type=float, default=IOS)
    ap.add_argument("--verticality", type=float, default=VERT_MIN,
                    help="min vertical-edge ratio, 0 disables the smear-FP filter")
    ap.add_argument("--uncover", type=float, default=UNCOVER,
                    help="tile runs at a scale if >= this fraction is uncovered "
                         "by detections from coarser scales (default: 0.5)")
    ap.add_argument("--max-det", type=int, default=MAX_DET)
    args = ap.parse_args()

    scales = tuple(int(s) for s in args.scales.split(","))

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")
    oh, ow = img.shape[:2]
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # single-pass baseline (what the plain ONNX run gives)
    x, scale, _ = preprocess(img)
    full = decode(sess.run(None, {"images": x})[0], scale, ow, oh)
    print(f"image {ow}x{oh} | single-pass: {len(full)} detection(s)")
    for x1, y1, x2, y2, s in full:
        print(f"  conf={s:.3f}  xyxy=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

    det, dt, stats = detect(sess, img, scales, args.overlap, args.conf,
                            args.verticality, args.uncover)
    ran = sum(n for _, _, n in stats)
    cascade = " -> ".join(f"{t}px {r}/{w} tiles" for t, w, r in stats)
    print(f"\nadaptive cascade scales={scales} overlap={args.overlap} "
          f"({ran} tiles run): {len(det)} detection(s), {dt:.0f} ms (CPU)")
    print(f"  per scale: {cascade}")
    for x1, y1, x2, y2, s in det:
        print(f"  conf={s:.3f}  xyxy=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

    vis = np.vstack([draw(img, full, (0, 140, 255)),
                     np.full((6, ow, 3), 255, np.uint8),
                     draw(img, det, (0, 220, 0))])
    cv2.imwrite(args.output, vis)
    print(f"\nsaved: {args.output} (top: single-pass orange | bottom: multi-scale green)")


if __name__ == "__main__":
    main()
