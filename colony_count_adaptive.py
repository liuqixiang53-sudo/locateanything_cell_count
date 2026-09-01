#!/usr/bin/env python3
"""
Adaptive Colony Counting - Independent two-stage: coarse locate then density-stratified.

Stage 1: coarse locate on original image → raw count N
Stage 2:
  - sparse   (N < sparse_threshold):   use stage-1 result directly (centers only)
  - medium   (sparse_threshold ≤ N < dense_threshold): CLAHE + re-locate + center-distance dedup
  - dense    (N ≥ dense_threshold):     CLAHE + re-locate + watershed split + dedup
  - full_image_box fallback:            CLAHE + re-locate; if still full-image → whole-image watershed

Fully self-contained: does NOT import from colony_count_enhance_la.

Usage:
    # Single image
    python colony_count_adaptive.py --image_path image/1.png

    # Directory of images
    python colony_count_adaptive.py --image_dir image --output_dir ./colony_results_adaptive

    # Adjust density thresholds
    python colony_count_adaptive.py --image_path test.png --sparse_threshold 40 --dense_threshold 150
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add Eagle/Embodied to path for locateanything_worker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Eagle", "Embodied"))
from locateanything_worker import LocateAnythingWorker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ============================================================
# Preprocessing (CLAHE)
# ============================================================

def preprocess_image(
    image: Image.Image,
    output_path: str = None,
    clahe_clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Image.Image:
    """CLAHE contrast enhancement in LAB color space (preserves color)."""
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        rgb = img_np.copy()
    else:
        rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(result_rgb)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result.save(output_path)
        print(f"Preprocessed image saved to: {output_path}")

    return result


# ============================================================
# Post-processing helpers
# ============================================================

def nms(boxes: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Non-Maximum Suppression to remove overlapping boxes."""
    if not boxes:
        return []

    for box in boxes:
        box["area"] = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])

    boxes = sorted(boxes, key=lambda b: b["area"], reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        filtered = []
        for box in boxes:
            x1 = max(current["x1"], box["x1"])
            y1 = max(current["y1"], box["y1"])
            x2 = min(current["x2"], box["x2"])
            y2 = min(current["y2"], box["y2"])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            union_area = current["area"] + box["area"] - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou < iou_threshold:
                filtered.append(box)
        boxes = filtered
    return keep


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
    """Parse model output <box><x1><y1><x2><y2></box> into pixel-coordinate boxes."""
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        px1, py1 = x1 / 1000 * image_width, y1 / 1000 * image_height
        px2, py2 = x2 / 1000 * image_width, y2 / 1000 * image_height
        boxes.append({
            "x1": min(px1, px2),
            "y1": min(py1, py2),
            "x2": max(px1, px2),
            "y2": max(py1, py2),
        })
    return boxes


def dedup_by_center_distance(boxes: list[dict], distance_threshold: float = 3.0) -> list[dict]:
    """
    Deduplicate boxes by center-point distance.

    For dense/tiny colonies, IoU-based NMS is too aggressive (overlapping colonies
    have high IoU even though they are distinct). Instead we keep a box only if its
    center is farther than `distance_threshold` from every already-kept center.
    Uses centroid (center_x/center_y) if available, else bbox center.
    """
    if not boxes:
        return []

    for b in boxes:
        if "center_x" in b and "center_y" in b:
            b["cx"] = b["center_x"]
            b["cy"] = b["center_y"]
        else:
            b["cx"] = (b["x1"] + b["x2"]) / 2
            b["cy"] = (b["y1"] + b["y2"]) / 2

    boxes = sorted(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)

    keep = []
    for b in boxes:
        duplicate = False
        for k in keep:
            d = ((b["cx"] - k["cx"]) ** 2 + (b["cy"] - k["cy"]) ** 2) ** 0.5
            if d < distance_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(b)

    for b in keep:
        b.pop("cx", None)
        b.pop("cy", None)
    return keep


def _compute_region_centroid(image: Image.Image, box: dict) -> tuple[float, float]:
    """Compute the intensity-weighted centroid of a box region for accurate center."""
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_width, img_height = image.size
    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = min(img_width, int(box["x2"]))
    y2 = min(img_height, int(box["y2"]))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2

    region = img_np[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m = cv2.moments(binary)
        if m["m00"] > 0:
            cx_rel = m["m10"] / m["m00"]
            cy_rel = m["m01"] / m["m00"]
            return float(x1 + cx_rel), float(y1 + cy_rel)
    except Exception:
        pass

    return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2


def watershed_split_large_boxes(
    image: Image.Image,
    boxes: list[dict],
    area_factor: float = 2.0,
    min_split_area_ratio: float = 0.0005,
    dist_threshold_ratio: float = 0.3,
) -> list[dict]:
    """
    Split large/clustered boxes using watershed segmentation.

    For boxes whose area > area_factor * median, apply watershed to separate
    touching colonies. Each sub-box also gets centroid (center_x/center_y)
    computed via cv2.moments() for accurate center placement.
    """
    if not boxes:
        return boxes

    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_width, img_height = image.size
    img_area = img_width * img_height

    for box in boxes:
        box["area"] = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])

    areas = [b["area"] for b in boxes]
    median_area = sorted(areas)[len(areas) // 2] if areas else 0

    result_boxes = []
    for box in boxes:
        is_large = box["area"] > area_factor * median_area and box["area"] > median_area * 1.5

        if not is_large:
            cx, cy = _compute_region_centroid(image, box)
            box["center_x"] = cx
            box["center_y"] = cy
            result_boxes.append(box)
            continue

        pad = 5
        x1 = max(0, int(box["x1"]) - pad)
        y1 = max(0, int(box["y1"]) - pad)
        x2 = min(img_width, int(box["x2"]) + pad)
        y2 = min(img_height, int(box["y2"]) + pad)

        if x2 - x1 < 10 or y2 - y1 < 10:
            cx, cy = _compute_region_centroid(image, box)
            box["center_x"] = cx
            box["center_y"] = cy
            result_boxes.append(box)
            continue

        region = img_np[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_max = dist_transform.max() if dist_transform.max() > 0 else 1
        _, sure_fg = cv2.threshold(dist_transform, dist_threshold_ratio * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg)

        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        region_bgr = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(region_bgr, markers)

        min_sub_area = img_area * min_split_area_ratio
        sub_boxes = []
        for label in range(2, num_labels + 1):
            mask = (markers == label).astype(np.uint8)
            if mask.sum() == 0:
                continue
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            sx1, sx2 = xs.min(), xs.max()
            sy1, sy2 = ys.min(), ys.max()
            sub_area = (sx2 - sx1) * (sy2 - sy1)
            if sub_area < min_sub_area:
                continue

            m = cv2.moments(mask)
            if m["m00"] > 0:
                cx_rel = m["m10"] / m["m00"]
                cy_rel = m["m01"] / m["m00"]
            else:
                cx_rel = (sx1 + sx2) / 2
                cy_rel = (sy1 + sy2) / 2

            sub_boxes.append({
                "x1": float(x1 + sx1),
                "y1": float(y1 + sy1),
                "x2": float(x1 + sx2),
                "y2": float(y1 + sy2),
                "area": float(sub_area),
                "center_x": float(x1 + cx_rel),
                "center_y": float(y1 + cy_rel),
            })

        if sub_boxes:
            print(f"    Watershed: split 1 large box (area={box['area']:.0f}) into {len(sub_boxes)} sub-boxes")
            result_boxes.extend(sub_boxes)
        else:
            cx, cy = _compute_region_centroid(image, box)
            box["center_x"] = cx
            box["center_y"] = cy
            result_boxes.append(box)

    return result_boxes


def full_image_watershed(
    image: Image.Image,
    clahe_clip_limit: float = 2.0,
    dist_threshold_ratio: float = 0.3,
    min_area_ratio: float = 0.0005,
) -> list[dict]:
    """
    Whole-image watershed fallback when model returns a single full-image box.

    Uses traditional CV: CLAHE + Otsu + distance transform + watershed.
    Returns boxes with centroid (center_x/center_y).
    """
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_width, img_height = image.size
    img_area = img_width * img_height

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_max = dist_transform.max() if dist_transform.max() > 0 else 1
    _, sure_fg = cv2.threshold(dist_transform, dist_threshold_ratio * dist_max, 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)

    min_area = img_area * min_area_ratio
    boxes = []
    for label in range(2, num_labels + 1):
        mask = (markers == label).astype(np.uint8)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        sx1, sx2 = xs.min(), xs.max()
        sy1, sy2 = ys.min(), ys.max()
        sub_area = (sx2 - sx1) * (sy2 - sy1)
        if sub_area < min_area:
            continue

        m = cv2.moments(mask)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = (sx1 + sx2) / 2
            cy = (sy1 + sy2) / 2

        boxes.append({
            "x1": float(sx1),
            "y1": float(sy1),
            "x2": float(sx2),
            "y2": float(sy2),
            "area": float(sub_area),
            "center_x": float(cx),
            "center_y": float(cy),
        })

    print(f"[Full-image watershed] Detected {len(boxes)} regions")
    return boxes


# ============================================================
# Enhanced prompt
# ============================================================

ENHANCED_PROMPT = (
    "Locate all the instances that match the following description: colony. "
    "These appear as circular, oval, or semi-circular spots of varying sizes on the agar surface. "
    "Colonies may be white, yellow, or other colors and have varying contrast with the background. "
    "When colonies are crowded or touching, they may be deformed, squeezed into "
    "ellipses or half-circles. Even if colonies are touching or overlapping, "
    "each colony must be enclosed in a SEPARATE bounding box. "
    "Do NOT group multiple colonies into one large box. "
    "Each bounding box should tightly enclose a single colony without extra background. "
    "Count ALL visible colonies including those near or touching the edge of the petri dish - "
    "partial colonies at the boundary should also be counted."
)


# ============================================================
# Adaptive Colony Counter
# ============================================================

class AdaptiveColonyCounter:
    """Two-stage adaptive colony counter using LocateAnythingWorker."""

    def __init__(self, model_path: str = None, device: str = "cuda", dtype=torch.bfloat16, worker=None):
        self.device = device
        self.dtype = dtype

        if worker is not None:
            self.worker = worker
            print("Using shared LocateAnythingWorker instance.")
        else:
            if model_path is None:
                raise ValueError("model_path is required when worker is not provided.")
            print(f"Loading model from: {model_path}")
            print(f"Using device: {device}")
            self.worker = LocateAnythingWorker(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )
            print("Model loaded successfully.")

    @staticmethod
    def _boxes_to_centers(boxes: list[dict]) -> list[dict]:
        """Convert boxes to center points. Prefer centroid fields over bbox center."""
        centers = []
        for box in boxes:
            if "center_x" in box and "center_y" in box:
                centers.append({"x": box["center_x"], "y": box["center_y"]})
            else:
                centers.append({
                    "x": (box["x1"] + box["x2"]) / 2,
                    "y": (box["y1"] + box["y2"]) / 2,
                })
        return centers

    def count_colonies_adaptive(
        self,
        image: Image.Image,
        sparse_threshold: int = 50,
        dense_threshold: int = 150,
        nms_iou_threshold: float = 0.3,
        dedup_distance: float = 3.0,
        clahe_clip_limit: float = 2.0,
        tile_grid_size: int = 8,
        watershed_area_factor: float = 2.0,
        watershed_dist_ratio: float = 0.3,
        watershed_min_area_ratio: float = 0.0005,
        verbose: bool = True,
    ) -> dict:
        """
        Two-stage adaptive colony counting.

        Returns dict with keys: count, raw_count, stage, boxes, centers, answer, coarse_answer.
        """
        image_width, image_height = image.size

        # ============================================================
        # Stage 1: Coarse locate on ORIGINAL image
        # ============================================================
        print("\n[Stage 1] Coarse locate on original image...")
        result = self.worker.predict(image, ENHANCED_PROMPT, verbose=verbose)
        coarse_answer = result["answer"]
        boxes = parse_boxes(coarse_answer, image_width, image_height)
        boxes = nms(boxes, nms_iou_threshold)
        n_raw = len(boxes)
        print(f"[Stage 1] Raw count (after NMS): {n_raw}")

        # Detect full-image-box degradation
        full_image_box = False
        if n_raw == 1:
            b = boxes[0]
            box_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            img_area = image_width * image_height
            if box_area > 0.8 * img_area:
                full_image_box = True
                print(
                    f"[Stage 1] WARNING: single full-image box detected "
                    f"(coverage={box_area/img_area:.1%}) → full-image watershed fallback"
                )

        # ============================================================
        # Stage 2: Density-stratified strategy
        # ============================================================
        if full_image_box:
            # ---- Full-image box: CLAHE + re-locate ----
            stage = "full_image_box_relocate"
            print(
                f"[Stage 2] Full-image box detected "
                f"→ CLAHE + re-locate"
            )
            processed = preprocess_image(
                image,
                clahe_clip_limit=clahe_clip_limit,
                tile_grid_size=tile_grid_size,
            )
            if processed.mode != "RGB":
                processed = processed.convert("RGB")

            result2 = self.worker.predict(processed, ENHANCED_PROMPT, verbose=verbose)
            final_answer = result2["answer"]
            boxes2 = parse_boxes(final_answer, image_width, image_height)
            boxes2 = nms(boxes2, nms_iou_threshold)

            # If still a single full-image box, fall back to whole-image watershed
            still_full = False
            if len(boxes2) == 1:
                b = boxes2[0]
                box_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                if box_area > 0.8 * image_width * image_height:
                    still_full = True

            if still_full:
                stage = "full_image_watershed"
                print(
                    f"[Stage 2] Re-locate still full-image box "
                    f"→ full-image watershed fallback"
                )
                final_boxes = full_image_watershed(
                    image,
                    clahe_clip_limit=clahe_clip_limit,
                    dist_threshold_ratio=watershed_dist_ratio,
                    min_area_ratio=watershed_min_area_ratio,
                )
            else:
                for box in boxes2:
                    cx, cy = _compute_region_centroid(image, box)
                    box["center_x"] = cx
                    box["center_y"] = cy
                final_boxes = dedup_by_center_distance(boxes2, dedup_distance)
                print(f"[Stage 2] After CLAHE + re-locate + dedup: {len(final_boxes)}")

        elif n_raw < sparse_threshold:
            # ---- Sparse: reuse stage-1 result ----
            stage = "sparse"
            print(
                f"[Stage 2] Density=sparse ({n_raw} < {sparse_threshold}) "
                f"→ reuse stage-1 result + centroid refinement"
            )
            final_boxes = []
            for box in boxes:
                cx, cy = _compute_region_centroid(image, box)
                box["center_x"] = cx
                box["center_y"] = cy
                final_boxes.append(box)
            final_answer = coarse_answer

        elif n_raw < dense_threshold:
            # ---- Medium: CLAHE + re-locate + center-distance dedup ----
            stage = "medium"
            print(
                f"[Stage 2] Density=medium ({sparse_threshold} ≤ {n_raw} < {dense_threshold}) "
                f"→ CLAHE + re-locate + center-distance dedup"
            )
            processed = preprocess_image(
                image,
                clahe_clip_limit=clahe_clip_limit,
                tile_grid_size=tile_grid_size,
            )
            if processed.mode != "RGB":
                processed = processed.convert("RGB")

            result2 = self.worker.predict(processed, ENHANCED_PROMPT, verbose=verbose)
            final_answer = result2["answer"]
            boxes2 = parse_boxes(final_answer, image_width, image_height)

            for box in boxes2:
                cx, cy = _compute_region_centroid(image, box)
                box["center_x"] = cx
                box["center_y"] = cy

            final_boxes = dedup_by_center_distance(boxes2, dedup_distance)
            print(f"[Stage 2] After CLAHE + re-locate + dedup: {len(final_boxes)}")

        else:
            # ---- Dense: CLAHE + re-locate + watershed split + dedup ----
            stage = "dense"
            print(
                f"[Stage 2] Density=dense ({n_raw} ≥ {dense_threshold}) "
                f"→ CLAHE + re-locate + watershed split + dedup"
            )
            processed = preprocess_image(
                image,
                clahe_clip_limit=clahe_clip_limit,
                tile_grid_size=tile_grid_size,
            )
            if processed.mode != "RGB":
                processed = processed.convert("RGB")

            result2 = self.worker.predict(processed, ENHANCED_PROMPT, verbose=verbose)
            final_answer = result2["answer"]
            boxes2 = parse_boxes(final_answer, image_width, image_height)

            before_split = len(boxes2)
            boxes2 = watershed_split_large_boxes(
                image,
                boxes2,
                area_factor=watershed_area_factor,
                min_split_area_ratio=watershed_min_area_ratio,
                dist_threshold_ratio=watershed_dist_ratio,
            )
            print(
                f"[Stage 2] Watershed: {before_split} → {len(boxes2)} boxes "
                f"(split large/clustered regions)"
            )

            final_boxes = dedup_by_center_distance(boxes2, dedup_distance)
            print(f"[Stage 2] After CLAHE + re-locate + watershed + dedup: {len(final_boxes)}")

        centers = self._boxes_to_centers(final_boxes)

        return {
            "count": len(final_boxes),
            "raw_count": n_raw,
            "stage": stage,
            "boxes": final_boxes,
            "centers": centers,
            "answer": final_answer,
            "coarse_answer": coarse_answer,
        }


# ============================================================
# Visualization
# ============================================================

def visualize_centers(image: Image.Image, result: dict, output_path: str):
    """Draw ONLY colony center points (centroids) and count text."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    centers = result.get("centers", [])
    count = result.get("count", 0)

    for c in centers:
        cx, cy = c["x"], c["y"]
        radius = 2
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill="red",
            outline="white",
            width=1,
        )

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    count_text = f"Colony Count: {count}"
    bbox = draw.textbbox((0, 0), count_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 5
    draw.rectangle(
        [(margin, margin), (margin + text_width + margin * 2, margin + text_height + margin * 2)],
        fill="white",
    )
    draw.text((margin * 2, margin), count_text, fill="black", font=font)

    draw_image.save(output_path)
    print(f"Visualization saved to: {output_path}")


# ============================================================
# Processing functions
# ============================================================

def process_single_image(
    counter: AdaptiveColonyCounter,
    image_path: str,
    output_dir: str,
    **kwargs,
) -> dict:
    """Process a single image and save results."""
    print(f"\nProcessing: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    result = counter.count_colonies_adaptive(image, **kwargs)

    print(f"\n{'='*50}")
    print(
        f"FINAL RESULT: {result['count']} colonies "
        f"(stage={result['stage']}, raw={result['raw_count']})"
    )
    print(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    # Save original copy
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    image.save(original_path)

    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    visualize_centers(image, result, vis_path)

    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}_result.json")
    result_data = {
        "image_path": image_path,
        "count": result["count"],
        "raw_count": result["raw_count"],
        "stage": result["stage"],
        "boxes": result["boxes"],
        "centers": result["centers"],
        "answer": result["answer"],
    }
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def process_directory(
    counter: AdaptiveColonyCounter,
    dir_path: str,
    output_dir: str,
    **kwargs,
) -> dict:
    """Process all images in a directory."""
    image_files = [
        f for f in os.listdir(dir_path)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]

    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    image_files.sort(key=natural_key)

    if not image_files:
        print(f"No images found in: {dir_path}")
        return {}

    print(f"Found {len(image_files)} images in: {dir_path}")

    results = {}
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(dir_path, img_file)
        print(f"\n[{i}/{len(image_files)}] ", end="")
        result = process_single_image(
            counter, img_path, output_dir=output_dir, **kwargs
        )
        results[img_file] = {
            "count": result["count"],
            "raw_count": result["raw_count"],
            "stage": result["stage"],
        }

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    total = 0
    for img_file, info in results.items():
        print(
            f"  {img_file}: {info['count']} colonies "
            f"(stage={info['stage']}, raw={info['raw_count']})"
        )
        total += info["count"]
    print("-" * 50)
    print(f"  Total: {total} colonies across {len(results)} images")
    print("=" * 50)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Colony Counting (Two-stage: coarse locate → density-stratified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python colony_count_adaptive.py --image_path image/1.png

  # Directory of images
  python colony_count_adaptive.py --image_dir image --output_dir ./colony_results_adaptive

  # Adjust density thresholds
  python colony_count_adaptive.py --image_path test.png --sparse_threshold 40 --dense_threshold 150
        """,
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to a single image to process",
    )
    parser.add_argument(
        "--image_dir", type=str, help="Path to a directory of images to process",
    )
    parser.add_argument(
        "--model_path", type=str, default="./LocateAnything-3B",
        help="Path to the LocateAnything model (default: ./LocateAnything-3B)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./colony_results_adaptive",
        help="Directory to save results (default: ./colony_results_adaptive)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)",
    )

    # Density thresholds
    parser.add_argument(
        "--sparse_threshold", type=int, default=50,
        help="Below this raw count → sparse strategy (default: 50)",
    )
    parser.add_argument(
        "--dense_threshold", type=int, default=150,
        help="At/above this raw count → dense strategy (default: 150)",
    )

    # NMS / dedup
    parser.add_argument(
        "--nms_iou_threshold", type=float, default=0.3,
        help="IoU threshold for NMS in stage-1 (default: 0.3)",
    )
    parser.add_argument(
        "--dedup_distance", type=float, default=3.0,
        help="Center-distance threshold (px) for dedup (default: 3.0)",
    )

    # CLAHE
    parser.add_argument(
        "--clahe_clip_limit", type=float, default=2.0,
        help="CLAHE clip limit for medium stage (default: 2.0)",
    )
    parser.add_argument(
        "--tile_grid_size", type=int, default=8,
        help="CLAHE tile grid size (default: 8)",
    )

    # Watershed
    parser.add_argument(
        "--watershed_area_factor", type=float, default=2.0,
        help="Box is 'large' if area > factor * median (default: 2.0)",
    )
    parser.add_argument(
        "--watershed_dist_ratio", type=float, default=0.3,
        help="Distance transform threshold ratio (default: 0.3)",
    )
    parser.add_argument(
        "--watershed_min_area_ratio", type=float, default=0.0005,
        help="Min sub-region area ratio to keep (default: 0.0005)",
    )

    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        parser.error("Either --image_path or --image_dir must be specified")
    if args.image_path and args.image_dir:
        parser.error("Cannot specify both --image_path and --image_dir")

    print("\n=== Adaptive Colony Counter (Independent) ===")
    print(f"Model:  {args.model_path}")
    print(f"Device: {args.device}")
    print(
        f"Density thresholds: sparse<{args.sparse_threshold}, "
        f"medium<{args.dense_threshold}, dense≥{args.dense_threshold}"
    )
    print(
        f"NMS IoU: {args.nms_iou_threshold}, dedup dist: {args.dedup_distance}\n"
        f"CLAHE clip: {args.clahe_clip_limit}, tile: {args.tile_grid_size}\n"
        f"Watershed: area_factor={args.watershed_area_factor}, "
        f"dist_ratio={args.watershed_dist_ratio}, "
        f"min_area_ratio={args.watershed_min_area_ratio}"
    )
    print()

    counter = AdaptiveColonyCounter(
        model_path=args.model_path,
        device=args.device,
    )

    kwargs = {
        "sparse_threshold": args.sparse_threshold,
        "dense_threshold": args.dense_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "dedup_distance": args.dedup_distance,
        "clahe_clip_limit": args.clahe_clip_limit,
        "tile_grid_size": args.tile_grid_size,
        "watershed_area_factor": args.watershed_area_factor,
        "watershed_dist_ratio": args.watershed_dist_ratio,
        "watershed_min_area_ratio": args.watershed_min_area_ratio,
    }

    if args.image_path:
        process_single_image(
            counter, args.image_path, output_dir=args.output_dir, **kwargs,
        )
    else:
        process_directory(
            counter, args.image_dir, output_dir=args.output_dir, **kwargs,
        )


if __name__ == "__main__":
    main()