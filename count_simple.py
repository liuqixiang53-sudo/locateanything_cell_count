#!/usr/bin/env python3
"""
Simple Cell Counting Script - No preprocessing, "cell" prompt.

Usage:
    # Single image
    python count_simple.py --image_path image/8.jpg

    # Directory of images
    python count_simple.py --image_dir "20260608 jishu" --output_dir ./cell_results_jishu
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Add Eagle/Embodied to path for locateanything_worker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Eagle", "Embodied"))
from locateanything_worker import LocateAnythingWorker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


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

            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area

            if iou < iou_threshold:
                filtered.append(box)

        boxes = filtered

    return keep


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
    """Parse model output into pixel-coordinate bounding boxes."""
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({
            "x1": x1 / 1000 * image_width,
            "y1": y1 / 1000 * image_height,
            "x2": x2 / 1000 * image_width,
            "y2": y2 / 1000 * image_height,
        })
    return boxes


def visualize_result(image, result, output_path):
    """Draw cell center points on the image and save it."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    boxes = result.get("boxes", [])
    count = result.get("count", 0)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        radius = 2
        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            fill="red",
            outline="white",
            width=1,
        )

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (IOError, OSError):
        font = ImageFont.load_default()

    count_text = f"Cell Count: {count}"
    bbox = draw.textbbox((0, 0), count_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 5
    draw.rectangle(
        [(margin, margin), (margin + text_width + margin * 2, margin + text_height + margin * 2)],
        fill="white",
    )
    draw.text(
        (margin * 2, margin),
        count_text,
        fill="black",
        font=font,
    )

    draw_image.save(output_path)
    print(f"Visualization saved to: {output_path}")


def process_single_image(
    worker: LocateAnythingWorker,
    image_path: str,
    output_dir: str,
    nms_iou_threshold: float = 0.3,
) -> dict:
    """Process a single image and save results."""
    print(f"\nProcessing: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    image_width, image_height = image.size

    # No preprocessing - use original image directly
    # Use enhanced prompt that describes touching/deformed cell characteristics
    prompt = (
        "Locate all the instances that match the following description: cell. "
        "These appear as small circular, oval, or semi-circular shapes. "
        "When cells are crowded or touching, they may be deformed, squeezed into "
        "ellipses or half-circles. Even if cells are touching or overlapping, "
        "each cell must be enclosed in a SEPARATE bounding box. "
        "Do NOT group multiple cells into one large box. "
        "Each bounding box should tightly enclose a single cell."
    )
    result = worker.predict(image, prompt, verbose=True)
    answer = result["answer"]

    # Parse boxes
    boxes = parse_boxes(answer, image_width, image_height)
    raw_count = len(boxes)
    print(f"Raw detection: {raw_count} boxes")

    # Apply NMS
    boxes = nms(boxes, nms_iou_threshold)
    print(f"After NMS: {len(boxes)} boxes")

    # Print raw output
    print(f"Raw output: {answer}")
    print(f"Detected cells: {len(boxes)}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    # Save original copy
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    image.save(original_path)

    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    visualize_result(image, {"boxes": boxes, "count": len(boxes)}, vis_path)

    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}_result.json")
    result_data = {
        "image_path": image_path,
        "count": len(boxes),
        "raw_count": raw_count,
        "boxes": boxes,
        "answer": answer,
        "settings": {
            "preprocess": "none",
            "cell_type": "cell",
            "nms_iou_threshold": nms_iou_threshold,
        },
    }
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    print(f"\n{'='*50}")
    print(f"FINAL RESULT: {len(boxes)} cells")
    print(f"{'='*50}")

    # Clean up GPU memory to prevent OOM on subsequent images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"count": len(boxes), "raw_count": raw_count, "boxes": boxes, "answer": answer}


def process_directory(
    worker: LocateAnythingWorker,
    dir_path: str,
    output_dir: str,
    nms_iou_threshold: float = 0.3,
) -> dict:
    """Process all images in a directory."""
    image_files = [
        f for f in os.listdir(dir_path)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]
    # Sort by natural order (numeric-aware)
    import re as _re
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r'(\d+)', s)]
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
            worker, img_path, output_dir=output_dir, nms_iou_threshold=nms_iou_threshold
        )
        results[img_file] = result["count"]

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    total = 0
    for img_file, count in results.items():
        print(f"  {img_file}: {count} cells")
        total += count
    print("-" * 50)
    print(f"  Total: {total} cells across {len(results)} images")
    print("=" * 50)

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple Cell Counting (No preprocessing, 'cell' prompt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python count_simple.py --image_path image/8.jpg

  # Directory of images
  python count_simple.py --image_dir "20260608 jishu" --output_dir ./cell_results_jishu
        """,
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to a single image to process",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to a directory of images to process",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./LocateAnything-3B",
        help="Path to the LocateAnything model (default: ./LocateAnything-3B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cell_results_simple",
        help="Directory to save results (default: ./cell_results_simple)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for NMS (default: 0.3)",
    )

    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        parser.error("Either --image_path or --image_dir must be specified")

    if args.image_path and args.image_dir:
        parser.error("Cannot specify both --image_path and --image_dir")

    # Load model
    print(f"Loading model from: {args.model_path}")
    print(f"Using device: {args.device}")

    worker = LocateAnythingWorker(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully.")

    if args.image_path:
        process_single_image(
            worker,
            args.image_path,
            output_dir=args.output_dir,
            nms_iou_threshold=args.nms_iou_threshold,
        )
    else:
        process_directory(
            worker,
            args.image_dir,
            output_dir=args.output_dir,
            nms_iou_threshold=args.nms_iou_threshold,
        )


if __name__ == "__main__":
    main()