#!/usr/bin/env python3
"""
可视化脚本 - 生成各模型检测结果对比图。

Usage:
    python visualize.py                 # 生成所有对比图
    python visualize.py --image 1.jpg   # 只生成某张图
"""

import os
import sys
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_font(size=16):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_points(draw_image: Image.Image, points: list[dict], color: str = "red",
                radius: int = 3, label: str = "") -> Image.Image:
    img = draw_image.copy()
    draw = ImageDraw.Draw(img)

    for p in points:
        x, y = p["x"], p["y"]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline="white", width=1)

    if label:
        font = load_font(14)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([4, 4, 4 + tw + 8, 4 + th + 8], fill="white")
        draw.text((8, 6), label, fill=color, font=font)

    return img


def generate_comparison(image_id: str = None):
    image_dir = os.path.join(PROJECT_ROOT, "对照实验", "images")
    gt_dir = os.path.join(PROJECT_ROOT, "对照实验", "ground_truth", "lable")
    results_base = os.path.join(PROJECT_ROOT, "对照实验", "results")
    vis_dir = os.path.join(PROJECT_ROOT, "对照实验", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    model_dirs = {}
    for d in os.listdir(results_base):
        full = os.path.join(results_base, d)
        if os.path.isdir(full) and not d.startswith("."):
            model_dirs[d] = full

    if not model_dirs:
        print("未找到模型结果")
        return

    model_results = {}
    for name, path in model_dirs.items():
        model_results[name] = {}
        for f in os.listdir(path):
            if f.endswith("_result.json") and f != "summary.json":
                with open(os.path.join(path, f), "r", encoding="utf-8") as fp:
                    model_results[name][f.replace("_result.json", "")] = json.load(fp)

    image_files = []
    for f in os.listdir(image_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_files.append(f)

    import re as _re
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]
    image_files.sort(key=natural_key)

    if image_id:
        image_files = [f for f in image_files if Path(f).stem == image_id.replace(".jpg", "").replace(".png", "")]
        if not image_files:
            print(f"未找到图片: {image_id}")
            return

    model_names = sorted(model_results.keys())
    cols = 1 + len(model_names)  # GT + 各模型
    grid_padding = 8
    cell_w, cell_h = 400, 400

    # 生成逐图对比
    for img_file in image_files:
        basename = Path(img_file).stem
        img_path = os.path.join(image_dir, img_file)
        original = Image.open(img_path).convert("RGB")

        # 缩放
        original.thumbnail((cell_w, cell_h), Image.LANCZOS)
        w, h = original.size

        grid_w = grid_padding * (cols + 1) + cell_w * cols
        grid_h = grid_padding * 2 + cell_h
        grid = Image.new("RGB", (grid_w, grid_h), "white")

        # GT
        font = load_font(14)
        bg = Image.new("RGB", (cell_w, cell_h), (240, 240, 240))
        draw = ImageDraw.Draw(bg)

        if os.path.exists(os.path.join(gt_dir, f"{basename}.json")):
            with open(os.path.join(gt_dir, f"{basename}.json"), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            gt_points = []
            for shape in gt_data.get("shapes", []):
                if shape.get("shape_type") == "point":
                    pt = shape["points"][0]
                    gt_points.append({"x": pt[0], "y": pt[1]})
            gt_count = len(gt_points)
            bg = draw_points(original, gt_points, "green", 3, f"GT: {gt_count}")
        else:
            bg = original.copy()
            draw = ImageDraw.Draw(bg)
            draw.text((4, 4), "GT: N/A", fill="gray", font=font)

        grid.paste(bg, (grid_padding, grid_padding))

        for col_idx, model_name in enumerate(model_names):
            x_offset = grid_padding + cell_w * (col_idx + 1) + grid_padding * (col_idx + 1)
            bg = Image.new("RGB", (cell_w, cell_h), (240, 240, 240))

            if basename in model_results[model_name]:
                result = model_results[model_name][basename]
                points = result.get("points", [])
                count = result.get("count", len(points))
                bg = draw_points(original, points, "red", 3, f"{model_name}: {count}")
            else:
                bg = original.copy()
                draw = ImageDraw.Draw(bg)
                draw.text((4, 4), f"{model_name}: N/A", fill="gray", font=font)

            grid.paste(bg, (x_offset, grid_padding))

        grid_path = os.path.join(vis_dir, f"{basename}_comparison.png")
        grid.save(grid_path)
        print(f"已生成: {grid_path}")

    print(f"\n可视化完成，结果保存至: {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成对比可视化")
    parser.add_argument("--image", type=str, help="指定图片，如 1.jpg")
    args = parser.parse_args()
    generate_comparison(args.image)