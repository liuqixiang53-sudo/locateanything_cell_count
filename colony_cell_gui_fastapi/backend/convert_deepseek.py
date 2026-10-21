#!/usr/bin/env python3
"""
将 DeepSeek 网页版聊天原文（含 ```json 代码块）转换为标准 {id}_result.json。

映射约定: 上传图片顺序 = 图片文件名自然排序，image_N -> 第 N 张。

Usage:
    python convert_deepseek.py
"""

import os
import sys
import re
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_DIR = os.path.join(PROJECT_ROOT, "对照实验", "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "对照实验", "results", "deepseek")
RAW_FILE = os.path.join(OUT_DIR, "deepseek_json_20261014_e958d2.json")


def natural_image_ids(image_dir: str) -> list[str]:
    files = [f for f in os.listdir(image_dir)
             if Path_suffix(f.lower()) in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}]

    def key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    files.sort(key=key)
    return [os.path.splitext(f)[0] for f in files]


def Path_suffix(name: str) -> str:
    i = name.rfind(".")
    return name[i:] if i >= 0 else ""


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("未找到 JSON 内容")
    return json.loads(m.group())


def main():
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    data = extract_json(raw_text)
    ids_sorted = sorted(data.keys(), key=lambda k: int(k.rsplit("_", 1)[1]))

    image_ids = natural_image_ids(IMAGE_DIR)
    if len(ids_sorted) != len(image_ids):
        print(f"错误: DeepSeek 返回 {len(ids_sorted)} 条, 测试集 {len(image_ids)} 张")
        return 1

    summary = {}
    for image_id, key in zip(image_ids, ids_sorted):
        entry = data[key]
        boxes = entry.get("boxes", [])
        from PIL import Image
        w, h = Image.open(os.path.join(IMAGE_DIR, f"{image_id}.jpg")).size
        points = [{"x": (b[0] + b[2]) / 2 / 1000 * w,
                   "y": (b[1] + b[3]) / 2 / 1000 * h} for b in boxes]
        count = entry.get("count", len(boxes))
        out = {
            "image": f"{image_id}.jpg",
            "model": "DeepSeek",
            "count": count,
            "raw_count": len(boxes),
            "points": points,
            "raw_response": f"[converted] {key} -> {image_id}.jpg | count={count}, boxes={len(boxes)}",
            "elapsed": None,
            "has_coordinates": len(points) > 0,
        }
        out_file = os.path.join(OUT_DIR, f"{image_id}_result.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        summary[image_id] = count

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"已转换 {len(summary)} 张: {summary}")


if __name__ == "__main__":
    sys.exit(main() or 0)
