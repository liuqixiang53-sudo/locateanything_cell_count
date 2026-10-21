#!/usr/bin/env python3
"""
半自动录制工具 - 用于豆包 / DeepSeek 网页版手动操作。

功能:
1. 逐张展示图片 + 统一 prompt
2. 你在网页版上传图片、粘贴 prompt、获取回复
3. 将模型回复粘贴回工具，工具自动解析并保存结果

Usage:
    python semi_auto_recorder.py --model 豆包
    python semi_auto_recorder.py --model DeepSeek
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def natural_sort_key(s):
    import re as _re
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
    """解析模型输出中的坐标"""

    # 格式1: <box><x1><y1><x2><y2></box>
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({
            "x1": x1 / 1000 * image_width,
            "y1": y1 / 1000 * image_height,
            "x2": x2 / 1000 * image_width,
            "y2": y2 / 1000 * image_height,
        })
    if boxes:
        return boxes

    # 格式2: [x1, y1, x2, y2] 或 (x1, y1, x2, y2)
    for m in re.finditer(r"[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    if boxes:
        return boxes

    # 格式3: QwenVL 风格
    for m in re.finditer(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({
            "x1": x1 / 1000 * image_width,
            "y1": y1 / 1000 * image_height,
            "x2": x2 / 1000 * image_width,
            "y2": y2 / 1000 * image_height,
        })
    if boxes:
        return boxes

    # 格式4: 纯文本坐标 "(x1, y1)-(x2, y2)"
    for m in re.finditer(r"\((\d+),\s*(\d+)\)-\((\d+),\s*(\d+)\)", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return boxes


def parse_simple_count(answer: str) -> int:
    """尝试从纯文本回答中提取数字（如 '共 113 个细胞'）"""
    patterns = [
        r"共\s*(\d+)\s*个",
        r"总共\s*(\d+)\s*个",
        r"有\s*(\d+)\s*个",
        r"count[:\s]*(\d+)",
        r"(\d+)\s*cells?",
        r"(\d+)\s*个细胞",
    ]
    for pat in patterns:
        m = re.search(pat, answer, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def run_recorder(model_name: str):
    image_dir = os.path.join(PROJECT_ROOT, "对照实验", "images")
    prompt_path = os.path.join(PROJECT_ROOT, "对照实验", "prompts", "cell_prompt.txt")
    output_dir = os.path.join(PROJECT_ROOT, "对照实验", "results",
                              "doubao" if "豆包" in model_name else "deepseek")
    os.makedirs(output_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as f:
        cell_prompt = f.read().strip()

    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print("未找到图片")
        return

    print(f"\n{'=' * 60}")
    print(f"半自动录制工具 - {model_name}")
    print(f"{'=' * 60}")
    print(f"\n图片目录: {image_dir}")
    print(f"结果目录: {output_dir}")
    print(f"共 {len(image_files)} 张图片\n")

    print("操作说明:")
    print(f"  1. 工具会显示当前图片路径和统一 prompt")
    print(f"  2. 在 {model_name} 网页版中上传图片、粘贴 prompt")
    print(f"  3. 将模型返回的完整文本粘贴回工具")
    print(f"  4. 输入 'skip' 跳过当前图片")
    print(f"  5. 输入 'quit' 退出\n")

    input("按 Enter 开始...")

    for i, img_file in enumerate(image_files, 1):
        basename = Path(img_file).stem
        result_file = os.path.join(output_dir, f"{basename}_result.json")

        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"\n[{i}/{len(image_files)}] {img_file} - 已完成 ({existing.get('count', '?')} 个细胞)")
            continue

        img_path = os.path.join(image_dir, img_file)

        from PIL import Image
        image = Image.open(img_path)
        w, h = image.size

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(image_files)}] 图片: {img_file} ({w}x{h})")
        print(f"{'=' * 60}")
        print(f"\n--- 统一 Prompt（复制到 {model_name} 网页版）---")
        print(cell_prompt)
        print(f"--- Prompt 结束 ---\n")

        while True:
            print("请将模型返回的完整文本粘贴到下方（多行输入，以单独一行的 'END' 结束）:")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == "END":
                    break
                lines.append(line)

            raw_response = "\n".join(lines)

            if raw_response.strip().lower() == "skip":
                print("跳过当前图片")
                break

            if raw_response.strip().lower() == "quit":
                print("退出录制")
                return

            if not raw_response.strip():
                print("输入为空，请重新输入\n")
                continue

            boxes = parse_boxes(raw_response, w, h)
            count = parse_simple_count(raw_response)

            if boxes:
                points = []
                for box in boxes:
                    cx = (box["x1"] + box["x2"]) / 2
                    cy = (box["y1"] + box["y2"]) / 2
                    points.append({"x": cx, "y": cy})
                count = len(points)
            elif count > 0:
                points = []
                print(f"未解析到坐标，但检测到计数: {count}")
            else:
                print("未检测到坐标或计数，请检查输入")
                retry = input("重新输入? (y/n): ").strip().lower()
                if retry == "y":
                    continue
                else:
                    points = []
                    count = 0
                    break

            print(f"解析结果: {count} 个细胞")
            print(f"坐标点: {len(points)} 个")

            confirm = input("确认保存? (y/n): ").strip().lower()
            if confirm == "y":
                save_data = {
                    "image": img_file,
                    "model": model_name,
                    "count": count,
                    "points": points,
                    "raw_response": raw_response,
                    "elapsed": None,
                }
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                print(f"已保存到: {result_file}")
                break
            else:
                print("重新输入\n")

    print(f"\n全部录制完成! 结果保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="半自动录制工具")
    parser.add_argument("--model", type=str, required=True,
                        choices=["豆包", "DeepSeek", "doubao", "deepseek"],
                        help="模型名称")
    args = parser.parse_args()

    model_name = args.model
    if model_name in ("doubao", "豆包"):
        model_name = "豆包"
    elif model_name in ("deepseek", "DeepSeek"):
        model_name = "DeepSeek"

    run_recorder(model_name)