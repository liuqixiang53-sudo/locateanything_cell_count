import os
import sys
import json
import re
import time
import traceback
import yaml
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Eagle", "Embodied"))
sys.path.insert(0, PROJECT_ROOT)

from locateanything_worker import LocateAnythingWorker
from count_simple import nms, parse_boxes
from model_adapters.ollama_adapter import OllamaVLAdapter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def natural_sort_key(s):
    import re as _re
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]


def get_image_files(image_dir: str) -> list[str]:
    files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=natural_sort_key)
    return [os.path.join(image_dir, f) for f in files]


def run_locateanything(image: Image.Image, prompt: str, worker: LocateAnythingWorker,
                       nms_iou: float = 0.3, temperature: float = 0.1,
                       generation_mode: str = "hybrid") -> dict:
    start = time.time()
    result = worker.predict(image, prompt, generation_mode=generation_mode,
                            temperature=temperature)
    elapsed = time.time() - start

    answer = result["answer"]
    w, h = image.size
    boxes = parse_boxes(answer, w, h)
    raw_count = len(boxes)
    boxes = nms(boxes, nms_iou)

    points = []
    for box in boxes:
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        points.append({"x": cx, "y": cy})

    return {
        "count": len(points),
        "raw_count": raw_count,
        "points": points,
        "raw_response": answer,
        "elapsed": elapsed,
        "boxes": boxes,
    }


def run_ollama_count(image: Image.Image, prompt: str, adapter: OllamaVLAdapter,
                     temperature: float = 0.1) -> dict:
    result = adapter.predict(image, prompt, temperature=temperature)
    answer = result["answer"]
    elapsed = result["elapsed"]

    count = 0
    for m in re.finditer(r"\d+", answer):
        count = int(m.group())
        break

    return {
        "count": count,
        "raw_count": count,
        "points": [],
        "raw_response": answer,
        "elapsed": elapsed,
        "has_coordinates": False,
    }


def run_ollama_boxes(image: Image.Image, prompt: str, adapter: OllamaVLAdapter,
                     nms_iou: float = 0.3, temperature: float = 0.1) -> dict:
    result = adapter.predict(image, prompt, temperature=temperature)
    answer = result["answer"]
    elapsed = result["elapsed"]

    w, h = image.size
    boxes = adapter.parse_boxes(answer, w, h)
    raw_count = len(boxes)
    boxes = nms(boxes, nms_iou)

    points = []
    for box in boxes:
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        points.append({"x": cx, "y": cy})

    return {
        "count": len(points),
        "raw_count": raw_count,
        "points": points,
        "raw_response": answer,
        "elapsed": elapsed,
        "has_coordinates": len(boxes) > 0,
    }


def run_experiment(config_path: str):
    config = load_config(config_path)
    exp = config["experiment"]
    models = config["models"]

    image_dir = os.path.join(PROJECT_ROOT, exp["image_dir"])
    output_dir = os.path.join(PROJECT_ROOT, exp["output_dir"])

    prompt_path = os.path.join(PROJECT_ROOT, "对照实验", "prompts", "cell_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        cell_prompt = f.read().strip()

    nms_iou = exp.get("nms_iou_threshold", 0.3)
    temperature = exp.get("temperature", 0.1)

    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"未找到图片: {image_dir}")
        return

    print(f"找到 {len(image_files)} 张图片")
    print(f"统一 Prompt:\n{cell_prompt}\n")

    # 延迟加载模型
    locateanything_worker = None
    ollama_adapters = {}

    for model_cfg in models:
        model_name = model_cfg["name"]
        model_type = model_cfg["type"]
        subdir = model_cfg["output_subdir"]
        model_output_dir = os.path.join(output_dir, subdir)
        os.makedirs(model_output_dir, exist_ok=True)

        if model_type == "manual":
            print(f"[跳过] {model_name} - 需要手动操作")
            continue

        if model_cfg.get("disabled", False):
            print(f"[跳过] {model_name} - 已禁用")
            continue

        print(f"\n{'=' * 60}")
        print(f"运行模型: {model_name}")
        print(f"{'=' * 60}")

        if model_type == "local":
            if locateanything_worker is None:
                model_path = os.path.join(PROJECT_ROOT, model_cfg["model_path"])
                print(f"加载模型: {model_path}")
                locateanything_worker = LocateAnythingWorker(
                    model_path=model_path, device="cuda"
                )
            generation_mode = model_cfg.get("generation_mode", "hybrid")
            worker = locateanything_worker

        elif model_type == "ollama":
            ollama_model = model_cfg["model_name"]
            ollama_host = model_cfg.get("ollama_host", "http://localhost:11434")
            if ollama_model not in ollama_adapters:
                print(f"连接 Ollama: {ollama_host} / {ollama_model}")
                ollama_adapters[ollama_model] = OllamaVLAdapter(
                    model_name=ollama_model, host=ollama_host
                )
            adapter = ollama_adapters[ollama_model]
            counting_mode = model_cfg.get("counting_mode", False)
            if counting_mode:
                ollama_prompt = model_cfg.get("counting_prompt", cell_prompt)
                print(f"使用计数模式, prompt: {ollama_prompt}")

        else:
            print(f"未知模型类型: {model_type}")
            continue

        results_summary = {}
        total_elapsed = 0

        for i, img_path in enumerate(image_files, 1):
            basename = Path(img_path).stem
            result_file = os.path.join(model_output_dir, f"{basename}_result.json")

            if os.path.exists(result_file):
                print(f"[{i}/{len(image_files)}] {basename} - 已存在，跳过")
                with open(result_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                results_summary[basename] = existing.get("count", 0)
                continue

            print(f"[{i}/{len(image_files)}] 处理: {basename}", end="", flush=True)

            try:
                image = Image.open(img_path).convert("RGB")

                if model_type == "local":
                    result = run_locateanything(
                        image, cell_prompt, worker,
                        nms_iou=nms_iou, temperature=temperature,
                        generation_mode=generation_mode,
                    )
                elif model_type == "ollama" and counting_mode:
                    result = run_ollama_count(
                        image, ollama_prompt, adapter,
                        temperature=temperature,
                    )
                else:
                    result = run_ollama_boxes(
                        image, cell_prompt, adapter,
                        nms_iou=nms_iou, temperature=temperature,
                    )

                elapsed = result["elapsed"]
                total_elapsed += elapsed
                print(f" -> {result['count']} 个细胞 ({elapsed:.1f}s)")

                save_data = {
                    "image": os.path.basename(img_path),
                    "model": model_name,
                    "count": result["count"],
                    "raw_count": result["raw_count"],
                    "points": result["points"],
                    "raw_response": result["raw_response"],
                    "elapsed": elapsed,
                    "has_coordinates": result.get("has_coordinates", len(result["points"]) > 0),
                }
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)

                results_summary[basename] = result["count"]

            except Exception as e:
                print(f" -> 失败: {e}")
                traceback.print_exc()
                results_summary[basename] = None

            if model_type == "local" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary_file = os.path.join(model_output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        print(f"\n{model_name} 完成. 总耗时: {total_elapsed:.1f}s")
        print(f"结果保存至: {model_output_dir}")

    print("\n全部模型运行完毕!")


if __name__ == "__main__":
    config_path = os.path.join(PROJECT_ROOT, "对照实验", "config.yaml")
    run_experiment(config_path)