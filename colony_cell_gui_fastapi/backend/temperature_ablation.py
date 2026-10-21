#!/usr/bin/env python3
"""
温度敏感性消融实验 - LocateAnything-3B

同一批图片、统一 prompt，仅改变采样温度，可指定重复次数评估随机性。
结果存入 对照实验/ablation/locateanything_t{temp}[_r{rep}]/，逐温度评估并聚合均值/标准差。

Usage:
    python temperature_ablation.py                     # 全网格推理 + 评估
    python temperature_ablation.py --temps 0,1 --repeats 3
    python temperature_ablation.py --evaluate          # 仅评估已有结果
"""

import os
import sys
import re
import json
import math
import argparse
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Eagle", "Embodied"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BACKEND_DIR)

from PIL import Image
import torch

from locateanything_worker import LocateAnythingWorker
from count_simple import nms, parse_boxes
from evaluation import load_ground_truth, compute_metrics, compute_point_metrics
from experiment_runner import get_image_files, run_locateanything

TEMPERATURES = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
NMS_IOU = 0.3
GENERATION_MODE = "hybrid"
ABLATION_BASE = "ablation"
POINT_MATCH_DISTANCE = 20


def temp_subdir(temp: float, rep: int = 1) -> str:
    return f"locateanything_t{temp:g}" + ("" if rep == 1 else f"_r{rep}")


def run_inference(temps=None, repeats=1):
    temps = temps if temps else TEMPERATURES
    image_files = get_image_files(os.path.join(PROJECT_ROOT, "对照实验", "images"))
    prompt = open(os.path.join(PROJECT_ROOT, "对照实验", "prompts", "cell_prompt.txt"),
                  encoding="utf-8").read().strip()

    model_path = os.path.join(PROJECT_ROOT, "LocateAnything-3B")
    print(f"加载模型: {model_path}")
    worker = LocateAnythingWorker(model_path=model_path, device="cuda")

    for temp in temps:
        for rep in range(1, repeats + 1):
            out_dir = os.path.join(PROJECT_ROOT, "对照实验", ABLATION_BASE, temp_subdir(temp, rep))
            os.makedirs(out_dir, exist_ok=True)
            summary = {}
            print(f"\n=== temperature = {temp} | run {rep}/{repeats} ===")
            for i, img_path in enumerate(image_files, 1):
                basename = os.path.splitext(os.path.basename(img_path))[0]
                result_file = os.path.join(out_dir, f"{basename}_result.json")
                if os.path.exists(result_file):
                    existing = json.load(open(result_file, encoding="utf-8"))
                    summary[basename] = existing.get("count", 0)
                    print(f"[{i}/{len(image_files)}] {basename} - 已存在，跳过")
                    continue
                print(f"[{i}/{len(image_files)}] 处理: {basename}", end="", flush=True)
                try:
                    image = Image.open(img_path).convert("RGB")
                    result = run_locateanything(
                        image, prompt, worker,
                        nms_iou=NMS_IOU, temperature=temp,
                        generation_mode=GENERATION_MODE,
                    )
                    print(f" -> {result['count']} ({result['elapsed']:.1f}s)")
                    save_data = {
                        "image": os.path.basename(img_path),
                        "model": f"LocateAnything-3B (T={temp})",
                        "count": result["count"],
                        "raw_count": result["raw_count"],
                        "points": result["points"],
                        "raw_response": result["raw_response"],
                        "elapsed": result["elapsed"],
                        "has_coordinates": True,
                    }
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                    summary[basename] = result["count"]
                except Exception as e:
                    print(f" -> 失败: {e}")
                    traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
    del worker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def metrics_for_dir(model_dir: str, gt_data: dict) -> dict:
    model_results = {}
    for f in os.listdir(model_dir):
        if f.endswith("_result.json"):
            model_results[f.replace("_result.json", "")] = json.load(
                open(os.path.join(model_dir, f), encoding="utf-8"))
    if not model_results:
        return {}

    gt_counts, pred_counts, per_image = [], [], {}
    gt_pts, pred_pts = {}, {}
    elapsed = []
    for image_id in gt_data:
        if image_id not in model_results:
            continue
        gt_count = gt_data[image_id].get("total_count", len(gt_data[image_id].get("points", [])))
        pred = model_results[image_id]
        gt_counts.append(gt_count)
        pred_counts.append(pred.get("count", 0))
        per_image[image_id] = {"gt": gt_count, "pred": pred.get("count", 0),
                               "error": pred.get("count", 0) - gt_count}
        gt_pts[image_id] = gt_data[image_id].get("points", [])
        pred_pts[image_id] = pred.get("points", [])
        elapsed.append(pred.get("elapsed") or 0)

    metrics = compute_metrics(gt_counts, pred_counts)
    pm = compute_point_metrics(gt_pts, pred_pts, POINT_MATCH_DISTANCE)
    return {
        **metrics, **pm,
        "avg_elapsed": round(sum(elapsed) / len(elapsed), 2),
        "n_images": len(gt_counts),
        "per_image": per_image,
    }


def mean_std(vals):
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return round(m, 2), round(sd, 2)


def evaluate():
    gt_data = load_ground_truth(os.path.join(PROJECT_ROOT, "对照实验", "ground_truth", "lable"))
    results_base = os.path.join(PROJECT_ROOT, "对照实验", ABLATION_BASE)

    all_results = {}
    for temp in TEMPERATURES:
        run_dirs = sorted(
            [d for d in os.listdir(results_base)
             if re.fullmatch(rf"locateanything_t{re.escape(f'{temp:g}')}(_r\d+)?", d)]
            if os.path.isdir(results_base) else [],
            key=lambda d: 0 if "_r" not in d else int(d.rsplit("_r", 1)[1]))
        if not run_dirs:
            continue

        runs = []
        for rd in run_dirs:
            m = metrics_for_dir(os.path.join(results_base, rd), gt_data)
            if m:
                m["dir"] = rd
                runs.append(m)
        if not runs:
            continue

        base = dict(runs[0])
        base.pop("per_image", None)
        mae_mean, mae_std = mean_std([r["mae"] for r in runs])
        f1_mean, f1_std = mean_std([r["f1"] for r in runs])
        base.update({
            "mae_mean": mae_mean, "mae_std": mae_std,
            "f1_mean": f1_mean, "f1_std": f1_std,
            "n_runs": len(runs),
            "runs": [{k: r[k] for k in ("dir", "mae", "mape", "precision", "recall", "f1")}
                     for r in runs],
        })
        all_results[f"{temp:g}"] = base
        std_txt = f" ± {mae_std}" if len(runs) > 1 else ""
        print(f"T={temp:<4} runs={len(runs)}  MAE={runs[0]['mae']}"
              f"{std_txt} (mean {mae_mean})  F1 mean={f1_mean}")

    out = os.path.join(PROJECT_ROOT, "对照实验", "temperature_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if all_results:
        best = min(all_results,
                   key=lambda t: (all_results[t]["mae_mean"],
                                  -all_results[t].get("f1_mean") or -all_results[t]["f1"],
                                  float(t)))
        b = all_results[best]
        print(f"\n最优温度（按 MAE 均值）: T={best}  MAE={b['mae_mean']}±{b['mae_std']} "
              f"F1={b.get('f1_mean')} ({b['n_runs']} 次)")
        print(f"已保存: {out}")
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true", help="仅评估已有结果")
    ap.add_argument("--temps", type=str, default=None, help="逗号分隔温度子集，如 0,1")
    ap.add_argument("--repeats", type=int, default=1, help="每温度重复次数")
    args = ap.parse_args()

    if not args.evaluate:
        temps = [float(x) for x in args.temps.split(",")] if args.temps else None
        run_inference(temps=temps, repeats=args.repeats)
    evaluate()


if __name__ == "__main__":
    main()
