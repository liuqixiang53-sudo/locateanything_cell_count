#!/usr/bin/env python3
"""
评估脚本 - 对比各模型结果与 ground truth。

Metrics:
  - MAE (平均绝对误差)
  - RMSE (均方根误差)
  - MAPE (平均绝对百分比误差)
  - Precision / Recall / F1 (基于点距离匹配)
  - 平均推理耗时

Usage:
    python evaluation.py
"""

import os
import sys
import json
import math
import yaml
from pathlib import Path
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_ground_truth(gt_dir: str) -> dict[str, dict]:
    gt = {}
    for f in os.listdir(gt_dir):
        if not f.endswith(".json"):
            continue
        if f == "README.md":
            continue
        with open(os.path.join(gt_dir, f), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        image_id = f.replace(".json", "")
        points = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "point":
                pt = shape["points"][0]
                points.append({"x": pt[0], "y": pt[1]})
        gt[image_id] = {
            "image": data.get("imagePath", f"{image_id}.jpg"),
            "total_count": len(points),
            "points": points,
        }
    return gt


def load_model_results(result_dir: str) -> dict[str, dict]:
    results = {}
    for f in os.listdir(result_dir):
        if f.endswith("_result.json") and f != "summary.json":
            with open(os.path.join(result_dir, f), "r", encoding="utf-8") as fp:
                data = json.load(fp)
            image_id = f.replace("_result.json", "")
            results[image_id] = data
    return results


def match_points(gt_points: list[dict], pred_points: list[dict],
                 distance_threshold: float = 10.0) -> tuple[int, int, int]:
    """
    基于匈牙利算法进行点匹配。
    Returns: (tp, fp, fn)
    """
    if not gt_points or not pred_points:
        return 0, len(pred_points), len(gt_points)

    n_gt = len(gt_points)
    n_pred = len(pred_points)

    cost_matrix = []
    for i in range(n_gt):
        row = []
        for j in range(n_pred):
            dx = gt_points[i]["x"] - pred_points[j]["x"]
            dy = gt_points[i]["y"] - pred_points[j]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            row.append(dist if dist <= distance_threshold else 1e9)
        cost_matrix.append(row)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] <= distance_threshold:
            tp += 1

    fp = n_pred - tp
    fn = n_gt - tp

    return tp, fp, fn


def compute_metrics(gt_counts: list[int], pred_counts: list[int]) -> dict:
    n = len(gt_counts)
    if n == 0:
        return {}

    errors = [abs(p - g) for p, g in zip(pred_counts, gt_counts)]
    squared_errors = [(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)]

    mae = sum(errors) / n
    rmse = math.sqrt(sum(squared_errors) / n)

    mape_vals = []
    for p, g in zip(pred_counts, gt_counts):
        if g > 0:
            mape_vals.append(abs(p - g) / g * 100)
    mape = sum(mape_vals) / len(mape_vals) if mape_vals else 0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


def compute_point_metrics(all_gt_points: dict[str, list[dict]],
                          all_pred_points: dict[str, list[dict]],
                          distance_threshold: float = 10.0) -> dict:
    total_tp, total_fp, total_fn = 0, 0, 0

    for image_id in all_gt_points:
        gt = all_gt_points.get(image_id, [])
        pred = all_pred_points.get(image_id, [])
        tp, fp, fn = match_points(gt, pred, distance_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def run_evaluation():
    config_path = os.path.join(PROJECT_ROOT, "对照实验", "config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    point_match_distance = eval_cfg.get("point_match_distance", 20)

    gt_dir = os.path.join(PROJECT_ROOT, "对照实验", "ground_truth", "lable")
    results_base = os.path.join(PROJECT_ROOT, "对照实验", "results")

    if not os.path.exists(gt_dir):
        print(f"Ground truth 目录不存在: {gt_dir}")
        print("请先完成标注后再运行评估")
        return

    gt_data = load_ground_truth(gt_dir)
    if not gt_data:
        print("未找到 ground truth 文件")
        return

    print(f"加载 {len(gt_data)} 个 ground truth")
    for image_id, data in gt_data.items():
        print(f"  {image_id}: {data.get('total_count', len(data.get('points', [])))} 个细胞")

    model_dirs = {}
    for d in os.listdir(results_base):
        full = os.path.join(results_base, d)
        if os.path.isdir(full) and not d.startswith("."):
            model_dirs[d] = full

    if not model_dirs:
        print("未找到模型结果")
        return

    all_results = {}

    print(f"\n{'=' * 80}")
    print("逐模型评估")
    print(f"{'=' * 80}")

    for model_name, model_dir in sorted(model_dirs.items()):
        model_results = load_model_results(model_dir)
        if not model_results:
            print(f"\n{model_name}: 无结果")
            continue

        gt_counts = []
        pred_counts = []
        per_image = {}

        gt_points_dict = {}
        pred_points_dict = {}

        for image_id in gt_data:
            if image_id not in model_results:
                continue
            gt_count = gt_data[image_id].get("total_count", len(gt_data[image_id].get("points", [])))
            pred_count = model_results[image_id].get("count", 0)
            gt_counts.append(gt_count)
            pred_counts.append(pred_count)
            per_image[image_id] = {"gt": gt_count, "pred": pred_count,
                                   "error": pred_count - gt_count}

            gt_points_dict[image_id] = gt_data[image_id].get("points", [])
            pred_points_dict[image_id] = model_results[image_id].get("points", [])

        count_metrics = compute_metrics(gt_counts, pred_counts)

        has_coords = any(
            len(model_results[iid].get("points", [])) > 0
            for iid in gt_data if iid in model_results
        )

        if has_coords:
            point_metrics = compute_point_metrics(gt_points_dict, pred_points_dict, point_match_distance)
        else:
            point_metrics = {"precision": None, "recall": None, "f1": None,
                             "total_tp": 0, "total_fp": 0, "total_fn": 0}

        elapsed_vals = [model_results[iid].get("elapsed", 0) or 0
                        for iid in gt_data if iid in model_results]
        avg_elapsed = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else 0

        success_count = sum(1 for iid in gt_data if iid in model_results)
        success_rate = success_count / len(gt_data) * 100

        result = {
            **count_metrics,
            **point_metrics,
            "avg_elapsed": round(avg_elapsed, 2),
            "success_rate": round(success_rate, 1),
            "per_image": per_image,
        }
        all_results[model_name] = result

        print(f"\n{model_name}:")
        print(f"  MAE: {count_metrics['mae']}, RMSE: {count_metrics['rmse']}, "
              f"MAPE: {count_metrics['mape']}%")
        if point_metrics["f1"] is not None:
            print(f"  Precision: {point_metrics['precision']:.4f}, "
                  f"Recall: {point_metrics['recall']:.4f}, "
                  f"F1: {point_metrics['f1']:.4f}")
        else:
            print(f"  Precision/Recall/F1: N/A (计数模式)")
        print(f"  平均耗时: {avg_elapsed:.1f}s, 成功率: {success_rate}%")

    eval_output = os.path.join(PROJECT_ROOT, "对照实验", "evaluation.json")
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果保存至: {eval_output}")

    return all_results


if __name__ == "__main__":
    run_evaluation()