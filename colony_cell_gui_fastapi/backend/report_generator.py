#!/usr/bin/env python3
"""
报告生成器 - 基于评估结果生成 Markdown 对比报告。

Usage:
    python report_generator.py
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_ground_truth_counts(gt_dir: str) -> dict[str, int]:
    gt_counts = {}
    if not os.path.exists(gt_dir):
        return gt_counts
    for f in os.listdir(gt_dir):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(gt_dir, f), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        image_id = f.replace(".json", "")
        points = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "point":
                pt = shape["points"][0]
                points.append({"x": pt[0], "y": pt[1]})
        gt_counts[image_id] = len(points)
    return gt_counts


def generate_report():
    eval_path = os.path.join(PROJECT_ROOT, "对照实验", "evaluation.json")
    if not os.path.exists(eval_path):
        print("请先运行 evaluation.py 生成评估结果")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    config_path = os.path.join(PROJECT_ROOT, "对照实验", "config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    eval_cfg = config.get("evaluation", {})
    point_match_distance = eval_cfg.get("point_match_distance", 20)

    gt_dir = os.path.join(PROJECT_ROOT, "对照实验", "ground_truth", "lable")
    gt_counts = load_ground_truth_counts(gt_dir)

    lines = []
    lines.append("# 细胞计数多模型对比实验报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 实验概述\n")
    lines.append("- **任务**: 细胞计数（Cell Counting）")
    lines.append(f"- **测试图片数**: {len(gt_counts)} 张")
    lines.append("- **模型**: " + ", ".join(all_results.keys()))
    lines.append(f"- **点匹配距离阈值**: {point_match_distance} 像素\n")

    lines.append("## 2. 综合指标对比\n")

    headers = ["模型", "MAE ↓", "RMSE ↓", "MAPE% ↓", "Precision ↑", "Recall ↑", "F1 ↑", "平均耗时(s)", "成功率%"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["-" * len(h) for h in headers]) + "|")

    for model_name, result in sorted(all_results.items()):
        precision_str = f"{result['precision']:.4f}" if result['precision'] is not None else "N/A"
        recall_str = f"{result['recall']:.4f}" if result['recall'] is not None else "N/A"
        f1_str = f"{result['f1']:.4f}" if result['f1'] is not None else "N/A"
        row = [
            model_name,
            str(result["mae"]),
            str(result["rmse"]),
            f"{result['mape']:.1f}",
            precision_str,
            recall_str,
            f1_str,
            str(result["avg_elapsed"]),
            str(result["success_rate"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # 排名
    lines.append("## 3. 各指标排名\n")

    for metric, metric_name, ascending in [
        ("mae", "MAE（平均绝对误差）", True),
        ("f1", "F1 Score", False),
        ("rmse", "RMSE（均方根误差）", True),
        ("mape", "MAPE（平均绝对百分比误差）", True),
    ]:
        sorted_models = sorted(
            [(n, r) for n, r in all_results.items() if r[metric] is not None],
            key=lambda x: x[1][metric],
            reverse=not ascending,
        )
        lines.append(f"### {metric_name}\n")
        for rank, (name, result) in enumerate(sorted_models, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            lines.append(f"{emoji} **{name}**: {result[metric]}")
        lines.append("")

    lines.append("## 4. 逐图对比详情\n")

    all_image_ids = sorted(gt_counts.keys(),
                           key=lambda x: [int(t) if t.isdigit() else t for t in __import__('re').split(r'(\d+)', x)])

    detail_headers = ["图片", "Ground Truth"]
    detail_headers += [m for m in sorted(all_results.keys())]
    detail_headers += ["误差最小"]

    lines.append("| " + " | ".join(detail_headers) + " |")
    lines.append("|" + "|".join(["-" * len(h) for h in detail_headers]) + "|")

    for image_id in all_image_ids:
        row = [image_id, str(gt_counts.get(image_id, "?"))]
        best_error = float("inf")
        best_model = ""

        for model_name in sorted(all_results.keys()):
            per_image = all_results[model_name].get("per_image", {})
            info = per_image.get(image_id)
            if info:
                pred = info["pred"]
                error = info["error"]
                row.append(f"{pred} ({error:+d})")
                if abs(error) < abs(best_error):
                    best_error = error
                    best_model = f"{model_name}({pred})"
            else:
                row.append("N/A")

        row.append(best_model if best_model else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    lines.append("## 5. 结论与分析\n")

    best_f1 = max(
        [(n, r) for n, r in all_results.items() if r["f1"] is not None],
        key=lambda x: x[1]["f1"],
        default=(None, {"f1": 0}),
    )
    best_mae = min(all_results.items(), key=lambda x: x[1]["mae"])
    best_speed = min(all_results.items(), key=lambda x: x[1]["avg_elapsed"])

    if best_f1[0] is not None:
        lines.append(f"- **最佳 F1**: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    lines.append(f"- **最佳 MAE**: {best_mae[0]} ({best_mae[1]['mae']})")
    lines.append(f"- **最快推理**: {best_speed[0]} ({best_speed[1]['avg_elapsed']}s/张)")
    lines.append("")

    report_path = os.path.join(PROJECT_ROOT, "对照实验", "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    generate_report()