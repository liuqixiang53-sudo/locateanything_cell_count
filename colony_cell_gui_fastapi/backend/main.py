#!/usr/bin/env python3
"""
菌落与细胞计数分析软件 - FastAPI 后端 V1.0
基于 LocateAnything 模型，复用 count_simple.py 和 colony_count_adaptive.py

API 端点:
  POST /api/colony/count   - 菌落计数
  POST /api/cell/count     - 细胞计数（支持框选区域）
  GET  /api/image/{path}   - 获取图片
  GET  /api/health         - 健康检查

Usage:
    cd colony_cell_gui_fastapi/backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import io
import copy
import traceback
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---- 添加依赖路径 ----
# __file__ is in colony_cell_gui_fastapi/backend/main.py
# We need PROJECT_ROOT to be /data/qixiangliu/pycode/locateanthing_colony_count
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Eagle", "Embodied"))
sys.path.insert(0, PROJECT_ROOT)

from locateanything_worker import LocateAnythingWorker
from count_simple import nms as cell_nms, parse_boxes as cell_parse_boxes
from colony_count_adaptive import AdaptiveColonyCounter

# ============================================================
# 常量配置
# ============================================================
MODEL_PATH = os.path.join(PROJECT_ROOT, "LocateAnything-3B")
DEVICE = "cuda"
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "colony_cell_gui_fastapi", "uploads")
RESULT_DIR = os.path.join(PROJECT_ROOT, "colony_cell_gui_fastapi", "results")

CELL_PROMPT = (
    "Locate all the instances that match the following description: cell. "
    "These appear as small circular, oval, or semi-circular shapes. "
    "When cells are crowded or touching, they may be deformed, squeezed into "
    "ellipses or half-circles. Even if cells are touching or overlapping, "
    "each cell must be enclosed in a SEPARATE bounding box. "
    "Do NOT group multiple cells into one large box. "
    "Each bounding box should tightly enclose a single cell."
)

# 图片最大边长，限制显存使用
MAX_IMAGE_SIZE = 2048

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="菌落与细胞计数分析软件",
    description="基于 LocateAnything 模型的目标检测与计数服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 全局模型（延迟加载，菌落和细胞计数共用同一实例）
# ============================================================
SHARED_WORKER = None
COLONY_COUNTER = None


def get_shared_worker() -> LocateAnythingWorker:
    global SHARED_WORKER
    if SHARED_WORKER is None:
        print(f"Loading shared model from: {MODEL_PATH}")
        print(f"Using device: {DEVICE}")
        SHARED_WORKER = LocateAnythingWorker(model_path=MODEL_PATH, device=DEVICE)
    return SHARED_WORKER


def get_colony_counter() -> AdaptiveColonyCounter:
    global COLONY_COUNTER
    if COLONY_COUNTER is None:
        COLONY_COUNTER = AdaptiveColonyCounter(worker=get_shared_worker(), device=DEVICE)
    return COLONY_COUNTER


def get_cell_worker() -> LocateAnythingWorker:
    return get_shared_worker()


def cleanup_cuda_memory():
    """清理 CUDA 显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[CUDA Cleanup] Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")


def limit_image_size(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """限制图片最大边长，避免超大图片导致显存溢出"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


# ============================================================
# 工具函数
# ============================================================

def load_font(size=24):
    """加载字体"""
    candidates = [
        os.path.join(PROJECT_ROOT, "LocateAnything", "assets", "LXGWWenKai-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def image_to_base64(img: Image.Image) -> str:
    """将 PIL Image 转换为 base64 字符串"""
    buffered = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def draw_points_on_image(image: Image.Image, points: list, point_radius=5) -> Image.Image:
    """在图像上绘制点。红点=自动检测，绿点=手动添加"""
    if image is None:
        return None
    draw_image = image.copy()
    if draw_image.mode != "RGB":
        draw_image = draw_image.convert("RGB")
    draw = ImageDraw.Draw(draw_image)

    for p in points:
        x, y = p["x"], p["y"]
        color = (0, 255, 0) if p.get("source") == "manual" else (255, 0, 0)
        r = point_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)

    # 添加计数文字
    if points:
        font = load_font(24)
        count_text = f"Count: {len(points)}"
        bbox = draw.textbbox((0, 0), count_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = 8
        draw.rectangle(
            [margin, margin, margin + tw + margin * 2, margin + th + margin * 2],
            fill="white",
        )
        draw.text((margin * 2, margin), count_text, fill="black", font=font)

    return draw_image


def decode_image_file(file_data: bytes) -> Image.Image:
    """从字节数据解码为 PIL Image"""
    try:
        return Image.open(io.BytesIO(file_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片格式无效: {str(e)}")


# ============================================================
# 菌落计数接口
# ============================================================

@app.post("/api/colony/count")
async def colony_count(
    file: UploadFile = File(...),
    sparse_threshold: int = Form(50),
    dense_threshold: int = Form(150),
    nms_iou_threshold: float = Form(0.3),
    dedup_distance: float = Form(3.0),
    clahe_clip_limit: float = Form(2.0),
    point_radius: int = Form(2),
    tile_grid_size: int = Form(8),
    watershed_area_factor: float = Form(2.0),
    watershed_dist_ratio: float = Form(0.3),
    watershed_min_area_ratio: float = Form(0.0005),
):
    """菌落自动计数"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        image = limit_image_size(image)

        counter = get_colony_counter()
        result = counter.count_colonies_adaptive(
            image,
            sparse_threshold=int(sparse_threshold),
            dense_threshold=int(dense_threshold),
            nms_iou_threshold=float(nms_iou_threshold),
            dedup_distance=float(dedup_distance),
            clahe_clip_limit=float(clahe_clip_limit),
            tile_grid_size=int(tile_grid_size),
            watershed_area_factor=float(watershed_area_factor),
            watershed_dist_ratio=float(watershed_dist_ratio),
            watershed_min_area_ratio=float(watershed_min_area_ratio),
        )

        centers = result.get("centers", [])
        points = [{"x": c["x"], "y": c["y"], "source": "auto"} for c in centers]
        vis_image = draw_points_on_image(image, points, point_radius=int(point_radius))

        return JSONResponse({
            "success": True,
            "count": len(points),
            "raw_count": result.get("raw_count", 0),
            "stage": result.get("stage", ""),
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "count": 0,
            "points": [],
            "image_base64": None,
        }, status_code=500)
    finally:
        cleanup_cuda_memory()


@app.post("/api/colony/add_point")
async def colony_add_point(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    points_json: str = Form("[]"),
):
    """手动添加菌落点"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        points = json.loads(points_json)
        points.append({"x": float(x), "y": float(y), "source": "manual"})

        vis_image = draw_points_on_image(image, points)
        return JSONResponse({
            "success": True,
            "count": len(points),
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/colony/remove_point")
async def colony_remove_point(
    file: UploadFile = File(...),
    index: int = Form(...),
    points_json: str = Form("[]"),
):
    """删除菌落点"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        points = json.loads(points_json)
        if 0 <= index < len(points):
            points.pop(index)

        vis_image = draw_points_on_image(image, points)
        return JSONResponse({
            "success": True,
            "count": len(points),
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============================================================
# 细胞计数接口
# ============================================================

@app.post("/api/cell/count")
async def cell_count(
    file: UploadFile = File(...),
    x1: float = Form(None),
    y1: float = Form(None),
    x2: float = Form(None),
    y2: float = Form(None),
    nms_iou_threshold: float = Form(0.3),
    point_radius: int = Form(2),
    temperature: float = Form(0.1),
    generation_mode: str = Form("hybrid"),
):
    """细胞计数，支持框选区域"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        image = limit_image_size(image)

        crop_info = "使用整图"
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            crop_box = (int(x1), int(y1), int(x2), int(y2))
            if crop_box[2] > crop_box[0] + 10 and crop_box[3] > crop_box[1] + 10:
                image = image.crop(crop_box)
                crop_info = f"裁剪区域: ({crop_box[0]},{crop_box[1]})-({crop_box[2]},{crop_box[3]})"

        image_width, image_height = image.size

        worker = get_cell_worker()
        result = worker.predict(
            image,
            CELL_PROMPT,
            generation_mode=generation_mode,
            temperature=float(temperature),
        )

        answer = result["answer"]
        boxes = cell_parse_boxes(answer, image_width, image_height)
        raw_count = len(boxes)
        boxes = cell_nms(boxes, float(nms_iou_threshold))

        points = []
        for box in boxes:
            cx = (box["x1"] + box["x2"]) / 2
            cy = (box["y1"] + box["y2"]) / 2
            points.append({"x": cx, "y": cy, "source": "auto"})

        vis_image = draw_points_on_image(image, points, point_radius=int(point_radius))

        return JSONResponse({
            "success": True,
            "count": len(points),
            "raw_count": raw_count,
            "crop_info": crop_info,
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "count": 0,
            "points": [],
            "image_base64": None,
        }, status_code=500)
    finally:
        cleanup_cuda_memory()


@app.post("/api/cell/add_point")
async def cell_add_point(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    points_json: str = Form("[]"),
):
    """手动添加细胞点"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        points = json.loads(points_json)
        points.append({"x": float(x), "y": float(y), "source": "manual"})

        vis_image = draw_points_on_image(image, points)
        return JSONResponse({
            "success": True,
            "count": len(points),
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/cell/remove_point")
async def cell_remove_point(
    file: UploadFile = File(...),
    index: int = Form(...),
    points_json: str = Form("[]"),
):
    """删除细胞点"""
    try:
        file_data = await file.read()
        image = decode_image_file(file_data)
        points = json.loads(points_json)
        if 0 <= index < len(points):
            points.pop(index)

        vis_image = draw_points_on_image(image, points)
        return JSONResponse({
            "success": True,
            "count": len(points),
            "points": points,
            "image_base64": image_to_base64(vis_image),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============================================================
# 对照实验 API
# ============================================================

import subprocess as _subprocess


@app.post("/api/experiment/run")
async def experiment_run():
    """运行对照实验（LocateAnything + Ollama 模型）"""
    try:
        runner_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "experiment_runner.py"
        )
        proc = _subprocess.Popen(
            [sys.executable, runner_path],
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        output_lines = []
        for line in proc.stdout:
            output_lines.append(line.rstrip())

        proc.wait()

        return JSONResponse({
            "success": proc.returncode == 0,
            "output": "\n".join(output_lines),
            "returncode": proc.returncode,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/experiment/status")
async def experiment_status():
    """查询各模型结果完成情况"""
    results_base = os.path.join(PROJECT_ROOT, "对照实验", "results")
    gt_dir = os.path.join(PROJECT_ROOT, "对照实验", "ground_truth")

    status = {}
    total_images = 20

    if os.path.exists(gt_dir):
        gt_files = [f for f in os.listdir(gt_dir) if f.endswith("_gt.json")]
        if gt_files:
            total_images = len(gt_files)
            status["ground_truth"] = f"{len(gt_files)} / {total_images}"

    if os.path.exists(results_base):
        for model_dir in sorted(os.listdir(results_base)):
            full = os.path.join(results_base, model_dir)
            if os.path.isdir(full):
                result_files = [f for f in os.listdir(full) if f.endswith("_result.json")]
                status[model_dir] = f"{len(result_files)} / {total_images}"

    return JSONResponse({"status": status, "total_images": total_images})


@app.get("/api/experiment/results")
async def experiment_results():
    """获取评估结果"""
    eval_path = os.path.join(PROJECT_ROOT, "对照实验", "evaluation.json")
    if not os.path.exists(eval_path):
        return JSONResponse({"error": "评估结果尚未生成，请先运行 evaluation.py"}, status_code=404)

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/api/experiment/report")
async def experiment_report():
    """获取 Markdown 报告"""
    report_path = os.path.join(PROJECT_ROOT, "对照实验", "report.md")
    if not os.path.exists(report_path):
        return JSONResponse({"error": "报告尚未生成"}, status_code=404)

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return JSONResponse({"content": content})


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": SHARED_WORKER is not None}


# ============================================================
# 启动前端页面
# ============================================================

@app.get("/")
async def serve_frontend():
    """返回前端页面"""
    frontend_path = os.path.join(PROJECT_ROOT, "colony_cell_gui_fastapi", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "前端文件未找到"}, status_code=404)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )