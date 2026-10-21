#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spaces  # MUST BE THE ABSOLUTE FIRST IMPORT FOR ZEROGPU EMULATION

import gradio as gr
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tempfile
import re
import time
import base64
import gc
import io
import json
import uuid
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from huggingface_hub import CommitScheduler

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "LXGWWenKai-Bold.ttf")


def _get_first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _configure_hf_auth():
    model_token = _get_first_env(
        "MODEL_HF_TOKEN",
        "LOG_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    )
    log_token = _get_first_env(
        "LOG_HF_TOKEN",
        "MODEL_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    )
    shared_token = model_token or log_token
    if shared_token:
        for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            os.environ[name] = shared_token
    return model_token, log_token


MODEL_HF_TOKEN, LOG_HF_TOKEN = _configure_hf_auth()
HF_TOKEN = MODEL_HF_TOKEN

def _load_font(size=20):
    """加载中文字体（LXGW WenKai），需提前放置到 assets/ 目录"""
    if os.path.exists(_FONT_PATH):
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ============================================================
# 颜色 / 解析 / 绘制
# ============================================================
def get_color_for_label(label):
    colors = [
        (8, 145, 178), (220, 38, 38), (22, 163, 74), (37, 99, 235),
        (217, 119, 6), (147, 51, 234),
    ]
    idx = sum(ord(c) for c in label)
    return colors[idx % len(colors)]


def parse_mixed_results(text, category_str=""):
    results = []
    expected_cats = [c.strip().lower() for c in category_str.split("</c>") if c.strip()]

    ref_box_pattern = r"(<ref>.*?</ref>)|(<box>.*?</box>)"
    current_label = None
    found_structured = False

    for m in re.finditer(ref_box_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            label_raw = re.sub(r"</?ref>", "", token, flags=re.IGNORECASE).strip()
            if label_raw:
                current_label = label_raw
        else:
            content = re.sub(r"</?box>", "", token, flags=re.IGNORECASE)
            nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
            coords = [float(n) for n in nums]
            if not coords:
                continue
            label = current_label
            if label is None:
                label = expected_cats[0] if expected_cats else "object"
            if len(coords) == 4:
                results.append({"type": "box", "coords": coords, "label": label})
            elif len(coords) == 2:
                results.append({"type": "point", "coords": coords, "label": label})
            found_structured = True

    if found_structured:
        return results

    box_pattern = r"<box>(.*?)</box>"
    parts = re.split(box_pattern, text)
    for i in range(1, len(parts), 2):
        preceding_text = parts[i - 1].lower()
        content = parts[i]
        label = expected_cats[0] if expected_cats else "object"
        for cat in expected_cats:
            if cat in preceding_text:
                label = cat
                break
        nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
        coords = [float(n) for n in nums]
        if len(coords) == 4:
            results.append({"type": "box", "coords": coords, "label": label})
        elif len(coords) == 2:
            results.append({"type": "point", "coords": coords, "label": label})

    return results


def resize_image_short_side(image, short_side_size):
    w, h = image.size
    if w <= h:
        new_w = short_side_size
        scale_factor = new_w / w
        new_h = int(h * scale_factor)
    else:
        new_h = short_side_size
        scale_factor = new_h / h
        new_w = int(w * scale_factor)
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    return resized_image, scale_factor


def draw_on_frame(frame_bgr, results, draw_label=True):
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_draw = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(20)
    w_img, h_img = pil_img.size

    parsed = []
    for res in results:
        label = res.get("label", "object")
        color = get_color_for_label(label)
        if res.get("type") == "point":
            c = res["coords"]
            cx = max(0, min(w_img, c[0] * w_img / 1000))
            cy = max(0, min(h_img, c[1] * h_img / 1000))
            parsed.append(("point", label, color, cx, cy))
            continue
        if "is_pixel" in res:
            x1, y1, bw, bh = res["coords"]
            x2, y2 = x1 + bw, y1 + bh
        else:
            c = res["coords"]
            if len(c) < 4:
                continue
            x1 = c[0] * w_img / 1000
            y1 = c[1] * h_img / 1000
            x2 = c[2] * w_img / 1000
            y2 = c[3] * h_img / 1000
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        parsed.append(("box", label, color, x1, y1, x2, y2))

    for item in parsed:
        if item[0] == "box":
            _, _, color, x1, y1, x2, y2 = item
            fill_color = color + (65,)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=4)
        elif item[0] == "point":
            _, _, color, cx, cy = item
            r = 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    if draw_label:
        for item in parsed:
            if item[0] == "box":
                _, label, color, x1, y1, x2, y2 = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th = t_box[3] - t_box[1]
                tw = t_box[2] - t_box[0]
                pad_x, pad_y = 7, 4
                tag_h = th + pad_y * 2
                tag_w = tw + pad_x * 2
                tag_y = y1 - tag_h - 2
                if tag_y < 0:
                    tag_y = y2 + 2
                draw.rectangle([x1, tag_y, x1 + tag_w, tag_y + tag_h], fill=color)
                draw.text((x1 + pad_x, tag_y + pad_y), label, fill="white", font=font)
            elif item[0] == "point":
                _, label, color, cx, cy = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th, tw = t_box[3] - t_box[1], t_box[2] - t_box[0]
                tx, ty = cx + 14, cy - th // 2
                draw.rectangle([tx - 2, ty - 2, tx + tw + 6, ty + th + 4], fill=color)
                draw.text((tx + 2, ty), label, fill="white", font=font)

    combined = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


# ============================================================
# 模型
# ============================================================
class EagleWorker:
    def __init__(self, model_path, device="cuda", generation_mode: str = "hybrid"):
        self.model_id = model_path
        self.device = device
        self.dtype = torch.bfloat16
        self.generation_mode = generation_mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            _attn_implementation="sdpa",
            trust_remote_code=True,
            token=HF_TOKEN if HF_TOKEN else None,
        ).to(device).eval()
        print("Model Loaded Successfully!")

    def build_messages(self, image, categories, question_override=None):
        if question_override is not None:
            user_text = question_override
        else:
            category_set_str = "</c>".join(categories)
            user_text = f"Locate all the instances that matches the following description: {category_set_str}."
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]}]

    @torch.no_grad()
    def generate(self, image, categories, generation_mode=None,
                 max_new_tokens=4096, temp=0.7, top_p=0.9, top_k=50,
                 question_override=None):
        messages = self.build_messages(image, categories, question_override=question_override)
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        result = self.model.generate(
            pixel_values=pixel_values, input_ids=input_ids,
            attention_mask=attention_mask, image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer, max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode if generation_mode is not None else self.generation_mode,
            temperature=temp, do_sample=True, top_p=top_p,
            repetition_penalty=1.1, verbose=True,
        )

        token_sequence, out_info, output_text = [], "", ""
        if isinstance(result, tuple) and len(result) >= 3:
            output_text, token_sequence, out_info = result
            if generation_mode == "slow":
                token_sequence[-1] = ("ar", token_sequence[-1][1])
        else:
            output_text = result
        return output_text, token_sequence, out_info


# ============================================================
# 后处理
# ============================================================
def _postprocess_detections(detections, w, h):
    valid = []
    for det in detections:
        if det["type"] == "box":
            c = det["coords"]
            rx1 = max(0, min(w - 1, int(c[0] * w / 1000)))
            ry1 = max(0, min(h - 1, int(c[1] * h / 1000)))
            rx2 = max(0, min(w - 1, int(c[2] * w / 1000)))
            ry2 = max(0, min(h - 1, int(c[3] * h / 1000)))
            box_w, box_h = rx2 - rx1, ry2 - ry1
            if box_w <= 0 or box_h <= 0:
                continue
            valid.append({"type": "box", "coords": [rx1, ry1, box_w, box_h],
                          "is_pixel": True, "label": det["label"]})
        elif det["type"] == "point":
            valid.append(det)
    return valid


def _parse_out_info_dict(out_info: str) -> dict:
    stats = {}
    if not out_info:
        return stats
    cleaned = re.sub(r"^[Ss]tast?ic\s*[Ii]nfo\s*,?\s*", "", out_info.strip())
    for part in cleaned.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            stats[k.strip()] = v.strip()
    return stats


def generate_dynamic_html(token_sequence, out_info, raw_text):
    uid = f"a{int(time.time() * 1000)}"
    css = f"""
    <style>
        .dc-root-{uid} {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            border: 1px solid rgba(118, 185, 0, 0.25); border-radius: 12px;
            background: rgba(0, 0, 0, 0.55); overflow: visible;
        }}
        .dc-header-{uid} {{
            display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
            padding: 10px 14px;
            background: linear-gradient(135deg, rgba(118, 185, 0, 0.25) 0%, rgba(63, 98, 0, 0.35) 100%);
            border-bottom: 1px solid rgba(118, 185, 0, 0.2);
        }}
        .dc-header-title-{uid} {{ font-weight: 700; font-size: 0.82em; color: #d9f99d; letter-spacing: 0.04em; text-transform: uppercase; }}
        .dc-legend-{uid} {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .dc-legend-item-{uid} {{ display: flex; align-items: center; gap: 5px; font-size: 0.72em; color: rgba(226, 232, 240, 0.85); }}
        .dc-legend-dot-{uid} {{ width: 8px; height: 8px; border-radius: 2px; display: inline-block; }}
        .dc-row-{uid} {{ display: flex; gap: 10px; padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .dc-row-{uid}:last-child {{ border-bottom: none; }}
        .dc-val-{uid} {{
            flex: 1; line-height: 1.9; word-wrap: break-word; color: #cbd5e1; font-size: 0.85em;
            display: flex; flex-wrap: wrap; gap: 4px; align-items: center; align-content: flex-start;
        }}
        @keyframes tk-{uid} {{
            0%   {{ opacity: 0; transform: translateY(8px) scale(0.92); }}
            60%  {{ opacity: 1; transform: translateY(-2px) scale(1.02); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        .tk-mtp-{uid}, .tk-ar-{uid} {{
            opacity: 0; animation: tk-{uid} 0.35s ease-out forwards;
            border-radius: 5px; padding: 2px 7px; margin: 0;
            display: inline-block; max-width: 100%;
            font-size: 0.78em; font-weight: 600;
            font-family: 'Fira Code', Consolas, monospace;
            white-space: normal; word-break: break-all;
        }}
        .tk-mtp-{uid} {{ background: rgba(118, 185, 0, 0.15); border: 1px solid rgba(118, 185, 0, 0.55); color: #bbf7d0; }}
        .tk-ar-{uid} {{ background: rgba(230, 81, 0, 0.15); border: 1px solid rgba(230, 81, 0, 0.55); color: #fed7aa; }}
        .tk-stat-{uid} {{
            opacity: 0; animation: tk-{uid} 0.4s ease-out forwards;
            background: rgba(118, 185, 0, 0.12); border: 1px solid rgba(118, 185, 0, 0.35); border-radius: 6px;
            padding: 4px 12px; display: inline-block; font-size: 0.78em; color: #d9f99d; font-weight: 600;
        }}
        .dc-raw-{uid} {{ padding: 0 14px 12px; }}
        .dc-raw-{uid} summary {{ cursor: pointer; color: #94a3b8; font-size: 0.78em; user-select: none; }}
        .dc-raw-{uid} summary:hover {{ color: #76b900; }}
        .dc-raw-pre-{uid} {{
            background: rgba(0,0,0,0.45); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px;
            padding: 10px; margin-top: 8px;
            font-family: 'Fira Code', Consolas, monospace;
            font-size: 0.74em; color: #cbd5e1; white-space: pre-wrap; word-break: break-all;
        }}
    </style>
    """
    h = css + f'<div class="dc-root-{uid}">'
    h += (f'<div class="dc-header-{uid}">'
          f'<span class="dc-header-title-{uid}">Decoding Trace</span>'
          f'<div class="dc-legend-{uid}">'
          f'<div class="dc-legend-item-{uid}"><span class="dc-legend-dot-{uid}" style="background:#76b900;"></span>MTP Parallel</div>'
          f'<div class="dc-legend-item-{uid}"><span class="dc-legend-dot-{uid}" style="background:#e65100;"></span>AR Fallback</div>'
          f'</div></div>')
    tok_idx = 0
    if out_info:
        stats = _parse_out_info_dict(out_info)
        bits = []
        if "forward_step" in stats:
            bits.append(f"{stats['forward_step']} steps")
        if "num_tokens" in stats:
            bits.append(f"{stats['num_tokens']} tokens")
        if "num_boxes" in stats:
            bits.append(f"{stats['num_boxes']} boxes")
        if "switch_to_ar" in stats:
            n = stats["switch_to_ar"]
            bits.append(f"{n} AR fallback{'s' if n != '1' else ''}")
        if "tps" in stats:
            bits.append(f"{stats['tps']} tok/s")
        if "bps" in stats:
            bits.append(f"{stats['bps']} box/s")
        summary = " · ".join(bits) if bits else out_info.strip()
        h += (f'<div class="dc-row-{uid}" style="justify-content:flex-start;padding-top:8px;padding-bottom:4px;border-bottom:none;">'
              f'<span class="tk-stat-{uid}" style="animation-delay:0s">{summary}</span></div>')
    h += f'<div class="dc-row-{uid}"><div class="dc-val-{uid}">'
    if token_sequence:
        for item in token_sequence:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            decode_type = str(item[0]).lower()
            text = str(item[1])
            safe = text.replace("<", "&lt;").replace(">", "&gt;")
            delay = f"{tok_idx * 0.06:.2f}s"
            cls = f"tk-ar-{uid}" if decode_type == "ar" else f"tk-mtp-{uid}"
            h += f'<span class="{cls}" style="animation-delay:{delay}">{safe}</span> '
            tok_idx += 1
    h += '</div></div>'
    if raw_text:
        safe_raw = raw_text.replace("<", "&lt;").replace(">", "&gt;")
        h += (f'<div class="dc-raw-{uid}"><details open><summary>Raw Response</summary>'
              f'<div class="dc-raw-pre-{uid}">{safe_raw}</div></details></div>')
    h += '</div>'
    return h


def generate_raw_prompt(task_type, category):
    if not category:
        category = "objects"
    cats = "</c>".join(c.strip() for c in category.split(",") if c.strip())
    if task_type == "Detection":
        return f"Locate all the instances that matches the following description: {cats}."
    elif task_type == "Grounding":
        return f"Locate all the instances that match the following description: {cats}."
    elif task_type == "OCR":
        return "Detect all the text in box format."
    elif task_type == "GUI":
        return f"Locate the region that matches the following description: {cats}."
    elif task_type == "Pointing":
        return f"Point to: {cats}."
    else:
        return f"Locate all the instances that matches the following description: {cats}."


# ============================================================
# 模型初始化
# ============================================================
GLOBAL_WORKER = None

def get_worker():
    global GLOBAL_WORKER
    if GLOBAL_WORKER is None:
        try:
            MODEL_PATH = os.environ.get("MODEL_PATH", "nvidia/LocateAnything-3B")
            print(f"Loading model inside @spaces.GPU context: {MODEL_PATH}")
            GLOBAL_WORKER = EagleWorker(MODEL_PATH)
        except Exception as e:
            print(f"Failed to load model: {e}. Will run in Mock Mode.")
            GLOBAL_WORKER = None
    return GLOBAL_WORKER


def _prepare_image_for_model(pil_img, short_size):
    process_img = pil_img.copy()
    if short_size is not None and short_size > 0:
        process_img, _ = resize_image_short_side(process_img, min(int(short_size), 1024))
    else:
        if min(process_img.size) > 1024:
            process_img, _ = resize_image_short_side(process_img, 1024)
    return process_img


# ============================================================
# 用户数据收集（HuggingFace Public Dataset）
# 策略：one-record-per-file，配合按日目录 + 容器级 SESSION_ID
# 每条记录：data/<YYYY-MM-DD>/<session_id>__<entry_id>.jsonl
# CommitScheduler 只会新增文件，不会覆盖其它 session 的数据
# ============================================================
LOG_DATASET_REPO = os.environ.get("LOG_DATASET_REPO", "woshichaoren123/log")
_LOG_DIR = Path(tempfile.mkdtemp(prefix="hf_log_"))
_SESSION_ID = uuid.uuid4().hex[:8]
_log_scheduler = None

if LOG_DATASET_REPO and LOG_HF_TOKEN:
    try:
        _log_scheduler = CommitScheduler(
            repo_id=LOG_DATASET_REPO,
            repo_type="dataset",
            folder_path=str(_LOG_DIR),
            path_in_repo="data",
            every=3,
            token=LOG_HF_TOKEN,
            squash_history=False,
        )
        print(f"[LOG] Dataset logging enabled -> {LOG_DATASET_REPO} "
              f"(session={_SESSION_ID}, dir={_LOG_DIR})")
    except Exception as e:
        _log_scheduler = None
        print(f"[LOG] Dataset logging disabled: {e}")
else:
    print("[LOG] Dataset logging disabled (LOG_HF_TOKEN not set)")


def _pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _atomic_write_text(path: Path, text: str):
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _log_to_dataset(
    input_type, category, model_mode, raw_prompt,
    output_text="", input_image=None, output_image=None,
    extra=None,
):
    if _log_scheduler is None:
        return
    try:
        entry_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        date_str = time.strftime("%Y-%m-%d", time.gmtime())

        input_b64 = None
        if input_image is not None and isinstance(input_image, Image.Image):
            input_b64 = _pil_to_b64(input_image)

        output_b64 = None
        if output_image is not None and isinstance(output_image, Image.Image):
            output_b64 = _pil_to_b64(output_image)

        record = {
            "id": entry_id,
            "session_id": _SESSION_ID,
            "timestamp": ts,
            "input_type": input_type,
            "category": category,
            "model_mode": model_mode,
            "raw_prompt": raw_prompt,
            "output_text": output_text,
            "input_image_b64": input_b64,
            "output_image_b64": output_b64,
        }
        if extra:
            record.update(extra)

        day_dir = _LOG_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        log_file = day_dir / f"{_SESSION_ID}__{entry_id}.jsonl"

        payload = json.dumps(record, ensure_ascii=False) + "\n"
        with _log_scheduler.lock:
            _atomic_write_text(log_file, payload)
    except Exception as e:
        print(f"[LOG] Failed to log to dataset: {e}")


def _maybe_log_inference(
    input_type: str,
    category: str,
    model_mode: str,
    raw_prompt: str,
    output_text: str,
    input_path: str | None = None,
    output_path: str | None = None,
    extra: dict | None = None,
):
    try:
        input_image = None
        output_image = None

        if input_path and os.path.exists(input_path):
            if input_type == "image":
                input_image = Image.open(input_path).convert("RGB")
            elif input_type == "video":
                cap = cv2.VideoCapture(input_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if output_path and os.path.exists(output_path) and input_type == "image":
            output_image = Image.open(output_path).convert("RGB")

        categories_list = [c.strip() for c in category.split(",") if c.strip()]
        _log_to_dataset(
            input_type=input_type,
            category=", ".join(categories_list) if categories_list else category,
            model_mode=model_mode,
            raw_prompt=raw_prompt,
            output_text=output_text,
            input_image=input_image,
            output_image=output_image,
            extra=extra,
        )
    except Exception as e:
        print(f"[LOG] Failed to prepare log record: {e}")


# ============================================================
# GPU 时间预算与推理保护（按模式区分）
# ============================================================
GPU_HARD_LIMIT_IMAGE = 30
GPU_HARD_LIMIT_VIDEO = 240
PHASE2_RESERVE = 55
SAFETY_MARGIN = 25
EST_SECONDS_PER_FRAME = 20


@spaces.GPU(duration=120, size="xlarge")
def run_image_gpu_api(
    image_path: str, category: str, model_mode: str, temp: float, top_p: float, top_k: int,
    short_size: int | None, question_override: str | None
):
    image_in = Image.open(image_path).convert("RGB")
    categories_list = [c.strip() for c in category.split(",") if c.strip()]
    category_str = "</c>".join(categories_list)

    process_img = _prepare_image_for_model(image_in, short_size)

    worker = get_worker()
    if worker:
        output_text, token_sequence, out_info = worker.generate(
            process_img, categories_list, model_mode,
            temp=temp, top_p=top_p, top_k=top_k,
            question_override=question_override,
        )
    else:
        # Mock mode fallback
        output_text = "Mock detection: <ref>sushi</ref><box><240><480><620><940></box> and <ref>book</ref><box><50><120><400><380></box>"
        token_sequence = []
        out_info = "forward_step=1;num_tokens=18;num_boxes=2;tps=45;bps=15"

    detections = parse_mixed_results(output_text, category_str)
    frame_bgr = cv2.cvtColor(np.array(image_in), cv2.COLOR_RGB2BGR)
    out_img_bgr = draw_on_frame(frame_bgr, detections, draw_label=True)
    output_image = Image.fromarray(cv2.cvtColor(out_img_bgr, cv2.COLOR_BGR2RGB))

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    out_img_path = os.path.join(temp_dir, "output.png")
    output_image.save(out_img_path)

    stats = _parse_out_info_dict(out_info)

    # Simplified summary lists
    detections_summary = []
    for det in detections:
        detections_summary.append({
            "label": det.get("label", "object"),
            "type": det.get("type", "box"),
            "coords": [round(c, 2) for c in det.get("coords", [])]
        })

    html = generate_dynamic_html(token_sequence, out_info, output_text)
    return out_img_path, stats, output_text, detections_summary, html


@spaces.GPU(duration=240, size="xlarge")
def run_video_gpu_api(
    video_path: str, category: str, model_mode: str, temp: float, top_p: float, top_k: int,
    short_size: int | None, question_override: str | None, max_video_frames: int
):
    import subprocess as _sp

    total_start = time.time()
    max_frames = int(max_video_frames) if max_video_frames else 4

    categories_list = [c.strip() for c in category.split(",") if c.strip()]
    category_str = "</c>".join(categories_list)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    total = len(all_frames)

    if total == 0:
        raise ValueError("Failed to read any frames from the video.")

    # Sample frames
    if total <= max_frames:
        sample_indices = list(range(total))
    else:
        sample_indices = [int(round(i * (total - 1) / (max_frames - 1))) for i in range(max_frames)]

    sampled_frames = [all_frames[i] for i in sample_indices]
    n_sampled = len(sampled_frames)

    # Budget check
    time_already_used = time.time() - total_start
    available_for_inference = GPU_HARD_LIMIT_VIDEO - time_already_used - PHASE2_RESERVE - SAFETY_MARGIN
    estimated_inference_time = n_sampled * EST_SECONDS_PER_FRAME

    if estimated_inference_time > available_for_inference:
        max_feasible = max(1, int(available_for_inference // EST_SECONDS_PER_FRAME))
        if total <= max_feasible:
            sample_indices = list(range(total))
        else:
            sample_indices = [int(round(i * (total - 1) / (max_feasible - 1))) for i in range(max_feasible)]
        sampled_frames = [all_frames[i] for i in sample_indices]
        n_sampled = len(sampled_frames)

    out_fps = max(1.0, n_sampled / (total / fps)) if fps > 0 else 5.0
    del all_frames
    gc.collect()

    inference_results = []
    processed_count = 0
    early_stopped = False
    early_stop_reason = ""

    for i, frame in enumerate(sampled_frames):
        elapsed_since_start = time.time() - total_start
        remaining_total = GPU_HARD_LIMIT_VIDEO - elapsed_since_start

        if remaining_total < PHASE2_RESERVE + SAFETY_MARGIN:
            early_stopped = True
            early_stop_reason = f"GPU time budget running out. Only {remaining_total:.0f}s left."
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_img = _prepare_image_for_model(pil_img, short_size)

        worker = get_worker()
        if worker:
            output_text, _, _ = worker.generate(
                process_img, categories_list, model_mode,
                temp=temp, top_p=top_p, top_k=top_k,
                question_override=question_override,
            )
        else:
            output_text = f"Mock video detection: <ref>person</ref><box><100><150><800><900></box>"

        inference_results.append(output_text)
        processed_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if processed_count == 0:
        raise RuntimeError("GPU budget exceeded before processing any frames.")

    sampled_frames_for_draw = sampled_frames[:processed_count]
    inference_results_for_draw = inference_results[:processed_count]

    tmp_raw = tempfile.mktemp(suffix=".raw.mp4")
    out_video_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (vid_w, vid_h))

    detections_summary = []
    for i, (frame, output_text) in enumerate(zip(sampled_frames_for_draw, inference_results_for_draw)):
        detections = parse_mixed_results(output_text, category_str)
        valid_results = _postprocess_detections(detections, vid_w, vid_h)
        frame_to_draw = draw_on_frame(frame, valid_results, draw_label=True)
        out.write(frame_to_draw)

        for det in valid_results:
            detections_summary.append({
                "frame": i + 1,
                "label": det.get("label", "object"),
                "type": det.get("type", "box"),
                "coords": det.get("coords", [])
            })

    out.release()

    # ffmpeg re-encode
    elapsed_now = time.time() - total_start
    remaining_now = GPU_HARD_LIMIT_VIDEO - elapsed_now

    if remaining_now > 15:
        try:
            ffmpeg_timeout = max(10, int(remaining_now - 5))
            _sp.run(
                ["ffmpeg", "-y", "-i", tmp_raw, "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out_video_path],
                check=True, capture_output=True, timeout=ffmpeg_timeout,
            )
            os.remove(tmp_raw)
        except Exception:
            if os.path.exists(tmp_raw):
                os.replace(tmp_raw, out_video_path)
    else:
        os.replace(tmp_raw, out_video_path)

    total_time = time.time() - total_start
    stats = {
        "total_frames": total,
        "sampled_frames": n_sampled,
        "processed_frames": processed_count,
        "total_time_seconds": round(total_time, 2),
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason
    }

    raw_combined = "\n---\n".join(inference_results_for_draw)
    timing_summary = (
        f"Processed {processed_count}/{n_sampled} sampled frames "
        f"({total} total) in {total_time:.1f}s"
    )
    if early_stopped:
        timing_summary += f" — {early_stop_reason}"
    html = generate_dynamic_html([], "", timing_summary + "\n\n" + raw_combined)
    return out_video_path, stats, raw_combined, detections_summary, html


# ============================================================
# GRADIO SERVER APP
# ============================================================
app = Server()

# Serve static assets folder
assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1 style='color: #ef4444; font-family: Inter, sans-serif; text-align: center; margin-top: 100px;'>index.html is missing</h1>")


@app.api(name="run_inference")
def run_inference_api(
    input_type: str,
    image_file: Any = None,
    video_file: Any = None,
    task_type: str = "Detection",
    category: str = "objects",
    model_mode: str = "hybrid",
    temp: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    short_size: int | None = None,
    question_override: str | None = None,
    max_video_frames: int = 4
) -> tuple[FileData | None, FileData | None, dict]:
    """Exposed Gradio Queueing Endpoint for custom frontend interactions.
    
    ZeroGPU allocation is triggered directly at this endpoint boundary.
    Supports both FileData dict (from web uploads) and local strings (for examples).
    """
    try:
        if not category:
            category = "objects"
        
        final_prompt = question_override
        if not final_prompt:
            final_prompt = generate_raw_prompt(task_type, category)

        if input_type == "Image":
            if not image_file:
                return None, None, {"success": False, "error": "Please upload an image."}
            
            # Resolve image path (from either FileData upload or local example string)
            if isinstance(image_file, str):
                img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_file)
            elif isinstance(image_file, dict):
                img_path = image_file.get("path")
            else:
                img_path = getattr(image_file, "path", None)

            if not img_path or not os.path.exists(img_path):
                return None, None, {"success": False, "error": f"Invalid image file path: {img_path}"}

            out_img_path, stats, raw_text, detections, html = run_image_gpu_api(
                img_path, category, model_mode, temp, top_p, top_k, short_size, final_prompt
            )
            
            meta = {
                "success": True,
                "input_type": "Image",
                "stats": stats,
                "raw_text": raw_text,
                "detections": detections,
                "final_prompt": final_prompt,
                "html": html,
            }
            _maybe_log_inference(
                input_type="image",
                category=category,
                model_mode=model_mode,
                raw_prompt=final_prompt,
                output_text=raw_text,
                input_path=img_path,
                output_path=out_img_path,
                extra={"task_type": task_type, "detections": detections, "stats": stats},
            )
            return FileData(path=out_img_path), None, meta

        else:
            if not video_file:
                return None, None, {"success": False, "error": "Please upload a video."}

            # Resolve video path (from either FileData upload or local example string)
            if isinstance(video_file, str):
                vid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_file)
            elif isinstance(video_file, dict):
                vid_path = video_file.get("path")
            else:
                vid_path = getattr(video_file, "path", None)

            if not vid_path or not os.path.exists(vid_path):
                return None, None, {"success": False, "error": f"Invalid video file path: {vid_path}"}

            out_vid_path, stats, raw_text, detections, html = run_video_gpu_api(
                vid_path, category, model_mode, temp, top_p, top_k, short_size, final_prompt, max_video_frames
            )

            meta = {
                "success": True,
                "input_type": "Video",
                "stats": stats,
                "raw_text": raw_text,
                "detections": detections,
                "final_prompt": final_prompt,
                "html": html,
            }
            _maybe_log_inference(
                input_type="video",
                category=category,
                model_mode=model_mode,
                raw_prompt=final_prompt,
                output_text=raw_text,
                input_path=vid_path,
                extra={
                    "task_type": task_type,
                    "detections": detections,
                    "stats": stats,
                    "video_total_frames": stats.get("total_frames"),
                    "video_sampled_frames": stats.get("sampled_frames"),
                    "video_processed_frames": stats.get("processed_frames"),
                },
            )
            return None, FileData(path=out_vid_path), meta

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, {"success": False, "error": str(e)}


if __name__ == "__main__":
    app.launch(show_error=True)