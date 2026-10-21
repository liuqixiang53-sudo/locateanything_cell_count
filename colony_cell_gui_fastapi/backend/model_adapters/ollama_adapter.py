import requests
import json
import re
import time
import base64
import io
from typing import Optional
from PIL import Image


class OllamaVLAdapter:
    """Ollama 视觉语言模型适配器，支持 Qwen2.5-VL / Qwen3-VL"""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.chat_url = f"{self.host}/api/chat"

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image, prompt: str, temperature: float = 0.1,
                max_tokens: Optional[int] = None) -> dict:
        """
        调用 Ollama API 进行视觉推理。

        Returns:
            dict with keys: "answer", "elapsed"
        """
        image_b64 = self._image_to_base64(image)

        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": options,
        }

        start = time.time()
        resp = requests.post(self.chat_url, json=payload, timeout=600)
        elapsed = time.time() - start

        if resp.status_code != 200:
            return {"answer": "", "elapsed": elapsed, "error": resp.text}

        data = resp.json()
        message = data.get("message", {})
        answer = message.get("content", "")

        if not answer:
            thinking = message.get("thinking", "")
            nums = re.findall(r"\d+", thinking)
            if nums:
                answer = nums[-1]

        return {"answer": answer, "elapsed": elapsed}

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """解析模型输出中的坐标，支持多种格式"""

        # 格式1: <box><x1><y1><x2><y2></box>  (归一化 0-1000)
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

        # 格式2: [x1, y1, x2, y2] 或 (x1, y1, x2, y2)  (像素坐标)
        for m in re.finditer(r"[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if boxes:
            return boxes

        # 格式3: QwenVL 风格 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
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

        # 格式4: 纯文本坐标描述 "(x1, y1)-(x2, y2)" (像素坐标)
        for m in re.finditer(r"\((\d+),\s*(\d+)\)-\((\d+),\s*(\d+)\)", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        return boxes


def check_ollama_available(host: str = "http://localhost:11434") -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """获取已安装的 Ollama 模型列表"""
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []