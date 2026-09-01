#!/bin/bash
# 菌落与细胞计数分析软件 V1.0 - FastAPI 启动脚本
# 使用方法: ./run_fastapi.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"
VENV_DIR="$(dirname "${SCRIPT_DIR}")/locateanthing"

# 设置 CUDA 显存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo " 菌落与细胞计数分析软件 V1.0 (FastAPI)"
echo "============================================"
echo ""
echo "启动后端服务..."
echo "访问地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo ""
echo "CUDA 显存分配策略: expandable_segments:True"
echo ""

cd "${BACKEND_DIR}"
source "${VENV_DIR}/bin/activate"
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
