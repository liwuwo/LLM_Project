#!/bin/bash
# ============================================
# AutoDL 一键部署启动脚本
# 用法: bash deploy.sh
# ============================================

set -e

# ── 配置区（按需修改）──────────────────────
PROJECT_DIR="/root/autodl-tmp/LLM_Project"
MODEL_PATH="/root/autodl-tmp/models/cyankiwi/Qwen3___6-27B-AWQ-INT4"
VLLM_LOG="vllm_server.log"
USE_LOCAL_LLM=false   # 改为 true 使用本地模型
# ─────────────────────────────────────────────

cd "$PROJECT_DIR"

echo "=========================================="
echo "  LLM_Project AutoDL 部署"
echo "=========================================="

# 1. 安装 Python 依赖
echo "[1/4] 安装 Python 依赖..."
pip install -r requirements.txt -q

# 2. 启动 MySQL（如果未运行）
echo "[2/4] 检查 MySQL 服务..."
if ! pgrep -x mysqld > /dev/null; then
    service mysql start
    echo "  MySQL 已启动"
else
    echo "  MySQL 已在运行"
fi

# 3. 启动 vLLM 服务（仅本地模型模式需要）
if [ "$USE_LOCAL_LLM" = "true" ]; then
    echo "[3/4] 启动 vLLM 服务..."
    export OMP_NUM_THREADS=4
    export VLLM_USE_FLASHINFER_SAMPLER=0

    if pgrep -f "vllm serve" > /dev/null; then
        echo "  vLLM 已在运行，跳过启动"
    else
        nohup vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 1 \
            --trust-remote-code \
            --quantization compressed-tensors \
            --dtype auto \
            --gpu-memory-utilization 0.9 \
            --max-model-len 32768 \
            --max-num-seqs 8 \
            --compilation-config '{"cudagraph_mode": "none"}' \
            > "$VLLM_LOG" 2>&1 &
        echo "  vLLM 已启动，PID: $!"
        echo "  等待模型加载（约60秒）..."
        sleep 60
    fi
else
    echo "[3/4] 使用 DeepSeek 云端 API，跳过 vLLM 启动"
fi

# 4. 启动 Gradio UI
echo "[4/4] 启动 Gradio Web UI..."
echo ""
echo "=========================================="
echo "  启动完成！"
echo "  Web UI 地址: http://0.0.0.0:7860"
echo "  日志文件: $VLLM_LOG"
echo "=========================================="
echo ""

python ui.py
