#!/bin/bash
# ============================================================
# Qwen2.5-Omni-7B + Qwen3-ASR-1.7B 本地部署脚本
# 服务器工作目录: /home/chenyifu/YYZ
# ============================================================
set -e

WORK_DIR="/home/chenyifu/YYZ"
OMNI_PORT=8000
ASR_PORT=8001

# ---------- Step 1: 下载模型 ----------
download_models() {
    echo "===== Step 1: 下载模型 ====="

    # Qwen3-ASR-1.7B（约 3-4 GB）
    if [ ! -d "$WORK_DIR/Qwen3-ASR-1.7B" ]; then
        echo "下载 Qwen3-ASR-1.7B ..."
        huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir "$WORK_DIR/Qwen3-ASR-1.7B"
    else
        echo "Qwen3-ASR-1.7B 已存在，跳过下载"
    fi

    # Qwen2.5-Omni-7B（约 15 GB，需要较大磁盘空间）
    if [ ! -d "$WORK_DIR/Qwen2.5-Omni-7B" ]; then
        echo "下载 Qwen2.5-Omni-7B ..."
        huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir "$WORK_DIR/Qwen2.5-Omni-7B"
    else
        echo "Qwen2.5-Omni-7B 已存在，跳过下载"
    fi

    echo "模型下载完成"
}

# 国内用户可用 ModelScope 加速下载
download_models_modelscope() {
    echo "===== Step 1 (ModelScope): 下载模型 ====="

    pip install -U modelscope

    if [ ! -d "$WORK_DIR/Qwen3-ASR-1.7B" ]; then
        echo "从 ModelScope 下载 Qwen3-ASR-1.7B ..."
        modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir "$WORK_DIR/Qwen3-ASR-1.7B"
    fi

    if [ ! -d "$WORK_DIR/Qwen2.5-Omni-7B" ]; then
        echo "从 ModelScope 下载 Qwen2.5-Omni-7B ..."
        modelscope download --model Qwen/Qwen2.5-Omni-7B --local_dir "$WORK_DIR/Qwen2.5-Omni-7B"
    fi

    echo "模型下载完成"
}

# ---------- Step 2: 安装依赖 ----------
install_deps() {
    echo "===== Step 2: 安装 Python 依赖 ====="

    pip install -U pip

    # 基础依赖
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install fastapi uvicorn pydantic Pillow soundfile

    # Qwen2.5-Omni 所需依赖
    pip uninstall -y transformers
    pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
    pip install accelerate
    pip install qwen-omni-utils[decord] -U

    # Qwen3-ASR 所需依赖（transformers 后端，不用 vLLM）
    pip install -U qwen-asr

    # Flash Attention 2（大幅降低显存占用）
    pip install -U flash-attn --no-build-isolation

    echo "依赖安装完成"
}

# ---------- Step 3: 启动 Omni 服务 (端口 8000) ----------
start_omni() {
    echo "===== Step 3: 启动 Qwen2.5-Omni-7B 服务 (端口 $OMNI_PORT) ====="

    cd "$WORK_DIR"
    nohup python omni_server.py > logs/omni_server.log 2>&1 &
    OMNI_PID=$!
    echo "Qwen2.5-Omni-7B PID: $OMNI_PID"

    # 等待服务就绪
    echo "等待 Omni 服务就绪..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$OMNI_PORT/health | grep -q "ok"; then
            echo "Omni 服务已就绪 (端口 $OMNI_PORT)"
            return 0
        fi
        sleep 5
    done
    echo "警告: Omni 服务启动超时，请检查 logs/omni_server.log"
}

# ---------- Step 4: 启动 ASR 服务 (端口 8001) ----------
start_asr() {
    echo "===== Step 4: 启动 Qwen3-ASR-1.7B 服务 (端口 $ASR_PORT) ====="

    cd "$WORK_DIR"
    nohup python asr_sever.py > logs/asr_server.log 2>&1 &
    ASR_PID=$!
    echo "Qwen3-ASR-1.7B PID: $ASR_PID"

    # 等待服务就绪
    echo "等待 ASR 服务就绪..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:$ASR_PORT/health | grep -q "ok"; then
            echo "ASR 服务已就绪 (端口 $ASR_PORT)"
            return 0
        fi
        sleep 3
    done
    echo "警告: ASR 服务启动超时，请检查 logs/asr_server.log"
}

# ---------- Step 5: 启动 Memory 项目 ----------
start_memory() {
    echo "===== Step 5: 启动 Memory 项目 (live_adapter) ====="

    cd "$WORK_DIR"
    nohup python live_adapter.py \
        --host 0.0.0.0 \
        --port 8070 \
        --main-api-base http://localhost:$OMNI_PORT/v1 \
        --main-model qwen2.5-omni-7b \
        --summarizer-api-base http://localhost:$OMNI_PORT/v1 \
        --summarizer-model qwen2.5-omni-7b \
        --longterm-api-base http://localhost:$OMNI_PORT/v1 \
        --longterm-model qwen2.5-omni-7b \
        > logs/live_adapter.log 2>&1 &
    MEM_PID=$!
    echo "Memory 项目 PID: $MEM_PID"
    echo "Memory 项目已启动 (端口 8070)"
}

# ---------- 状态检查 ----------
check_status() {
    echo ""
    echo "========== 服务状态 =========="
    echo "Omni (8000): $(curl -s http://localhost:$OMNI_PORT/health 2>/dev/null || echo '未启动')"
    echo "ASR  (8001): $(curl -s http://localhost:$ASR_PORT/health 2>/dev/null || echo '未启动')"
    echo "=============================="
}

# ---------- 停止服务 ----------
stop_all() {
    echo "正在停止所有服务..."
    pkill -f "omni_server.py" 2>/dev/null || true
    pkill -f "asr_sever.py" 2>/dev/null || true
    pkill -f "live_adapter.py" 2>/dev/null || true
    echo "所有服务已停止"
}

# ---------- 主流程 ----------
main() {
    mkdir -p "$WORK_DIR/logs"
    mkdir -p "$WORK_DIR/result"

    echo "=========================================="
    echo "  Qwen 本地模型部署脚本"
    echo "  工作目录: $WORK_DIR"
    echo "  Omni 端口: $OMNI_PORT"
    echo "  ASR  端口: $ASR_PORT"
    echo "=========================================="
    echo ""

    case "${1:-all}" in
        download)
            download_models
            ;;
        download-cn)
            download_models_modelscope
            ;;
        deps)
            install_deps
            ;;
        start)
            start_omni
            start_asr
            check_status
            ;;
        start-omni)
            start_omni
            ;;
        start-asr)
            start_asr
            ;;
        start-memory)
            start_memory
            check_status
            ;;
        status)
            check_status
            ;;
        stop)
            stop_all
            ;;
        all)
            download_models
            install_deps
            start_omni
            start_asr
            start_memory
            check_status
            ;;
        *)
            echo "用法: bash deploy.sh [command]"
            echo ""
            echo "命令:"
            echo "  all          完整部署（下载+安装+启动）"
            echo "  download     下载模型（HuggingFace）"
            echo "  download-cn  下载模型（ModelScope，国内推荐）"
            echo "  deps         安装 Python 依赖"
            echo "  start        启动 Omni + ASR 服务"
            echo "  start-omni   仅启动 Omni 服务"
            echo "  start-asr    仅启动 ASR 服务"
            echo "  start-memory 启动 Memory 项目 (live_adapter)"
            echo "  status       查看服务状态"
            echo "  stop         停止所有服务"
            echo ""
            echo "示例:"
            echo "  bash deploy.sh download-cn   # 先下载模型"
            echo "  bash deploy.sh deps          # 安装依赖"
            echo "  bash deploy.sh start         # 启动服务"
            ;;
    esac
}

main "$@"
