#!/bin/bash
set -e

WORK_DIR="/home/chenyifu/YYZ/memory"
OMNI_PORT=8000
ASR_PORT=8001

install_deps() {
    pip install -U pip
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install fastapi uvicorn pydantic Pillow soundfile
    pip uninstall -y transformers
    pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
    pip install accelerate
    pip install qwen-omni-utils[decord] -U
    pip install -U qwen-asr
    pip install -U flash-attn --no-build-isolation
}

start_omni() {
    cd "$WORK_DIR"
    nohup python omni_server.py > logs/omni_server.log 2>&1 &
    echo "omni PID: $!"
    for i in $(seq 1 120); do
        if curl -s http://localhost:$OMNI_PORT/health | grep -q "ok"; then
            echo "omni ready (:${OMNI_PORT})"
            return 0
        fi
        sleep 5
    done
    echo "omni timeout, check logs/omni_server.log"
}

start_asr() {
    cd "$WORK_DIR"
    nohup python asr_sever.py > logs/asr_server.log 2>&1 &
    echo "asr PID: $!"
    for i in $(seq 1 60); do
        if curl -s http://localhost:$ASR_PORT/health | grep -q "ok"; then
            echo "asr ready (:${ASR_PORT})"
            return 0
        fi
        sleep 3
    done
    echo "asr timeout, check logs/asr_server.log"
}

start_memory() {
    cd "$WORK_DIR"
    nohup python live_adapter.py \
        --host 0.0.0.0 --port 8070 \
        --main-api-base http://localhost:$OMNI_PORT/v1 \
        --main-model qwen2.5-omni-7b \
        --summarizer-api-base http://localhost:$OMNI_PORT/v1 \
        --summarizer-model qwen2.5-omni-7b \
        --longterm-api-base http://localhost:$OMNI_PORT/v1 \
        --longterm-model qwen2.5-omni-7b \
        > logs/live_adapter.log 2>&1 &
    echo "memory PID: $!"
}

check_status() {
    curl -s http://localhost:$OMNI_PORT/health 2>/dev/null || echo "omni: down"
    curl -s http://localhost:$ASR_PORT/health 2>/dev/null || echo "asr: down"
}

stop_all() {
    pkill -f "omni_server.py" 2>/dev/null || true
    pkill -f "asr_sever.py" 2>/dev/null || true
    pkill -f "live_adapter.py" 2>/dev/null || true
    echo "stopped"
}

main() {
    mkdir -p "$WORK_DIR/logs" "$WORK_DIR/result"
    case "${1:-start}" in
        deps)    install_deps ;;
        start)   start_omni; start_asr; check_status ;;
        omni)    start_omni ;;
        asr)     start_asr ;;
        memory)  start_memory ;;
        status)  check_status ;;
        stop)    stop_all ;;
        all)     install_deps; start_omni; start_asr; start_memory; check_status ;;
        *)       echo "usage: bash deploy.sh {deps|start|omni|asr|memory|status|stop|all}" ;;
    esac
}

main "$@"
