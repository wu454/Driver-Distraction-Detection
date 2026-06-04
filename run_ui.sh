#!/usr/bin/env bash
# 启动 Tier-1 可视化界面
cd "$(dirname "$0")"
pip install -q gradio 2>/dev/null || true
python driver_distraction_ui.py --device "${1:-auto}" --port "${2:-7860}"
