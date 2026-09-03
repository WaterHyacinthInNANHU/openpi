#!/usr/bin/env bash
# 看训练进度: bash /data1/Franka_RealRobot/openpi/runme_watch_train_franka.sh
# Ctrl-C 退出查看，不影响训练本体。
cd /data1/Franka_RealRobot/openpi/logs 2>/dev/null || { echo "还没开过训（无 logs 目录）"; exit 1; }
L=$(ls -t train_*.log 2>/dev/null | head -1)
[ -z "${L:-}" ] && { echo "还没有训练 log"; exit 1; }
N=$(pgrep -cf "scripts/train.py pi05_droid_franka_lora" || true)
echo "== log: $L | 训练进程数: ${N:-0} =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "== 实时输出 (Ctrl-C 退出) =="
tail -n 30 -f "$L"
