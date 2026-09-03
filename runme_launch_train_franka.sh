#!/usr/bin/env bash
# 后台开训（nohup 版）: bash /data1/Franka_RealRobot/openpi/runme_launch_train_franka.sh
# 训练本体 = runme_finetune_franka.sh（8 卡, bs32, 20k steps, ckpt 落 /data1）。
set -euo pipefail
cd /data1/Franka_RealRobot/openpi
mkdir -p logs

if pgrep -f "scripts/train.py pi05_droid_franka_lora" >/dev/null; then
  echo "已有同名训练在跑，先 pkill -f 'scripts/train.py pi05_droid_franka_lora' 再重开"
  exit 1
fi

LOG="logs/train_$(date +%m%d_%H%M).log"
nohup bash runme_finetune_franka.sh > "$LOG" 2>&1 &
echo "已开训 (PID $!)  log: /data1/Franka_RealRobot/openpi/$LOG"
echo "盯进度: tail -f /data1/Franka_RealRobot/openpi/$LOG"
