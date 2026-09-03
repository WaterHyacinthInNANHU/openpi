#!/usr/bin/env bash
# 跟着看 10task_v2 训练的进度条 (step / 速度 / 剩余时间) 和 loss 行。Ctrl-C 退出, 不影响训练。
#   bash watch_train_10task_v2.sh
# 进度条走 logging (实时); "Step N: loss=..." 行走块缓冲, 会滞后几千步才落盘, 想实时看 loss 用 watch_loss_10task.sh 那套读 wandb。
LOG=$(ls -t /data1/Franka_RealRobot/openpi/logs/train_10task_v2_*.log 2>/dev/null | head -1)
[ -z "${LOG:-}" ] && { echo "没找到 log"; exit 1; }
PID=$(pgrep -f "^/data1/Franka_RealRobot/openpi/.venv/bin/python scripts/train.py pi05_droid_franka_lora_10task_v2" | head -1)
echo "log: $LOG | 进程: ${PID:-不在跑}"
grep -i -m3 "Traceback\|out of memory\|RESOURCE_EXHAUSTED" "$LOG" | cut -c1-160
grep "Progress on:" "$LOG" | tail -1 | sed -E 's/.*Progress on: //; s/ postfix:.*//'
tail -n0 -f "$LOG" | grep --line-buffered "Progress on:\|^Step \|Traceback\|RESOURCE_EXHAUSTED" | sed -u -E 's/.*Progress on: /\r/; s/ postfix:.*//; s/^\r/\r/' | while IFS= read -r line; do case "$line" in $'\r'*) printf '%s   ' "$line";; *) printf '\n%s\n' "$line";; esac; done
