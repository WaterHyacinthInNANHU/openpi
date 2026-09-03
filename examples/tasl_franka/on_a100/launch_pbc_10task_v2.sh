#!/bin/bash
# AXIS Server 3 — cotrain 200k 底座 + 10task 真机数据 LoRA 微调
# 用法:  bash launch_cotrain.sh            (默认 8 卡 batch 64)
#        BATCH=32 bash launch_cotrain.sh   (退回 batch 32)
set -u

W=/localdisk/tasl_franka_finetune
CFG=pi05_cotrain_franka_lora_10task_pbc_v2
EXP=${EXP:-cotrain_10task_pbc_v2_b64}
BATCH=${BATCH:-64}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
MEMFRAC=${MEMFRAC:-0.92}

source /localdisk/dihong_workspace/env.sh 2>/dev/null   # 代理(HF 走 GFW 外)
export PYTHONPATH=$W/openpi/src                         # 让 openpi 解析到我们这棵树
export HF_LEROBOT_HOME=/localdisk/dihong_workspace/lerobot_home
export OPENPI_DATA_HOME=$W/cache/openpi
export CUDA_VISIBLE_DEVICES=$GPUS
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC
export WANDB_MODE=offline
ulimit -n 1048576 2>/dev/null

# ---- 起飞前自检:任一条不过就不起 ----
FILTER=$W/filters/nonidle_ranges_tail10.json
BASE=/localdisk/dihong_workspace/runs/ckpts/pi05_axis_droid_cotrain/cotrain_sim25/199999
DS=$HF_LEROBOT_HOME/ZhixuLi/tasl-fr3-10task-pbc-v2

[ -r "$FILTER" ] || { echo "FATAL: 缺静止帧过滤文件 $FILTER"; exit 1; }
[ -d "$BASE/params" ] || { echo "FATAL: 缺底座权重 $BASE/params"; exit 1; }
[ -r "$BASE/assets/Devon018/Franka-Datasets-v2/norm_stats.json" ] || { echo "FATAL: 缺底座 norm stats"; exit 1; }
N=$(ls "$DS/data/chunk-000/" 2>/dev/null | wc -l)
[ "$N" -eq 392 ] || { echo "FATAL: 数据集不完整,只有 $N/392 个 parquet"; exit 1; }
[ -r "$DS/meta/info.json" ] || { echo "FATAL: 缺 meta/info.json"; exit 1; }

echo "=========================================="
echo " config      : $CFG"
echo " exp         : $EXP"
echo " GPUs        : $GPUS  (n=$NGPU, fsdp=$NGPU, mem_frac=$MEMFRAC)"
echo " global batch: $BATCH   (每卡 $((BATCH/NGPU)))"
echo " 底座        : $BASE"
echo " norm stats  : 底座自带 Devon018/Franka-Datasets-v2"
echo " 数据        : $DS  ($N parquet)"
echo " 过滤        : $FILTER"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=========================================="

cd $W/openpi
exec $W/openpi/.venv/bin/python scripts/train.py $CFG \
    --exp-name=$EXP \
    --batch-size=$BATCH \
    --fsdp-devices=$NGPU \
    --num-workers=8 \
    --overwrite
