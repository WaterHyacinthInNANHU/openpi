# examples/tasl_franka — TASL FR3 数据处理 + 训练启动(labserver `/data1/Franka_RealRobot/data_pipeline/` 的同步副本)

- `INDEX.md`:所有脚本清单(tasl-1 采集/合并侧 + labserver 训练侧)
- `HOW_TO_ENABLE_IDLE_FILTER.md`:dummy-action(静止帧)过滤的来龙去脉,v1 → v2 → 实体数据集 + 尾 10 帧 json
- `filter_v2_pack/`:自包含的 v2 过滤打包版(export / 尾 10 json / verify / 对拍),含 `train_side/` 的两份 config 和启动脚本
- `on_labserver/`:训练启动 / watcher / pbc 数据集转换;`on_tasl1/`:采集机侧脚本

对应 config:`pi05_droid_franka_lora_10task_v2`(主线)、`pi05_pbc_franka_lora_10task_v2`(PBC 底座),见 `src/openpi/training/config.py`。
