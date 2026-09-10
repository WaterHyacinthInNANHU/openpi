# examples/tasl_franka — TASL FR3 真机项目在 openpi 里的全部代码（训练 + 评测）

分支 `tasl-franka`（WaterHyacinthInNANHU/openpi）= AXIS `main` + `RealWorldFR3` 微调分支 + A100 配置 + tasl-1 评测栈。
数采（collect 面板、遥操/GELLO/Pico、NUC 控制器镜像、数据集导出与过滤工具）在 tasl-lab/RLinf 的 `tasl/`，不在这里。

| 目录 | 内容 |
|---|---|
| `src/openpi/training/config.py` | 训练配置 `pi05_droid_franka_lora_10task_v2`（DROID 底座）、`pi05_cotrain_franka_lora_10task_pbc_v2` / `_cfg_v2`（AXIS 共训底座）；serve 配置 `pi05_cotrain_franka_serve` |
| `src/openpi/policies/rlinf_franka_{droid,pbc}.py` | 数据 repack（8 维 state/action 拆分、相机键）与中心裁剪变换 |
| `src/openpi/training/quality_conditioning.py`, `slb_cfg.py` | CFG 的 `\nQuality: N` 标签（训练侧） |
| `on_a100/` | AXIS Server 3 上的 PBC/CFG 启动脚本、环境说明、prompt 预检 |
| `on_labserver/` | labserver 上的 DROID 线训练启动/监控/上传脚本 |
| `filter_v2_pack/`, `HOW_TO_ENABLE_IDLE_FILTER.md` | 静止帧过滤（训练侧数据准备） |
| （已迁出）| **tasl-1 评测栈**（面板、RTC、NUC 客户端、启动脚本）现在住在 tasl-lab/RLinf 分支 `franka-fr3/realworld` 的 `tasl/` 下；RLinf 通过 submodule `third_party/openpi` 钉住本仓库的提交，实验 config 在那边 `tasl/experiments/`。本仓库只保留训练/serve 配置与策略变换。 |

---

## 原 README（数据处理 + labserver 训练，2026-08）

### TASL FR3 数据处理 + 训练启动(labserver `/data1/Franka_RealRobot/data_pipeline/` 的同步副本)

- `INDEX.md`:所有脚本清单(tasl-1 采集/合并侧 + labserver 训练侧)
- `HOW_TO_ENABLE_IDLE_FILTER.md`:dummy-action(静止帧)过滤的来龙去脉,v1 → v2 → 实体数据集 + 尾 10 帧 json
- `filter_v2_pack/`:自包含的 v2 过滤打包版(export / 尾 10 json / verify / 对拍),含 `train_side/` 的两份 config 和启动脚本
- `on_labserver/`:训练启动 / watcher / pbc 数据集转换;`on_tasl1/`:采集机侧脚本

对应 config:`pi05_droid_franka_lora_10task_v2`(主线)、`pi05_pbc_franka_lora_10task_v2`(PBC 底座),见 `src/openpi/training/config.py`。
