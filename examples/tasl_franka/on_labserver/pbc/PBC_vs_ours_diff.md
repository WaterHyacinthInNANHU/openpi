# PBC 预训练 vs 我们的 pbc 微调链路：逐项对比（2026-08-25）

来源：AXIS-Bench `myan/cotrain-phase` 的 `docs/reports/2026-08-24-cotrain-bc-audit.md`、`dihong/droid-training` 的两个
converter、openpi fork `box/server2-heldout-d8` 的 `axis_franka_policy.py` / `config.py::_axis_pretrain_config`、
ckpt 自带 README + norm_stats；我们这边是 `pi05_pbc_franka_lora_10task` + `tasl_fr3_10task_250ep(_pbc)` 实测。

| 项 | PBC 预训练 | 我们 (pbc 微调) | 判定 |
|---|---|---|---|
| 图像几何 | 两相机全高居中正方形 crop → 224（无黑边） | 同几何；训练侧从 letterbox 抠 126² 放大 1.78×，serve 侧 720²→224 | 几何一致，**清晰度不一致**（已知） |
| 图像来源 | sim 640×360 渲染 + DROID 180×320 真机 | ZED 2i / ZED Mini 1280×720 | 分辨率不同，crop 后都是 224，无需处理 |
| 夹爪 | sim 为 Robotiq；DROID 也是 Robotiq 2F-85 | Robotiq | 外观一致 |
| state | [q0..q6, closedness 0=开→1=闭] 8-D | DroidInputs 拼 [joints, gripper_position]，gripper 0..1、1=闭 | 一致 |
| action | [dq0..dq6 rad/s clip ±1, gripper position] | [dq 归一化 ±1 ≈ rad/s（拟合斜率 ~1）, gripper 0..1] | 一致 |
| 速度含义 | AXIS 半：state 差分的**实测**速度；DROID 半：**指令**速度 | Gello P 控制**指令**（有真机死区，25% 帧机器人没动） | 同 DROID 半；死区问题是我们数据自身的 |
| action 幅度 | actions std [0.175 0.286 0.162 0.280 …] | std [0.078 0.149 0.077 0.108 …] | 我们慢一半（和对 DROID stats 时一样），非 bug |
| fps | 15 | 15 | 一致 |
| action_horizon | 15 | 15（droid-lora 家族是 16） | 一致 |
| norm stats | 自算（25/75 混合），quantile norm | 复用 PBC 那份（`Devon018/Franka-Datasets-v2`） | 一致（同"沿用 base 的 stats"规则） |
| discrete_state_input | pi05 默认 True | pi05 默认 True | 一致 |
| prompt | `prompt_from_task=True`：DROID 取主指令，AXIS 取任务文本 | `prompt_from_task=True`，10 条人工重写指令 | 一致 |
| 图像增强 | openpi 默认（非 wrist 随机 crop 95% + 旋转 ±5° + 颜色抖动） | 同一段 openpi 代码 | 一致 |
| 静止帧过滤 | 两半都按 DROID 官方 idle 规则过滤（AXIS 去 8.9%，DROID 去 5.4%） | `pi05_pbc_franka_lora_10task` **未挂** filter_dict_path | **可选差异**：想对齐就挂 `nonidle_ranges.json`（v1，官方规则）或 v2 |
| 长 episode 截断 | AXIS 半丢 >60 s 的 episode；DROID 半不丢 | 最长 971 帧 = 65 s，仅 1 条超 60 s | 忽略 |
| 训练方式 | 全参数（3.35B 全训），bs 64，200k 步，10k warmup → 5e-5 常数，EMA 0.999 | LoRA（gemma_2b_lora + 300m_lora），bs 32，20k 步，1k warmup → 5e-5，无 EMA | 有意的选择，不是错配 |
| init | pi05_droid | PBC 199999（EMA 权重） | — |
| serve 路径 | audit 明确：必须走该 config 自己的 data_transforms | `pi05_pbc_*` 的 crop 在 data_transforms 里，dashboard 发全帧 | 一致 |

## 结论

除了清晰度这一项，真正能从 repo 看出来、且我们**还没对齐**的只有一个：静止帧过滤。PBC 两半数据都过滤了，
我们的 pbc config 现在没挂。其余（state/action 语义、夹爪方向、fps、horizon、stats、prompt、增强、serve 路径）
都对得上。

PBC 自己的 audit 里还有两条和我们相关的提醒：
1. 它的 DROID 半也被 crop 了，所以模型从没见过黑边——直接拿 letterbox 数据喂它是 domain shift（这正是做 `*_pbc` 的原因）。
2. 它没有 DROID-only 对照，"PBC 比 pi05_droid 强"这件事本身没被验证过；我们真机 eval 时最好 pi05_droid-lora 和 pbc-lora
   同一批 scene 配对跑。
