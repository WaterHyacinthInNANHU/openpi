# AXIS-Bench 里和 LoRA 微调有关的规则 / 配方（2026-08-25 整理）

来源：AXIS-Bench `docs/decisions/0003-compute-norm-stats-on-the-finetune-dataset.md`、
`docs/heldout_20_benchmark_spec.md` §4 / "Evaluation protocol"、openpi fork `box/server2-heldout-d8`
`config.py::_axis_heldout_multitask_config` / `_slb_freeze_filter`。

## 1. 硬规则：每次 finetune 都用 **finetune 数据自己算** 的 norm stats（ADR 0003）

- 明确反对沿用预训练 stats（也就反对 openpi `pi05_droid_finetune` 的"reuse DROID stats"建议）。
- 证据：robomimic 微调时沿用 AXIS 预训练 stats → 归一化后 per-dim std 1.2–3.4、p99|z| 6.7，成功率 ~0%；
  换成自算 stats（std≈1、p99|z|≈2.6）后成功率恢复，两条 p<0.05 的"结论"被撤回。
- 要求：训练前先验证 per-dim std ≈ 1、p99|z| ≈ 2.6；p99|z| 到 6.7 就是这个 bug 的签名。
- 注意他们的失败方向是 stats **过小 → z 爆掉**。我们实测（10task 数据，joints 0–6）：

| stats | actions z-std | p99\|z\| | state z-std | p99\|z\| |
|---|---|---|---|---|
| DROID（droid-lora 家族在用） | 0.36–0.97 | 2.65 | 0.42–1.22 | 2.93 |
| PBC（pbc config 现在在用） | 0.39–1.02 | 2.75 | 0.34–0.72 | 1.89 |
| 自算 | 1.0 | 3.64 | 1.0 | 2.83 |

我们是反方向：信号被压到 0.4–0.5 倍，没有爆掉的签名，但也不满足"std≈1"。pi05 实际用 quantile norm，
DROID/PBC stats 下 action 的 quantile-norm std 只有 0.13–0.35，自算能到 0.30–0.41。

## 2. 他们的 LoRA 配方（`pi05_axis_heldout_multitask`，400 demo 多任务）

| 项 | AXIS | 我们 pbc-lora 现状 |
|---|---|---|
| 变体 | gemma_2b_lora + gemma_300m_lora | 同 |
| LR | warmup 100 → **2.5e-5** 常数 | warmup 1k → 5e-5 常数 |
| batch | 32 | 32 |
| 训练量 | **5 epoch**（16.5k 步 / 106k idle-filtered 样本）；20-demo gate 用 20 epoch | 20k 步 ≈ 9.6 epoch（50k 那版是 24 epoch） |
| EMA | None | None |
| 图像 | image_center_crop=True | 同 |
| norm stats | 自算（ADR 0003） | 复用 PBC |
| 静止帧 | 只在 idle-filtered 样本上训 + 算 stats | 未挂 filter |
| horizon | 10（他们 eef 臂）/ 15（droid8） | 15 |
| 视觉塔 | 默认可训；提供 `freeze_vision` 开关 | 默认可训 |

## 3. `_slb_freeze_filter` 的提醒：openpi 默认 LoRA **没冻 SigLIP**

openpi 的 `get_freeze_filter` 只冻 LLM/action-expert 的非 LoRA 权重，~400M 的 SigLIP 图像塔整体可训。
他们在 25-demo、外观随机化的 SLB 数据上发现这会破坏预训练的视觉 grounding（成功率一律 0–20%、全 timeout），
于是加了 `freeze_vision=True`（再冻 `.*img.*`），称之为"标准低数据 VLA 配方"。多任务 400-demo 那版他们
仍然让视觉塔可训。我们的 droid-lora 和 pbc-lora 目前都是视觉塔可训 + 5e-5，250 条 demo 训 10–24 epoch，
属于要留意的组合。

## 4. 流程性要求
- stats 只在真正训练的 split（idle-filtered 后）上算；训练前核对维度（8-D）和是否 stale。
- 驱动脚本要**每次重算** stats，不要"缺了才算"（他们因 config 同名复用了旧渲染的 stats）。
- 对比不同预训练时，所有臂共用同一份（finetune 数据算的）stats，否则 normalization 成了第二个变量。

## 5. 决策记录
- 2026-08-25：用户拍板 **pbc v2 继续复用 PBC 的 norm stats**（openpi 官方 "reuse the base's stats" 做法，和 droid-lora 家族规则一致），
  不按 ADR 0003 自算。依据：实测 PBC stats 下 p99|z| 2.75，没有 ADR 里 6.7 那种爆掉签名，只是信号偏小；保持两家族同一规则便于对比。
- 视觉塔冻不冻、epoch 数、数据清晰度（重采 crop 模式）三项待问负责人。
