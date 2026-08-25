# filter_v2_pack — Franka 10task "dummy action" 过滤(v2)打包

自包含,只依赖 `numpy pyarrow pillow`(`requirements.txt`),**不需要** openpi / lerobot 环境。
在 labserver 上已验证:打包版对前 5 条 episode 的输出与正式 `tasl_fr3_10task_v2` 逐字节一致(parquet md5、stats、provenance 全同)。

## 一句话用法(TASL1)

```bash
pip install -r requirements.txt          # 已有 numpy/pyarrow/pillow 可跳过
SRC=/path/lerobot_home/franka/tasl_fr3_10task_250ep \
OUT=/path/lerobot_home/franka/tasl_fr3_10task_v2 \
FILTER_JSON=/path/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json \
bash run_all.sh
```

- `SRC` 原数据集,**一帧不动**;`OUT` 必须不存在(脚本拒绝覆盖);`PY=/xx/python` 可指定解释器。
- 先试水:加 `LIMIT=5` 只处理前 5 条源 episode(输出到临时目录即可)。
- 250 条全量约 5~10 分钟(瓶颈是逐条解码图像算 stats),建议 `nohup ... &`。

## 方法(和训练时用的完全一致)

```
tasl_fr3_10task_250ep ──(1) export_filtered_dataset.py──> tasl_fr3_10task_v2 ──(2) nonidle_ranges.py──> nonidle_ranges_tail10.json
                                                                    └──(3) verify_export.py ──> VERIFY OK
```

**(1) 物理删掉 dummy 段**(`export_filtered_dataset.py`)
- 判据 = openpi 官方 `compute_droid_nonidle_ranges.py` 的段级规则,4 个常量一字不改:
  `idle_threshold=1e-3`(rad/s,Δv 全关节都小于它 → 该帧 idle)、`min_idle_len=7`(连续 ≥7 帧 idle 才删)、
  `min_non_idle_len=16`(删完后剩余段 <16 帧整段丢弃)。
- 唯一区别:速度不用 *指令*(`actions[:, :7]`,GELLO 残差永远非零,官方规则只能删 3.8%),
  改用 **实测** `diff(state[:, 1:8]) * fps`(`--vel-source state`);
  再加 **夹爪保护** `--grip-guard 5`:`|Δgrip action| > 0.02` 的帧前后 ±5 帧永不判 idle(抓/放时手臂停着是有效监督)。
- 每个保留段 → 一条新 episode;图像/state/actions/task/is_success **整行原样拷贝**,只重编
  `episode_index / frame_index / timestamp / index / done`。段尾这一步 **不砍**(`--tail 0`),留给第 (2) 步。
- `meta/episodes_stats.jsonl` 按 lerobot `compute_stats` 算法重算(函数已内置);`meta/source_segments.json` 记每条新 episode 来自源哪条第几到第几帧;`tasks.jsonl` 原样复制。
- labserver 结果:250 ep / 66463 帧 → **392 ep / 62680 帧**(删 3783 = 5.7%,ep172 整条剔除,长度 16~936,中位 153)。

**(2) 尾 10 帧不做 chunk 起点**(`nonidle_ranges.py` = openpi 官方脚本 + 我们加的 3 个 flag,默认参数输出与官方逐字节一致)
- 在 **新** 数据集上用 **官方默认参数** 跑一遍,得到 `filter_dict_path` 用的 json:每条 episode `[[0, L-10]]`。
  这是官方 DROID 的语义(`filter_last_n_in_ranges=10`:段尾 chunk 含很多 idle 动作,不当起点;帧本身不删,仍可作为 chunk 目标)。
- 顺带证明导出后没有残留 ≥7 帧的静止段(`run_all.sh` 会核对每条都恰好 `[0, L-10)`)。
- labserver 结果:392 条全部 `[0, L-10)`,**58760** 个起点。

**(3) 核对**(`verify_export.py`):index/timestamp 连续、`done` 只在末帧、meta 三个文件条数一致、
按 provenance 与源逐行比对图像 bytes / state / actions / task_index / is_success。最后一行 `VERIFY OK` 才算过。

## 接到训练

openpi config(参见 labserver `config.py` 的 `pi05_droid_franka_lora_10task_v2`):
```python
data=LeRobotRLinfDROIDDataConfig(
    repo_id="franka/tasl_fr3_10task_v2",
    base_config=DataConfig(prompt_from_task=True,
        filter_dict_path="<FILTER_JSON>"),   # nonidle_ranges_tail10.json
    ...)
```
`HF_LEROBOT_HOME` 指向 `OUT` 的上两级目录。LeRobot 路径认 `filter_dict_path` 依赖 labserver 上的 `data_loader.py` 补丁(`SubsetDataset`),原版 openpi 没有;不用尾砍 json 也能直接训 `OUT`。

## 调参 / 变体

| 想要 | 怎么做 |
|---|---|
| 换保护窗口 / 夹爪阈值 | `GRIP_GUARD=8 GRIP_DELTA=0.02 bash run_all.sh` |
| 回到官方指令速度判据 | `python export_filtered_dataset.py --vel-source action --vel-scale 0.509 ...` |
| 只要 ranges json、不导出(原数据 + 尾砍模式) | `python nonidle_ranges.py --source lerobot --root <lerobot_home> --repo-id franka/tasl_fr3_10task_250ep --vel-source state --grip-guard 5 --out xxx.json`(labserver 上这版是 `filters/tasl_fr3_10task_250ep/nonidle_ranges_v2.json`,58760 起点) |

已知取舍:物理导出后段尾最后 15 帧当起点时 chunk 越界按 openpi 常规重复末帧(官方 DROID 管线同样无 `is_pad` 掩码),
含补帧样本 9.4%、补帧占目标位置 4.7%;用户已拍板不处理。
分析用脚本(死区统计、判据变体对比)在 labserver `data_pipeline/on_labserver/filter_v2_study/`,不在本包。

## 训练侧(`train_side/`)

主线 TrainConfig 原文、`LeRobotRLinfDROIDDataConfig` + repack transform、`data_loader.py` 的 `SubsetDataset` 补丁(diff + 整文件)、启动 / watcher 脚本、超参和 norm stats 说明,见 `train_side/README.md`。

## 和 labserver 逐字节对(`reference/` + `compare_reference.sh`)

`reference/` 里是 labserver 正式 `tasl_fr3_10task_v2` 的 392 个 parquet + 5 个 meta 文件的 md5、尾 10 帧 json 原件及其 md5、`source_segments.json` 原件。跑完 `run_all.sh` 后:
```bash
OUT=/path/lerobot_home/franka/tasl_fr3_10task_v2 \
FILTER_JSON=/path/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json \
bash compare_reference.sh
```
期望两行 `ALL IDENTICAL` / `IDENTICAL`。`source_segments.json` 里记了绝对源路径,SRC 放在别的位置时只有它会不同,parquet / stats / episodes 必须全同。
