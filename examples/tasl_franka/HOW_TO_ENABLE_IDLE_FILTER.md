# 如何开启 openpi 自带的静止帧过滤(idle filter)

实施并验证于 2026-08-24。**判据和常量完全是 openpi 官方的,没有改成自定义版本。**

---

## 一句话用法

训练时把 config 名字换成带 `_filtered` 的那个:

```bash
# 不过滤
setsid nohup bash /data1/vla-reasoning/launch_train_10task.sh          > /tmp/a.log 2>&1 < /dev/null &
# 开过滤
setsid nohup bash /data1/vla-reasoning/launch_train_10task_filtered.sh > /tmp/b.log 2>&1 < /dev/null &
```

两个 config 除过滤开关和步数外**完全一致**(同样的 LoRA 配方、优化器、学习率、DROID norm-stats)。

启动后日志里会打这一行,据此确认生效:

```
Idle filter: keeping 63909/66463 frames (96.2%) from .../meta/nonidle_ranges.json
```

---

## 数据流

```
① openpi/examples/droid/compute_droid_nonidle_ranges.py --source lerobot
        │   离线跑一次(每个数据集一次),训练时不 import 它
        ▼
② <dataset>/meta/nonidle_ranges.json
        │   {"0": [[0,272]], "1": [[0,329]], ..., "172": []}
        │   每条 episode 里哪些区间可以作为采样起点
        ▼
③ config.py 的 filter_dict_path 指向这个 json
        ▼
④ data_loader.py 训练时读它,构造可采样索引子集
```

**光改 config 不够** —— 它只写了一个文件路径,那个 json 必须先存在。

---

## 为什么需要改动(不能直接用)

openpi 本来就有 `DataConfig.filter_dict_path` 这个开关,默认值是官方预计算的
`gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json`。但两个坎:

1. **这个开关只在 RLDS 数据路径上生效。** `data_loader.py`:
   ```python
   if data_config.rlds_data_dir is not None:
       return create_rlds_data_loader(...)   # 只有这条读 filter_dict_path
   return create_torch_data_loader(...)      # 我们走的 LeRobot 路径,不读
   ```
   我们的 config 是 `LeRobotRLinfDROIDDataConfig`,`rlds_data_dir=None`,
   **填了 filter_dict_path 也不会有任何效果**。

2. **官方那份预计算的 json 对我们没用。** 它按 DROID 的 episode 标识
   (`recording_folderpath--file_path`)索引,我们的 episode 查不到会一律返回 False,
   结果是**全部数据被滤掉**。

---

## 改动清单(三处,都在 openpi repo 内,各留日期备份)

### 1. `examples/droid/compute_droid_nonidle_ranges.py` — 扩展官方离线脚本
备份:`compute_droid_nonidle_ranges.py.bak-20260824`

**没有另写脚本,而是给官方这个脚本加了一个数据源分支。** 原来它只能读 DROID 的 RLDS/TFDS,
现在多了 `--source lerobot`。过滤算法抽成了 `nonidle_ranges()` 函数,**逻辑和常量与原版一字不差**,
两个数据源共用同一份实现:

```python
idle = np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)
min_idle_len = 7                # 连续 >此数 的 idle 帧全部滤掉
min_non_idle_len = 16           # 非 idle 段短于此数则整段滤掉
filter_last_n_in_ranges = 10    # 每个保留区间末尾再砍掉这么多帧
```

LeRobot 分支只换了两处数据来源:

- **joint_velocities**:DROID 用 `action_dict/joint_velocity`(rad/s);
  我们用 parquet 的 `actions[:, :7]`(归一化到 [-1,1] 的关节速度指令),
  乘 `--vel-scale`(默认 **0.509**)换算回 rad/s。
  这个系数是实测的:把实际关节运动(`diff(state[1:8]) * fps`)对指令做线性回归,
  7 个关节相关系数 0.64~0.78。
- **episode key**:DROID 用 `recording_folderpath--file_path`;我们用 `episode_index`。

**原版的 RLDS 用法行为未变**,只是从写死路径改成了命令行参数。

用法:

```bash
cd /data1/Franka_RealRobot/openpi/examples/droid
UVPY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11

# 我们的 LeRobot 数据(换数据集时重跑这个)
PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages \
HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home \
$UVPY compute_droid_nonidle_ranges.py --source lerobot \
    --repo-id franka/<你的数据集> \
    --out /data1/Franka_RealRobot/lerobot_home/franka/<你的数据集>/meta/nonidle_ranges.json

# 原版 DROID RLDS 用法
$UVPY compute_droid_nonidle_ranges.py --source rlds \
    --builder-dir <path_to_droid_tfds> --out <path_to_json>
```

### 2. `src/openpi/training/data_loader.py` — 让 LeRobot 路径也认这个开关
备份:`data_loader.py.bak-20260824`(+60 行,纯新增)

- 新增 `SubsetDataset`:把数据集限制到一个索引子集
- 新增 `_nonidle_indices()`:把 `{episode_index: [[start,end), ...]}` 映射成全局帧索引
- `create_torch_dataset()` 里:`filter_dict_path` 不为 None 时套上 `SubsetDataset`

**关键设计(与官方一致):帧不从数据集里删除,只缩小「可采样的索引集合」。**
所以 action chunk 仍然读原始的连续动作,**永远不会跨越被过滤掉的缺口**。

> 反面教材:如果改成物理删帧再把 timestamp 重排成连续,缺口会被抹平,
> 跨切口的 chunk 会静默拼出一个"瞬移"。8/9 那次的处理就是这样,
> 而且 openpi 的 loss 对 action chunk 的 16 步**没有任何掩码**(`is_pad` 全仓零引用),
> 所以这种错误标签会照单全收。

### 3. `src/openpi/training/config.py` — 新增一个 config
备份:`config.py.bak-20260824`
新增 `pi05_droid_franka_lora_10task_filtered`(约第 1132 行)。

---

## 过滤效果(tasl_fr3_10task_250ep)

```
episodes          : 250 (整条被滤掉 1 条 —— ep172,只有 4 帧)
frames            : 66463 -> 保留 63909 (96.2%),滤掉 2554 (3.8%)
ranges per episode: 中位 1,最多 2
```

很温和 —— `min_idle_len=7` 的门槛把零散短停顿都放过了,只删成段的真静止。

---

## 验证记录

| 检查项 | 结果 |
|---|---|
| 不开过滤时数据集长度 | 66,463 ✅ |
| 开过滤后长度 | 63,909 ✅ 与 ranges.json 期望值完全一致 |
| 独立复算允许集合 | 63,909,与 patch 算出的 keep **逐元素一致** ✅ |
| 抽样(首/中/尾) | `actions (16,8)`、`image (3,224,224)`、prompt 正确解析 ✅ |
| 被滤掉的 2,554 帧 | 确认**不在**可采样集合里 ✅ |
| 扩展后的官方脚本 vs 之前的独立脚本 | 产出的 json **逐条完全一致** ✅ |
| 两个 config 加载 | 都通过 ✅ |

---

## 注意

- **目前 17 个 checkpoint 全部是未过滤数据训出来的**;`_filtered` 这个 config 还没跑过训练。
- `_filtered` 默认 20,000 步,和未过滤的 20k 基准对齐,方便 A/B 对照。
- openpi 这三个文件属主是 `zli538`(别人的 checkout,权限恰好可写),已各留一份日期备份。
- **起训练前先看 `nvidia-smi`**:GPU 是共享的。如果和别人共用同一张卡,
  `XLA_PYTHON_CLIENT_MEM_FRACTION` 必须降下来(49140 MiB 总量,别人占 8.4G 时
  0.9 会 OOM,0.8 才安全);更稳妥的做法是只列空闲卡。
  `launch_train_10task_filtered.sh` 里 `GPUS` 和 `MEMFRAC` 两个变量都可以覆盖。

---

## v2 加强版过滤(2026-08-24 晚,已生成、未训练)

**问题**:官方判据 `idle = all(|Δv| < 1e-3)` 用的是 *指令* 速度。我们的指令来自 GELLO P 控制残差,
永远带噪、永远非零(5% 分位就有 0.084),所以 v1 只删 3.8%;但按实测速度 `diff(state[1:8])×15`
看,机器人静止的帧有 18.8%。这批"指令说动、机器人没动"的帧就是 dummy action:小指令落在真机死区
(`|cmd| < ~0.13` 时 37~73% 的帧没动,`≥0.2` 才线性跟随),教模型输出过小的动作 → eval 太慢/不动。

**v2 判据**(官方规则和 4 个常量一字不改,只换速度来源 + 加夹爪保护):

```bash
cd /data1/Franka_RealRobot/openpi/examples/droid
HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home \
../../.venv/bin/python compute_droid_nonidle_ranges.py --source lerobot \
    --repo-id franka/tasl_fr3_10task_250ep \
    --vel-source state --grip-guard 5 \
    --out /data1/Franka_RealRobot/filters/tasl_fr3_10task_250ep/nonidle_ranges_v2.json
```

| 新 flag | 含义 | 默认 |
|---|---|---|
| `--vel-source {action,state}` | `action` = 指令速度(原行为);`state` = 实测速度 `diff(state[:,1:8])×fps` | `action` |
| `--grip-guard N` | `\|Δgrip\| > --grip-delta` 的帧前后 ±N 帧永不判 idle(抓/放时手臂停着是有效监督) | 0(关) |
| `--grip-delta` | 夹爪事件阈值 | 0.02 |

默认参数输出和 v1 **逐字节一致**(已回归);RLDS 路径没动。脚本备份 `compute_droid_nonidle_ranges.py.bak-20260824-2321`。

**效果**:66463 → 58760 帧(保留 88.4%,删 11.6%);夹爪动作帧误删 2.8%;整条删掉 1 条(ep172);每条段数中位 1、最多 6。
独立复算逐元素一致(`data_pipeline/on_labserver/filter_v2_study/`)。

**存哪**:原始 parquet 一帧未动。ranges json 在 `/data1/Franka_RealRobot/filters/tasl_fr3_10task_250ep/nonidle_ranges_v2.json`
(dataset 的 `meta/` 属主是 vla-reasoning,zli538 写不进去);v1 的 `meta/nonidle_ranges.json` 保留。

**训练**:config `pi05_droid_franka_lora_10task_v2`(config.py 第 1189 行,备份 `config.py.bak-20260824-2330`),
30k 步、每 5k 存 ckpt、warmup 3000、官方 cosine lr (2.5e-5 → 2.5e-6);数据/模型/LoRA 配方和 10task 一致。启动脚本 `data_pipeline/on_labserver/launch_train_10task_v2.sh`(**还没启动**,等指令)。
换新 pretrain ckpt 时只改该 config 的 `weight_loader` 路径。

**其他备选判据的数字**(同一份数据,都套官方段级规则):实测 `|v|<0.01`+保护±8 删 16.7%/夹爪误删 2.6%;
`|v|<0.01`+保护±5 删 17.9%/7.2%;`|cmd|<0.13` 死区判据删 28.1% 但夹爪误删 42.5%(不可用)。
DROID 自己的 policy_learning repo 不看速度,用遥操 `movement_enabled` 标志物理删帧;我们 `intervene_flag` 全 1,没这个信号。

### v2 的最终形态:物理导出的数据集(2026-08-25)

用户要求"滤掉的帧真的删掉、做成全新数据集、原版保留"。于是 v2 从"ranges 文件"改成**实体数据集**:

```
/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_v2/     ← 训练用 (repo_id franka/tasl_fr3_10task_v2)
/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep/  ← 原版,一帧未动
```

- 导出脚本:`on_labserver/export_10task_v2.py`。段的判据/常量直接 import 官方 `compute_droid_nonidle_ranges.py`
  (实测速度 + 夹爪保护 ±5),**尾砍设为 0**:物理导出后段与段之间没有缺口,chunk 越过段尾按 LeRobot 常规补末帧。
  所以删的是 5.7%(3783 帧),比 ranges 版的 11.6% 少的那部分就是每段尾部的 10 帧(它们不是 dummy 帧,只是 ranges 版为了
  不让 chunk 伸进缺口而不许当起点)。
- 每个保留段 → 一条独立 episode:**392 条 / 62680 帧**,长度 16~936(中位 153),每 task 30~50 条;ep172 整条没进来。
  图像/state/action/task 整行原样拷贝(bytes 级一致),只重编 episode_index / frame_index / timestamp / index / done。
- `meta/episodes_stats.jsonl` 按 lerobot `compute_stats` 的算法重算;`meta/source_segments.json` 记每条新 episode 来自原版哪条第几到第几帧。
- 核对:逐行与源比对图像 bytes / state / action / task_index 全部一致;timestamp、index 连续;openpi loader 读出 62680 条,
  段尾样本 chunk 越界补末帧正确(`filter_v2_study/verify_v2_export.py`)。
- config `pi05_droid_franka_lora_10task_v2` 现在 `repo_id="franka/tasl_fr3_10task_v2"`,**没有** `filter_dict_path`。
  ranges 版的 `filters/.../nonidle_ranges_v2.json` 和 loader 补丁保留,想 A/B 时把 repo_id 换回 250ep 并加回 filter_dict_path 即可。

**已知取舍(2026-08-25,用户拍板"不动")**:实体版不砍段尾,段尾最后 15 帧当起点时 chunk 越界用末帧重复填充(openpi 无 `is_pad` 掩码,官方 DROID 管线同样重复末帧)。
含补帧样本 9.4%(原版 5.6%),补帧占目标位置 4.7%(原版 2.8%),被重复的段尾动作中位 |cmd| 0.153。想根治可给 `compute_loss` 加 `actions_is_pad` 掩码(LeRobot 已返回该键)。
与 v1 的关系:v1 删的 2554 帧里 2303 帧是尾砍(非静止),真静止 251 帧中 v2 删了 232,剩 19 帧(ep98/ep232)判据差异。

### 最终决定(2026-08-25 01:10):主线回到官方 ranges 模式

用户看过官方 `compute_droid_nonidle_ranges.py` 的 docstring(区间尾砍 10 帧是因为这些 chunk 含大量 idle 动作;`min_idle_len=7`
是因为 DROID 策略每 chunk 执行前 8 步,这样"策略不会卡在输出静止动作")后,决定**按官方做**:

| config | 数据 | 说明 |
|---|---|---|
| `pi05_droid_franka_lora_10task_v2`(**主线,在训**) | `tasl_fr3_10task_250ep` + `filters/.../nonidle_ranges_v2.json` | 帧不删;5.7% dummy 段 + 每区间末尾 10 帧不做起点;58760 起点 |
| `pi05_droid_franka_lora_10task_v2_physical`(备用) | `tasl_fr3_10task_v2`(392 ep / 62680 帧) | 物理删 5.7%,段尾不砍,补帧 4.7% |

两者其余超参相同:30k 步、每 5k 存、warmup 3000、cosine 2.5e-5→2.5e-6、LoRA、无 EMA、DROID stats。
实体版训练跑到 939 步后被杀掉重开(无 ckpt 产出);ranges 版 01:09 起,exp `pi05_droid_franka_lora_10task_v2_v0`,GPU 1-4。
⚠️ config.py 同时被别的 session 加了 `pi05_pbc_*` 两个 config(第 1338 / 1362 行),改 config 前 `grep -n TrainConfig(` 确认布局,别用旧备份整体覆盖。

### 真·最终(2026-08-25 01:16):主线 = 实体数据集 tasl_fr3_10task_v2 + 尾 10 帧 json

用户明确"我的数据集就是 tasl_fr3_10task_v2",于是改成:
- 数据集 `lerobot_home/franka/tasl_fr3_10task_v2`(物理删掉 5.7% dummy 段,392 ep / 62680 帧,原版 250ep 保留)
- `filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json`:用官方脚本在该数据集上跑一遍,392 条每条恰好 `[0, L-10)`(也证明导出后无残留静止段),58760 起点
- config `pi05_droid_franka_lora_10task_v2`(config.py:1235)= 这两者;`_physical` config 已删;`filters/tasl_fr3_10task_250ep/nonidle_ranges_v2.json` 留作"原数据 + ranges"的 A/B
- 与"原数据 + ranges"的差别只在段尾起点 chunk 的末几步:前者补末帧(合成),后者读真实 dummy 动作;都约 0.26% 目标位置
- 训练 01:16 起,exp `pi05_droid_franka_lora_10task_v2_v0`,GPU 1-4;之前两次(实体版 939 步、ranges 版 ~300 步)均已杀掉,无 ckpt 产出

### 打包版(2026-08-25,给 TASL1 用)

`data_pipeline/filter_v2_pack/`(tarball `data_pipeline/filter_v2_pack.tar.gz`):自包含,只依赖 numpy/pyarrow/pillow。
`run_all.sh` = 导出 `tasl_fr3_10task_v2` → 官方脚本生成尾 10 帧 json → 逐行核对;用法见包内 `README.md`。
labserver 上 `LIMIT=5` 冒烟:输出与正式 v2 数据集逐字节一致(parquet md5 / stats / provenance)。

### pbc 线收敛(2026-08-25 12:50)

config.py 里两个旧 pbc config 已删(备份 `.bak-20260825-1250`),只留 `pi05_pbc_franka_lora_10task_v2`(config.py:1286)= v2 配方 + PBC 底座/stats/centre-crop,
数据 `tasl_fr3_10task_v2_pbc`(`on_labserver/pbc/make_pbc_dataset.py` 从 v2 生成)+ 同一份尾 10 帧 json。启动 `on_labserver/pbc/launch_train_pbc_10task_v2.sh`。打包版 `train_side/` 已含。
