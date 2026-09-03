# 我们对 OpenPI 做的适配修改

整理于 2026-08-24 · 机器:TASL Lab Server · repo:`/data1/Franka_RealRobot/openpi`

上游基线:`Physical-Intelligence/openpi`,提交 `fd7caad`(在 `e061c09` "Pi05 + PyTorch support #634" 之后)。

**改动总量:8 个文件,+638 / −41 行。** 其中真正修改上游既有逻辑的只有一个文件,其余都是纯新增。

```
examples/droid/compute_droid_nonidle_ranges.py  +140 −41   ← 唯一改了既有逻辑的
src/openpi/policies/rlinf_franka_droid.py        +39        ← 新文件
src/openpi/training/config.py                   +275        ← 纯追加
src/openpi/training/data_loader.py               +60        ← 纯追加
runme_cal_stats_franka.sh                        +11        ← 新文件
runme_finetune_franka.sh                         +20        ← 新文件
runme_preflight_franka.sh                        +15        ← 新文件
scripts/franka_preflight_norm_check.py           +78        ← 新文件
```

---

## 为什么需要适配

OpenPI 的 π0.5-DROID 微调流程假设你的数据长得像 DROID。我们的数据来自 RLinf 在 TASL
FR3 台架上的采集管线,有三处对不上:

1. **列的组织方式不同。** DROID 把关节位置、夹爪、各路相机分成独立字段;
   RLinf 把 8 维状态塞进一个 `state` 列,相机叫 `image` / `extra_view_image`。
2. **数据格式不同。** 官方给自采数据准备的路径是 LeRobot,但**很多功能只在 RLDS 路径上实现**,
   静止帧过滤就是其中之一。
3. **没有现成的 TrainConfig。** 官方最接近的是 `pi05_droid_finetune`(20k 步 / batch 32),
   但它指向 DROID 的数据配置。

---

## 一、`src/openpi/policies/rlinf_franka_droid.py`(新文件,39 行)

**作用:把 RLinf 的列布局翻译成 `DroidInputs` 认识的键。**

RLinf 写出来的 LeRobot v2.1 数据集是这样的:

```
state[8]         = [gripper_position, q0..q6]       ← 夹爪在第 0 位
actions[8]       = [dq0..dq6, gripper]              ← 关节速度在前 7 位(注意顺序和 state 不一样)
image            = 外部 ZED 2i,224×224
extra_view_image = 腕部 ZED Mini,224×224
```

这个 transform 把它拆成:

```python
"observation/joint_position":        state[1:8]
"observation/gripper_position":      state[0:1]
"observation/exterior_image_1_left": image
"observation/wrist_image_left":      extra_view_image
"actions":                           actions          # 训练时是 [action_horizon, 8]
```

> ⚠️ **`state` 和 `actions` 的顺序不一样**(夹爪一个在头、一个在尾)。
> 分析数据时如果按同一套下标去对,会得到"动作和实际运动无关"的错误结论。
> 我们踩过这个坑:错位时 7 个关节的相关系数只有 0.02~0.06,对齐后是 0.64~0.78。

---

## 二、`src/openpi/training/config.py`(纯追加,275 行)

### 2.1 `LeRobotRLinfDROIDDataConfig`(第 458 行)

抄自官方的 `LeRobotDROIDDataConfig`,只把 repack 步骤换成上面那个 `RLinfFrankaDroidRepack`。
其余(DroidInputs / DroidOutputs、关节速度动作空间、不加 delta transform)完全一致。

### 2.2 五个 TrainConfig

| 行号 | 名字 | 数据 | 步数 | 说明 |
|---|---|---|---|---|
| 944 | `pi05_droid_franka_lora` | 16 条 | 20k | v0,8/5 首次微调 |
| 996 | `pi05_droid_franka_lora_5k` | 16 条(声称过滤过) | 5k | v2,8/9 |
| 1042 | `pi05_droid_franka_lora_5k_calibra50` | 8 条(Calibra 子集) | 5k | v3,8/9 |
| 1088 | `pi05_droid_franka_lora_10task` | **250 条 / 10 task** | 20k→50k | v4,当前主线 |
| 1143 | `pi05_droid_franka_lora_10task_filtered` | 同上 + 静止帧过滤 | 20k | v5,**还没跑过** |

五个共享同一套超参(改自官方 `pi05_droid_finetune`):

```python
model:      Pi0Config(pi05=True, action_dim=32, action_horizon=16,
                      paligemma_variant="gemma_2b_lora",
                      action_expert_variant="gemma_300m_lora")
weights:    gs://openpi-assets/checkpoints/pi05_droid/params
norm stats: 复用 DROID 的(assets_dir 指向 pi05_droid/assets,asset_id="droid")
optimizer:  AdamW(clip_gradient_norm=1.0)
lr:         warmup 1000 步到 5e-5,之后恒定(decay_steps=1e6 让余弦衰减不生效)
ema_decay:  None
batch_size: 32(全局,8 卡各 4)
```

**唯一相对官方改过的超参是 `peak_lr=5e-5`**(官方 `CosineDecaySchedule` 默认 2.5e-5)。

---

## 三、`src/openpi/training/data_loader.py`(纯追加,60 行)

**作用:让 LeRobot 数据路径也支持 `filter_dict_path`(静止帧过滤)。**

### 问题

OpenPI 本来就有这个开关,但只在 RLDS 路径上生效:

```python
# data_loader.py 的分支
if data_config.rlds_data_dir is not None:
    return create_rlds_data_loader(...)   # 只有这条把 filter_dict_path 传下去
return create_torch_data_loader(...)      # 我们走这条,参数根本传不到
```

我们的 config 是 `LeRobotRLinfDROIDDataConfig`,`rlds_data_dir=None`,
所以**即使把 `filter_dict_path` 填上也不会有任何效果**。

### 改动

新增两样,并在 `create_torch_dataset()` 末尾接上:

```python
class SubsetDataset(Dataset[T_co]):   # 把数据集限制到一个索引子集
def _nonidle_indices(dataset_meta, filter_dict_path)  # {episode: [[s,e),...]} -> 全局帧索引
```

### 关键设计:不删帧,只缩小可采样索引

这一点和官方的设计意图一致,也是**最容易做错的地方**:

- ✅ **我们的做法**:帧全部留在数据集里,只是"哪些帧能作为 action chunk 的起点"这个集合变小。
  chunk 本身照样读原始的连续动作,**永远不会跨越被过滤掉的缺口**。
- ❌ **错误做法**:物理删帧,然后把 timestamp 重排成连续。这样缺口被抹平,
  跨切口的 chunk 会把切口两侧的动作直接拼接,**训练模型去预测一个"瞬移"**,而且完全静默不报错。

> 8/9 那次的处理就是错误做法(`fix_idlefiltered_timestamps.py`)。
> 而且我们实测确认:**OpenPI 的 loss 对 action chunk 的 16 步没有任何掩码** ——
> `is_pad` 在整个 openpi 源码里零引用,LeRobot 明明返回了 `actions_is_pad`,
> `compute_loss` 却对整个 `(b,16,32)` 张量直接算。所以这种错误标签会被照单全收。

---

## 四、`examples/droid/compute_droid_nonidle_ranges.py`(+140 / −41)

**唯一修改了上游既有逻辑的文件。** 作用是给这个离线脚本加一个数据源分支。

### 为什么不新写一个脚本

一开始我确实另写了一个,但那样会出现两份几乎相同的过滤逻辑,以后容易走样。
现在的做法是:**把过滤算法抽成 `nonidle_ranges()` 函数,两个数据源共用同一份实现。**

### 改了什么

| | 原版 | 现在 |
|---|---|---|
| 数据源 | 只能读 DROID 的 RLDS/TFDS | `--source rlds`(原行为)/ `--source lerobot`(新增) |
| 路径配置 | 源码里两个写死的占位符,得手改文件才能跑 | 命令行参数 `--builder-dir` / `--repo-id` / `--out` |
| 算法 | 直筒式脚本 | 抽成 `nonidle_ranges(joint_velocities)` 函数 |
| tensorflow | 顶层 import | 懒加载(跑 lerobot 分支不需要装 TF) |

**过滤判据和四个常量一字未改:**

```python
idle = np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)
min_idle_len = 7                # 连续 >此数 的 idle 帧整段滤掉
min_non_idle_len = 16           # 非 idle 段短于此数则整段滤掉
filter_last_n_in_ranges = 10    # 每个保留区间末尾再砍这么多帧
```

LeRobot 分支只换两处数据来源:

- **joint_velocities**:DROID 用 `action_dict/joint_velocity`(rad/s);
  我们用 parquet 的 `actions[:, :7]`(归一化到 [-1,1] 的关节速度指令),
  乘 `--vel-scale`(默认 **0.509**)换算回 rad/s。
  这个系数是**实测**的:把实际关节运动 `diff(state[1:8]) × fps` 对指令做线性回归,
  7 个关节相关系数 0.64~0.78,整体 0.72。
- **episode key**:DROID 用 `recording_folderpath--file_path`;我们用 `episode_index`。

**RLDS 分支的行为没有改变**(除了路径从写死改成参数)。

### 效果

```
250 条 episode(整条被滤掉 1 条 —— ep172,只有 4 帧)
66,463 → 63,909 帧,保留 96.2%,滤掉 3.8%
每条的保留区间数:中位 1,最多 2
```

---

## 五、辅助脚本(4 个新文件,8/5 加的)

| 文件 | 作用 |
|---|---|
| `scripts/franka_preflight_norm_check.py` | 训练前检查我们的动作/状态归一化后是否落在 DROID 的分位数范围内 |
| `runme_preflight_franka.sh` | 跑上面那个检查 |
| `runme_cal_stats_franka.sh` | 用自采数据重算 norm stats(备选方案,最终没用,一直用的 DROID 的) |
| `runme_finetune_franka.sh` | 早期的微调启动命令 |

---

## 上游的几个行为(我们没改,但必须知道)

这些不是我们改的,是 OpenPI 本身的设计,踩过才知道:

### 1. 所谓的 "LoRA 微调",视觉塔是全参数训练的

freeze filter 的正则只覆盖 `.*llm.*`,而 SigLIP 视觉编码器在 `PaliGemma/img/…` 下,**不在冻结名单里**。
从训练日志的参数表实测(冻结参数转 bf16、可训练留 fp32):

| 类别 | 参数量 | 是否训练 |
|---|---|---|
| SigLIP 视觉编码器 | **414.8 M** | ✅ 全量 |
| LoRA 适配器(2B + action expert) | 50.0 M | ✅ |
| 动作/时间投影头 | 2.2 M | ✅ |
| **可训练小计** | **467.0 M(13.7%)** | |
| 语言模型主体 | 2,936.5 M | ❄️ 冻结 |

**89% 的可训练参数是视觉塔,不是 LoRA。** 官方自己的 `pi0_aloha_lora` / `pi0_libero_lora`
也是同一行为,所以是刻意的配方而非疏漏。但对我们 250 条数据的规模,这是过拟合的主要来源。

### 2. action chunk 越界时补最后一帧,而且照常进 loss

实测(282 帧的 episode 0,`action_horizon=16`):

| 采样帧 | 距末尾 | `actions_is_pad` |
|---|---|---|
| 274 | 7 | 后 8 位是 1 |
| 281 | 0 | 后 15 位是 1 |

越界部分用最后一帧的动作重复填充。LeRobot 给了 `actions_is_pad` 掩码,
**但 openpi 不用它**。所以每条 episode 的最后 15 帧都在教模型"到终点就重复最后那个动作"。

### 3. loss 打印走块缓冲,tail 日志会滞后几千步

`train.py` 用 `pbar.write(f"Step {step}: ...")`,重定向到文件后要攒满约 4 KB 才落盘,
每行约 55 字节 → **滞后约 7000 步**。刚起训练时 `grep "^Step"` 是空的,看起来像没在算 loss,
其实一切正常(进度条走 logging 模块,行缓冲,照常刷新)。

**实时看 loss 用 `/data1/vla-reasoning/watch_loss_10task.sh`**(读 wandb 的 offline 数据,每次 log 就落盘)。

### 4. `keep_period` 决定哪些 ckpt 会被长期保留

orbax 的规则是「步数能被 `keep_period` 整除的永久保留,其余只留最近一个」。
**续训时改 `keep_period` 会让已有的、不是新周期倍数的 ckpt 被当成可清理对象删掉。**
我们 20k→50k 续训时特意保持 `keep_period=2000` 不变,只把 `save_interval` 从 2000 调到 4000。

---

## 备份与回退

每个被修改的上游文件都留了日期备份,和原文件同目录:

```
src/openpi/training/config.py.bak-{20260806, 20260809, 20260821, 20260823, 20260824}
src/openpi/training/data_loader.py.bak-20260824
examples/droid/compute_droid_nonidle_ranges.py.bak-20260824
```

回退某一处:`cp <文件>.bak-<日期> <文件>`

看完整改动:

```bash
cd /data1/Franka_RealRobot/openpi
git diff fd7caad -- .          # 只有我们的改动(fd7caad 是上游最后一个提交)
git diff --stat fd7caad -- .
```

> ⚠️ 这个 repo 的属主是 **`zli538`**(别人的 checkout,权限恰好是 777 所以我们能写)。
> 改之前一定要备份,不要做整文件覆盖式的编辑。

---

## 验证记录

| 检查项 | 结果 |
|---|---|
| 五个 config 都能 `get_config()` 加载 | ✅ |
| 不开过滤时数据集长度 | 66,463 ✅ |
| 开过滤后长度 | 63,909 ✅ 与 ranges.json 期望值完全一致 |
| 独立复算的允许集合 vs patch 算出的 keep | **逐元素一致** ✅ |
| 过滤后抽样(首/中/尾) | `actions (16,8)`、`image (3,224,224)`、prompt 正确解析 ✅ |
| 被滤掉的 2,554 帧 | 确认不在可采样集合里 ✅ |
| 扩展后的官方脚本 vs 之前的独立脚本产物 | **逐条完全一致** ✅ |
| FK(用于过滤研究)对标定真值 | 位置误差 0.00 mm,姿态误差 0.01° ✅ |
| 实际训练 | 20k + 续训到 50k 全程无报错,loss 1.591 → 0.0049 ✅ |

---

## 相关文档

| 文档 | 内容 |
|---|---|
| `HOW_TO_ENABLE_IDLE_FILTER.md` | 怎么开启静止帧过滤、完整用法 |
| `data_pipeline/INDEX.md`(labserver) | 全部数据处理脚本清单 |
| `EVAL-交接说明.md` | 真机 eval:ckpt 放哪、前端怎么起 |
| `README.md` | 本交付目录的总说明 |
