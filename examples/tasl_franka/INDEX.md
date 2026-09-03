# TASL Franka 数据处理脚本清单

清点于 2026-08-24。**脚本分散在两台机器上,而且大部分原本在 `/tmp` 里(重启即丢)**,
已归档到持久位置并拷贝一份到本地。

| | 持久位置 |
|---|---|
| tasl-1(数据在这台,绝大多数处理脚本也在) | `~/work/tasl_data_pipeline/` |
| labserver(训练在这台) | `/data1/vla-reasoning/`(原有)+ `/data1/vla-reasoning/tasl_pipeline/`(新归档) |
| 本地副本 | 本目录 |

---

## tasl-1 —— 数据处理主力

### 01_survey_audit
| 脚本 | 干什么 |
|---|---|
| `ds_survey.py` | 扫 11 个采集目录,列出每个的 episode 数/帧数/fps/prompt/特征列 |
| `ds_tasks.py` | 查 `tasks.jsonl` 与 `episodes.jsonl` 的 task 对应关系(用它发现了 T1-a 的 typo、T1-b 的双 task、T4-b 占位符) |
| `ds_check2.py` | 核对 11 个数据集的 state/action 维度、图像尺寸、chunk 布局是否一致(合并可行性的依据) |
| `audit_data.py` | 审计"训练前做了什么预处理":帧数是否等于源数据之和、`is_success`/`intervene_flag` 分布、静止帧比例 |
| `audit2.py` | episode 长度分布、异常短轨迹、低速帧占比 |

### 02_merge
| 脚本 | 干什么 |
|---|---|
| `prompts.json` | **10 个 task 的最终 prompt 表**(逐个看画面重写过),合并脚本读它 |
| `merge_10task.py` | 把 11 个数据集合并成 250 条 / 10 task。三件关键事:`task_index` 按数据集偏移重映射、`episode_index` 重编号、保留 parquet 内嵌的 `huggingface` schema 元数据 |
| `verify_merge.py` | 逐条校验 250 个 parquet:task_index 解析出的 prompt 是否正确、index 是否连续、hf 元数据是否还在、图像能否解码 |
| `fix_meta.py` | 只改 `tasks.jsonl` / `episodes.jsonl` 的 prompt(改 T4-a green→orange 时用的,不动 4.6 GB 的 parquet) |

### 03_filter_study
| 脚本 | 干什么 |
|---|---|
| `fk.py` | **FR3 正运动学**(modified DH),`fk(q)` 返回 base→flange 的 4×4。用 5 月标定 anchor 验过,误差 0.00 mm / 0.01° |
| `idle_check.py` | 检验 `actions` 是不是实测关节速度(发现列序 state=[grip,q0..q6] vs action=[dq0..dq6,grip] 不一样) |
| `idle2.py` | 单条 episode 的带时延相关分析 |
| `idle3.py` | 对齐列序后重算相关(+0.72)+ 用实测 Δq 重新统计静止帧 |
| `idle4.py` | 静止帧在轨迹中的位置分布、头尾连续静止段、各 task 静止比例 |
| `filters.py` | 方法 A:笛卡尔 Δ位移/Δ转角分布 + 20% 分位阈值 |
| `filters2.py` / `filters3.py` | 方法 B(SCIZOR 式 state-action 去重)+ 方法 C(ISR 等运动学距离重采样) |
| `compare.py` | 三种过滤删掉的帧集合交叉对比 |
| `full_stat.py` | 方法 A + 段级规则在全量 250 条上的影响 |
| `variant.py` | 段级规则的几种变体对比(min_non / min_final) |
| `grip.py` | **发现夹爪 bug**:纯笛卡尔判据误删了 27.6% 的夹爪动作帧 |
| `final_filter.py` | 加夹爪判据 + 抓取保护窗后的最终对比(误删降到 3.9%) |

### 04_visualization
| 脚本 | 干什么 |
|---|---|
| `viz_10task.py` | 每个 task 抽一条轨迹渲染成左右双视角视频 |
| `contact.py` / `strip.py` | 每个 task 的多帧联系表 / 单条轨迹的帧条(看 prompt 对不对时用的) |
| `plot_loss.py` | 从 wandb offline 数据画 loss + grad norm 曲线 |
| `viz_filter.py` | 过滤前后对照视频:被删帧标红 + 底部保留/删除时间轴 |
| `viz_filter2.py` | 同上,**含夹爪判据和抓取保护窗**(推荐用这个) |

### 05_hf_upload / 06_video_format / 07_frontend
| 脚本 | 干什么 |
|---|---|
| `hf_push.py` | 把合并后的数据集推到 HF |
| `hf_meta_push.py` | 只补传两个 meta 文件(改 prompt 时用) |
| `to_video_ds.py` | 图像内嵌 parquet → h264 mp4(给 LeRobot 可视化器用) |
| `build_video_meta.py` | 生成视频型的 `info.json` / stats(键名改成 `observation.images.*`) |
| `verify_video_ds.py` | 逐条比对 250 个 parquet 行数与 500 个 mp4 的实际解码帧数 |
| `push_video_ds.py` | 推视频版数据集 |
| `update_tasks.py` | 把 eval 前端的任务库替换成训练时用的 10 条 prompt |
| `fix_layouts.py` | 清掉任务库里指向不存在 layout 的引用 |

---

## labserver —— 训练侧

### 原有(`/data1/vla-reasoning/`)
| 脚本 | 干什么 |
|---|---|
| `launch_train_10task.sh` | **当前这轮 50k 的启动脚本**(已改成 `--resume`) |
| `watch_loss_10task.sh` | 实时读 loss。**不要 tail 文本日志** —— openpi 的 loss 行走块缓冲 stdout,会滞后几千步 |
| `launch_train_v2.sh` / `launch_train_v2_calibra50.sh` | 8/9 两轮的启动脚本 |
| `fix_idlefiltered_timestamps.py` | 8/9 那次删帧后重排 timestamp/frame_index/index(**注意:这个做法会把时间缺口抹平,导致 action chunk 静默跨越切口**) |
| `hf_upload.sh` / `auto_launch_v2_train.sh` | 早期的上传/自动启动 |
| `calibra/analyze.py`、`export_calibra50.py`、`taskdist.py` | Calibra 子集实验 |

### 新归档(`/data1/vla-reasoning/tasl_pipeline/`)
| 脚本 | 干什么 |
|---|---|
| `hf_pull.py` | 从 HF 拉数据集到 `HF_LEROBOT_HOME` |
| `push_ckpts.py` | 把 10 个 ckpt 传到 HF(只传 params+assets,不传 train_state) |
| `new_cfg.py` | 生成 10task 的 TrainConfig 文本块 |
| `apply_official.py` | 把 openpi 官方 idle 判据套到我们数据上实测 |
| `reverse_filter.py` | 从 8/9 的过滤前后数据集反推当时的判据(结论:删的不是静止帧) |
| `chunk_test.py` | **实测 action chunk 越界时的行为**(末帧重复填充 + `actions_is_pad`,但 openpi 不用这个掩码) |

### 训练配置
```
/data1/Franka_RealRobot/openpi/src/openpi/training/config.py     ← 我们的 config 追加在这里
    第 944 行  pi05_droid_franka_lora            (v0, 20k)
    第 996 行  pi05_droid_franka_lora_5k         (v2, 5k)
    第 1042 行 pi05_droid_franka_lora_5k_calibra50
    第 1088 行 pi05_droid_franka_lora_10task     (当前, 50k)
备份: config.py.bak-{20260806, 20260809, 20260821, 20260823}
```
⚠️ 这个文件属主是 **zli538**(别人的 checkout,权限恰好可写),改之前务必备份。

---

## 数据/产物位置

| | 路径 |
|---|---|
| 采集原始数据(11 个目录) | tasl-1 `~/rlinf_data/datasets/`(root 属主) |
| 合并后数据集 | tasl-1 `~/work/merged/tasl_fr3_10task_250ep/` |
| 视频版数据集 | tasl-1 `~/work/merged/tasl_fr3_10task_250ep_video/` |
| 训练用数据集 | labserver `/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep/` |
| checkpoint | labserver `/data1/Franka_RealRobot/checkpoints/pi05_droid_franka_lora_10task/pi05_droid_franka_lora_10task_v0/` |
| HF 数据集 | `Litian2002/tasl-fr3-10task-250ep`(图像版)· `-video`(可视化版) |
| HF checkpoint | `Litian2002/pi05-droid-franka-lora-10task` |

---

## 待办

1. **过滤脚本还没写**(只有研究脚本)。定了方案再实现:建议按「段级导出成独立 episode」,
   避免 action chunk 跨越切口 —— 详见和 `chunk_test.py` 相关的结论。
2. 这些脚本多数是一次性分析脚本,路径写死、没有参数化。要长期用得整理成一个带 CLI 的包。

---

## 2026-08-24 晚追加:v2 过滤

| 脚本 / 文件 | 干什么 |
|---|---|
| `on_labserver/filter_v2_study/dummy_census.py` | dummy 帧普查:指令零值 / 指令小于阈值 / 实测静止 / 官方判据,各占多少、头尾段、分 task |
| `on_labserver/filter_v2_study/deadband.py` | 指令 vs 实测速度的时延(lag 0/1/2)与死区表:`\|cmd\|<0.13` 基本不动,`≥0.2` 才线性 |
| `on_labserver/filter_v2_study/variants.py` / `variantB_grip.py` | 6 种判据套官方段级规则的删除比例 / 夹爪误删 / 残留静止段 |
| `on_labserver/launch_train_10task_v2.sh` | **10task_v2 启动脚本(未启动)**,GPUS / MEMFRAC / EXP 可覆盖 |
| `/data1/Franka_RealRobot/filters/tasl_fr3_10task_250ep/nonidle_ranges_v2.json` | v2 ranges(实测速度 + 夹爪保护 ±5),66463 → 58760;**已被实体数据集取代,留作 A/B** |
| `on_labserver/export_10task_v2.py` | **把 v2 保留段物理导出成新数据集** `lerobot_home/franka/tasl_fr3_10task_v2`(392 ep / 62680 帧),原版不动 |
| openpi `config.py:1235` `pi05_droid_franka_lora_10task_v2` | **主线**:tasl_fr3_10task_v2(物理删 dummy)+ 尾 10 帧 json(58760 起点),30k 步、每 5k 存、warmup 3k、官方 cosine lr |
| `filter_v2_pack/` + `filter_v2_pack.tar.gz` | **v2 过滤打包版(给 TASL1)**:自包含 export / 尾10 json / verify 三脚本 + `run_all.sh` + README,只依赖 numpy/pyarrow/pillow;冒烟输出与正式 v2 数据集逐字节一致 |

详见 `HOW_TO_ENABLE_IDLE_FILTER.md` 末尾 "v2 加强版过滤"。
