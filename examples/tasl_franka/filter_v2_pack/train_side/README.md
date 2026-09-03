# train_side — 训练侧要的全部东西(和生成数据集无关)

两条线,配方相同,只差底座 / stats / 图像几何:`pi05_droid_franka_lora_10task_v2`(pi05-droid 底座,letterbox,DROID stats)和 `pi05_pbc_franka_lora_10task_v2`(PBC 底座,centre-crop,PBC stats)。别混用:PBC 底座预训练时没见过黑边。

| 文件 | 干什么 |
|---|---|
| `config_pi05_droid_franka_lora_10task_v2.py` | **主线 TrainConfig 原文**(labserver config.py:1230-1276),贴进 `_CONFIGS` 列表即可;换 pbc 底座只改 `weight_loader`(和 `assets_dir`,若新 ckpt 自带 norm stats) |
| `data_config_LeRobotRLinfDROIDDataConfig.py` + `rlinf_franka_droid.py` | config 里 `data=` 用的 DataConfigFactory 和它的 repack transform(state[8]=[grip,q0..q6] 拆开、image/extra_view_image 两路相机、关节速度动作无 delta)。TASL1 的 openpi 若已有 `pi05_droid_franka_lora`(commit 45c420d)就已经带了 |
| `data_loader_subsetdataset.patch` / `data_loader.py` | 让 LeRobot 路径认 `filter_dict_path` 的补丁(+60 行)/ 打好补丁的整文件。`cd openpi && git apply train_side/data_loader_subsetdataset.patch` |
| `config_pi05_pbc_franka_lora_10task_v2.py` | **pbc 版 TrainConfig**(labserver 唯一的 pbc config):配方和 v2 完全一样,只换底座(PBC 199999 EMA params)、norm stats(PBC 自带 `Devon018/Franka-Datasets-v2`)、图像几何(centre-crop),action_horizon 15。附 `PBC_BASE_*` 常量 |
| `data_config_LeRobotRLinfPbcDataConfig.py` + `rlinf_franka_pbc.py` | pbc 的 DataConfigFactory(= DROID 版 + `PbcCenterCropImages`,训练和 serve 同一段代码)和 crop helper |
| `make_pbc_dataset.py` | 把 letterbox 数据集转成 centre-crop 几何的 `*_pbc` 数据集(`--src tasl_fr3_10task_v2 --dst tasl_fr3_10task_v2_pbc`);帧索引不变,尾 10 帧 json 直接复用。labserver 上已跑 |
| `launch_train_pbc_10task_v2.sh` | pbc 启动脚本 |
| `launch_train_10task_v2.sh` | 启动脚本(labserver 路径,按机器改 `cd`、`HF_LEROBOT_HOME`、`PY`、json 路径);`GPUS= MEMFRAC= EXP=` 可覆盖,`setsid nohup ... &` 分离跑 |
| `watch_train_10task_v2.sh` | 看 train.py 进度条 / Step 行的 watcher,`tail -f` 最新 log |

训练配方(主线 `pi05_droid_franka_lora_10task_v2`):π0.5 pi05=True, action_dim 32, action_horizon 16, paligemma `gemma_2b_lora` + action expert `gemma_300m_lora`;
底座 `gs://openpi-assets/checkpoints/pi05_droid/params`;batch 32;30k 步;AdamW clip 1.0;cosine warmup 3000 → peak 2.5e-5 → 2.5e-6(decay_steps=30k 是整条曲线长度,含 warmup);
无 EMA;每 5000 步存 ckpt 且永久保留(`save_interval=keep_period=5000`,orbax `max_to_keep=1`);log 100 步;wandb online。

norm stats:**复用 DROID 原始 stats**(`assets_dir=gs://openpi-assets/checkpoints/pi05_droid/assets`, `asset_id="droid"`),不本地重算,之前的 5k / 10task 版也一样。
我们的 action 在 DROID 分位数下被压到 ±0.5 左右,但布局 / 符号 / 夹爪 [0,1](0 开 1 合)约定都对得上,只是线性缩放,不是"卡"的原因。

没有 `data_loader` 补丁的后果:`filter_dict_path` 被 LeRobot 路径静默忽略,训练用的是完整导出集(少"段尾 10 帧不作起点"这一步),数据集本身不受影响。
补丁语义:帧不删,只是不做 chunk 起点(仍可作为别的 chunk 的目标);`_nonidle_indices` 用 `episodes.jsonl` 的 length 把 `{episode: [[s,e),...]}` 映成全局帧下标。
