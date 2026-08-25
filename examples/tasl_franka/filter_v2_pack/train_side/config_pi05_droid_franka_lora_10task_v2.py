# pi05_droid_franka_lora_10task_v2 (labserver openpi/src/openpi/training/config.py, 2026-08-25)
# 与 pi05_droid_franka_lora 的差别只有: repo_id / filter_dict_path / 30k 步 / warmup 3k / save 5k。
# norm stats 用 DROID 原始的 (assets_dir=gs://openpi-assets/checkpoints/pi05_droid/assets, asset_id=droid), 不本地重算。
# 换 pbc 底座时改 weight_loader (和 assets_dir, 若新 ckpt 自带 norm stats)。
    TrainConfig(
        # 10task_v2 (2026-08-25): 数据 = tasl_fr3_10task_v2 (10task 按 v2 判据物理删掉 dummy 静止段, 每段一条 episode,
        # 392 ep / 62680 帧) + 官方 ranges 模式的尾砍: 每条末尾 10 帧不做 chunk 起点 (nonidle_ranges_tail10.json), 58760 起点。
        # 判据/导出/验证见 data_pipeline/HOW_TO_ENABLE_IDLE_FILTER.md "v2"。30k 步, warmup 10%, 官方 cosine lr, LoRA, DROID stats。
        # 换新 pretrain ckpt 时只改 weight_loader。
        name="pi05_droid_franka_lora_10task_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRLinfDROIDDataConfig(
            repo_id="franka/tasl_fr3_10task_v2",
            base_config=DataConfig(
                prompt_from_task=True,
                filter_dict_path="/data1/Franka_RealRobot/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json",
            ),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        checkpoint_base_dir="/data1/Franka_RealRobot/checkpoints",
        save_interval=5_000,
        keep_period=5_000,
        log_interval=100,
    ),
