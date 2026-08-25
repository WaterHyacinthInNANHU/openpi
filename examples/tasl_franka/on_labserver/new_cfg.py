    TrainConfig(
        # v4 (2026-08-21): 10 个 task 合并数据集(T1-a/b … T5-a/b,每个 task 25 条,
        # 共 250 条 / 66463 帧)。prompt 已逐个对照实际画面重写。
        # 20k 步 ≈ 9.6 epoch(此前 5k 步在 4067 帧上是 39 epoch)。
        # 其余超参与 pi05_droid_franka_lora_5k 完全一致。
        name="pi05_droid_franka_lora_10task",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRLinfDROIDDataConfig(
            repo_id="franka/tasl_fr3_10task_250ep",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
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
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        checkpoint_base_dir="/data1/Franka_RealRobot/checkpoints",
        save_interval=2_000,
        keep_period=2_000,
        log_interval=100,
    ),
