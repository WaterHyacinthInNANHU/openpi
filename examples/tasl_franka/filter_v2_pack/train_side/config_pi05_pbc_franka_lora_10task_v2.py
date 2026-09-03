# pi05_pbc_franka_lora_10task_v2 (labserver config.py, 2026-08-25): 配方 = pi05_droid_franka_lora_10task_v2, 只换底座/stats/图像几何。
# 依赖 LeRobotRLinfPbcDataConfig (config.py:487) + openpi/policies/rlinf_franka_pbc.py + PBC_BASE_* 常量 (config.py:524-527)。
    # PBC (in-house pi0.5 base `axis_pi05_droid_plainbc_v1`, centre-crop image geometry, PBC norm stats).
    # Never mix with the `pi05_droid_franka_*` (letterbox, DROID stats) family. Pipeline: data_pipeline/on_labserver/pbc/.
    #
    TrainConfig(
        # pbc_10task_v2 (2026-08-25): 配方 = pi05_droid_franka_lora_10task_v2 (30k 步, warmup 3k, cosine 2.5e-5→2.5e-6, LoRA, 无 EMA, 每 5k 存),
        # 只换三样: 底座 = PBC 199999 (EMA 权重), norm stats = PBC 自带 (Devon018/Franka-Datasets-v2), 图像 = centre-crop 几何
        # (数据 tasl_fr3_10task_v2_pbc, 由 make_pbc_dataset.py 从 tasl_fr3_10task_v2 生成, 帧索引不变, 尾 10 帧 json 直接复用)。
        # action_horizon=15 跟 PBC 预训练一致 (droid_lora 家族是 16)。
        name="pi05_pbc_franka_lora_10task_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRLinfPbcDataConfig(
            repo_id="franka/tasl_fr3_10task_v2_pbc",
            base_config=DataConfig(
                prompt_from_task=True,
                filter_dict_path="/data1/Franka_RealRobot/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json",
            ),
            assets=AssetsConfig(assets_dir=PBC_BASE_ASSETS_DIR, asset_id=PBC_BASE_ASSET_ID),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(PBC_BASE_PARAMS),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
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
PBC_BASE_STEP_DIR = "/data1/Franka_RealRobot/checkpoints/axis_pi05_droid_plainbc_v1/199999"
PBC_BASE_PARAMS = f"{PBC_BASE_STEP_DIR}/params"
PBC_BASE_ASSETS_DIR = f"{PBC_BASE_STEP_DIR}/assets"
PBC_BASE_ASSET_ID = "Devon018/Franka-Datasets-v2"  # <assets>/<asset_id>/norm_stats.json

