# LeRobotRLinfPbcDataConfig (labserver config.py:487)
@dataclasses.dataclass(frozen=True)
class LeRobotRLinfPbcDataConfig(DataConfigFactory):
    """PBC variant of LeRobotRLinfDROIDDataConfig (see openpi/policies/rlinf_franka_pbc.py).

    Identical repack / DroidInputs / DroidOutputs, plus `PbcCenterCropImages` inserted right after
    DroidInputs and therefore BEFORE the model transforms' `ResizeImages(224, 224)`.  Because the
    crop sits in `data_transforms`, it is applied at train time and at serve time alike; a square
    input (the `*_pbc` dataset frames) is a no-op, a live 1280x720 ZED frame becomes 720x720.

    Use this ONLY with the PBC base (`axis_pi05_droid_plainbc_v1`) and a `*_pbc` dataset.  Pointing
    it at the letterboxed `tasl_fr3_10task_250ep` would crop a 224x224 letterbox (no-op) and train
    a centre-crop model on padded frames -- a silent geometry mismatch.
    """

    # Set False only for A/B tests against the letterbox geometry.
    center_crop: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[rlinf_franka_droid.RLinfFrankaDroidRepack()])
        inputs: list[_transforms.DataTransformFn] = [droid_policy.DroidInputs(model_type=model_config.model_type)]
        if self.center_crop:
            inputs.append(rlinf_franka_pbc.PbcCenterCropImages())
        # Joint *velocity* actions (DROID-native, same as the PBC pretrain): no delta transform.
        data_transforms = _transforms.Group(inputs=inputs, outputs=[droid_policy.DroidOutputs()])
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
