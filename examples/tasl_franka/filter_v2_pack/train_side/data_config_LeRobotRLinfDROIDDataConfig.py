# LeRobotRLinfDROIDDataConfig (labserver openpi/src/openpi/training/config.py:459), pi05_droid_franka_lora* 共用。
# 依赖 openpi/policies/rlinf_franka_droid.py 的 RLinfFrankaDroidRepack (同目录附上)。
class LeRobotRLinfDROIDDataConfig(DataConfigFactory):
    """DROID data config for LeRobot datasets collected by the RLinf FR3 bench.

    Same pipeline as LeRobotDROIDDataConfig, but the repack step is a custom
    transform (RLinfFrankaDroidRepack) because the RLinf writer stores the
    8-D state as one column ([grip, q0..q6]) that must be split, and uses
    image/extra_view_image as camera keys. No second exterior view.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[rlinf_franka_droid.RLinfFrankaDroidRepack()])
        # Joint *velocity* actions (DROID-native): no delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
