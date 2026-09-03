# PBC pipeline (in-house pi0.5 base `axis_pi05_droid_plainbc_v1`)

Everything for the PBC base is namespaced `pbc` and kept apart from the `pi05_droid_franka_*`
(letterbox + DROID norm stats) family:

| what | where |
|---|---|
| base ckpt (EMA params + norm stats) | `/data1/Franka_RealRobot/checkpoints/axis_pi05_droid_plainbc_v1/<step>/` (use 199999) |
| image transform + crop helper | `openpi/src/openpi/policies/rlinf_franka_pbc.py` |
| data config | `LeRobotRLinfPbcDataConfig` in `openpi/src/openpi/training/config.py` |
| configs | `pi05_pbc_base` (serve-only, zero-shot), `pi05_pbc_franka_lora_10task` (LoRA fine-tune) |
| dataset builder | `make_pbc_dataset.py` → `lerobot_home/franka/tasl_fr3_10task_250ep_pbc` |
| train launcher | `launch_train_pbc_10task.sh` |
| ckpts | `/data1/Franka_RealRobot/checkpoints/pi05_pbc_franka_lora_10task/<exp>/` |

## Geometry (the whole reason this exists)

* pi05_droid: `resize_with_pad` — aspect-preserving letterbox, black bars, full FOV.
* PBC base: **full-height centre-square crop, then resize to 224** (AXIS-Bench round-3
  `center_crop`; same as RLinf `image_resize_mode: crop`). No bars, central 56 % of a 16:9 frame.

`PbcCenterCropImages` is inserted into `data_transforms.inputs` right after `DroidInputs`, so it runs
at train AND serve time; a square input is a no-op. That is what guarantees parity:

```
serve:  ZED 1280x720 ──crop 720x720──► ResizeImages ──► 224x224
train:  *_pbc frame 224x224 (already square, no-op) ──► ResizeImages (no-op) ──► 224x224
```

The RLinf openpi dashboard needs **no change** — it already sends full 1280x720 frames; the crop is
server-side and follows the config. (Its "policy view" preview tiles still show the letterbox; only
cosmetic.)

## Dataset

The 10-task data was collected with `pad`, so only 224x224 letterboxes exist (content rows 49–174).
`make_pbc_dataset.py` un-pads, crops the central 126x126 and LANCZOS-upsamples to 224. Geometry is
exact; resolution is a 1.78x upsample of what the live path sees. Recorded in `meta/info.json →
pbc_geometry`. Frame/episode/task indices are untouched, so `nonidle_ranges*.json` files of the source
apply verbatim (copied into `meta/`).

```
/data1/Franka_RealRobot/openpi/.venv/bin/python data_pipeline/on_labserver/pbc/make_pbc_dataset.py \
    --src /data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep \
    --dst /data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep_pbc
```

## Train / serve

```
setsid nohup bash /data1/Franka_RealRobot/data_pipeline/on_labserver/pbc/launch_train_pbc_10task.sh > /tmp/train_pbc_10task.log 2>&1 < /dev/null &

# zero-shot PBC base
serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_pbc_base \
    --policy.dir=/data1/Franka_RealRobot/checkpoints/axis_pi05_droid_plainbc_v1/199999
# fine-tuned
serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_pbc_franka_lora_10task \
    --policy.dir=/data1/Franka_RealRobot/checkpoints/pi05_pbc_franka_lora_10task/<exp>/<step>
```

## Differences vs `pi05_droid_franka_lora_10task_v2` (deliberate)

1. init = PBC base 199999 EMA params; 2. norm stats = PBC mixture stats (`Devon018/Franka-Datasets-v2`),
same "reuse the base's stats" rule as the DROID advice; 3. `action_horizon=15` (matches PBC pretrain
and pi05_droid; the droid-lora family used 16); 4. centre-crop geometry + `*_pbc` dataset.
Optimiser / LR / LoRA / batch / steps are identical.

## Provenance of the geometry (verified 2026-08-25)

* Implementation the PBC pretrain ran: openpi fork `WaterHyacinthInNANHU/openpi` branch
  `box/server2-heldout-d8`, `src/openpi/policies/axis_franka_policy.py::_center_crop_square`
  ("crop the WIDER axis to the shorter one, centred; no-op on a square frame"), applied by
  `AxisFrankaInputs(center_crop=True)` to **both cameras identically**, before `ResizeImages(224,224)`.
* That the plain-BC co-train arm (= this ckpt) had it on: AXIS-Bench branch `myan/cotrain-phase`,
  `docs/reports/2026-08-24-cotrain-bc-audit.md` finding #2 — "DROID images are centre-cropped to 56 %
  of their width. Stored 180×320; `center_crop=True` … also crops the DROID half to 180×180 …
  self-consistent only if serving uses this config's own transforms." Same audit confirms horizon 15.
* `pbc_center_square` here is the same arithmetic; parity between the serving path (1280×720 live)
  and the dataset builder (un-padded 224×126) was checked pixel-wise on a synthetic gradient
  (mean |diff| 0.08/255).

## 2026-08-25 12:50 收敛为一个 config

用户拍板:`pi05_pbc_base` / `pi05_pbc_franka_lora_10task` 两个 TrainConfig 已删(config.py 备份 `.bak-20260825-1250`),
只留 **`pi05_pbc_franka_lora_10task_v2`** = `pi05_droid_franka_lora_10task_v2` 的配方(30k / warmup 3k / cosine 2.5e-5→2.5e-6 / LoRA / 无 EMA / 每 5k 存)
+ PBC 底座 199999 + PBC norm stats + centre-crop 几何,数据 `tasl_fr3_10task_v2_pbc`(`make_pbc_dataset.py` 从 `tasl_fr3_10task_v2` 生成,帧索引不变,
`filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json` 直接复用),action_horizon 15。
`LeRobotRLinfPbcDataConfig` / `rlinf_franka_pbc.py` / `tasl_fr3_10task_250ep_pbc` 数据集 / 本目录脚本都保留。
启动脚本改名 `launch_train_pbc_10task_v2.sh`(wandb 默认 online)。zero-shot serve 老底座的话用 `--policy.config=pi05_pbc_franka_lora_10task_v2 --policy.dir=<PBC 199999>` 即可(data transforms 相同)。
