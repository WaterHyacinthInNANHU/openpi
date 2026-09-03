# Real-robot finetunes on AXIS Server 3 (8x A100-80GB)

Everything the TASL Franka real-robot finetunes need on the 8-GPU box, so a run is reproducible from
this repo plus the paths below. Server-side layout:

| path | what |
|---|---|
| `/localdisk/tasl_franka_cfg/openpi` | this repo, branch `tasl-franka-cfg` (main + `RealWorldFR3` + these configs) |
| `/localdisk/tasl_franka_cfg/lerobot_local` | the local lerobot fork `pyproject.toml` points at (`../lerobot_local`) |
| `/localdisk/tasl_franka_finetune/checkpoints/<config>/<exp>/` | checkpoints (PBC and CFG runs side by side) |
| `/localdisk/tasl_franka_finetune/filters/nonidle_ranges_tail10.json` | idle-frame filter (== `examples/tasl_franka/filter_v2_pack/reference/`) |
| `/localdisk/dihong_workspace/lerobot_home/ZhixuLi/tasl-fr3-10task-pbc-v2` | 10-task dataset, 392 episodes, centre-cropped |
| `/localdisk/dihong_workspace/runs/ckpts/pi05_axis_droid_cotrain/cotrain_sim25/199999` | PBC base |
| `/localdisk/dihong_workspace/runs/ckpts/pi05_axis_droid_cotrain_cfg/cotrain_cfg_phase/199999` | CFG base |

The box is a shared account: check `nvidia-smi` and the notes under `~/axis/` before launching, and never
kill someone else's trainer. `launch_*.sh` refuse to start if any compute process is already on the GPUs.

## Runs

| script | config | base | what differs |
|---|---|---|---|
| `launch_pbc_10task_v2.sh` | `pi05_cotrain_franka_lora_10task_pbc_v2` | cotrain_sim25 (PBC) | -- |
| `launch_cfg_10task_v2.sh` | `pi05_cotrain_franka_lora_10task_cfg_v2` | cotrain_cfg_phase (CFG) | init weights + `"\nQuality: 5"` prompt tag |

Shared recipe: LoRA (gemma_2b_lora + gemma_300m_lora), global batch 64 on 8 GPUs, 16k steps, AdamW clip 1.0,
cosine 2.5e-5 -> 2.5e-6 with 1,600 warmup steps, no EMA, seed 42, checkpoints every 2k steps.

Before a CFG run, confirm the prompts are tagged (~80.75% of samples, the rest are the unconditional branch):

```bash
.venv/bin/python examples/tasl_franka/on_a100/check_quality_prompts.py --n 2000
```

Serving a CFG checkpoint needs `--quality-tag 5` (and optionally `--guidance-scale`) on `scripts/serve_policy.py`;
without the tag the policy is queried on a prompt it saw only ~19% of the time.

## Environment

```bash
cd /localdisk/tasl_franka_cfg && git clone https://github.com/WaterHyacinthInNANHU/openpi.git && cd openpi
git checkout tasl-franka-cfg
cp -r /localdisk/dihong_workspace/lerobot_local ../lerobot_local     # pyproject: lerobot = { path = "../lerobot_local" }
source /localdisk/dihong_workspace/env.sh; export no_proxy="$no_proxy,download-r2.pytorch.org"
GIT_LFS_SKIP_SMUDGE=1 uv sync
```
