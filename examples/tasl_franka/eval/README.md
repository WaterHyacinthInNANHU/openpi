# TASL Franka FR3 — real-robot evaluation stack

Everything needed to evaluate a π0.5 checkpoint on the TASL FR3 bench (tasl-1 + NUC + two ZED cameras),
served from **this openpi checkout**. It used to live in the lab RLinf fork under `tasl/`; the data-collection
side (collect dashboard, teleop, GELLO/Pico, NUC controller image, dataset export/filter tools) stays there.

```
eval/
├── dashboards/openpi.py     the evaluation console (:8003): cameras, tasks/layouts, checkpoint switcher,
│                            policy loop, episode recorder; talks to serve_policy on :8000
├── dashboards/rlinf.py      older RLinf-container eval console (eval.sh path)
├── dashboards/{layout_store,task_store}.py   task + layout registry, SHARED with the collect dashboard
├── rtc/                     real-time chunking (async inference + guided inpainting); rtc/scripts/serve_policy.py
│                            is the "Load with RTC" serve variant (wraps this repo's scripts/serve_policy.py)
├── clients/droid_client.py  zerorpc client to the NUC polymetis controller (:4242)
├── tools/                   layout capture from datasets, OOD stats, rollout publishing, ZED viewer, H.264 writer
├── launch/                  start_openpi.sh / start_openpi_full.sh / openpi-stop.sh / nuc-restart.sh / eval.sh …
├── tests/                   RTC unit tests
└── docs/                    EVAL.md, ARCHITECTURE.md, CONTROLLER.md
```

## Paths (all overridable, none inside the repo)

| what | env | default |
|---|---|---|
| openpi checkout used for serving (`scripts/serve_policy.py`, `.venv`, `packages/openpi-client`) | `OPENPI_DIR` | this checkout (3 dirs up) |
| operator home even under `sudo` | `TASL_HOME` | `$SUDO_USER`'s home |
| mutable bench data: `eval_episodes/`, `saved_demo/`, `logs/`, `_dashboard_home.json`, `VLA-PatchLen-cp/` | `TASL_DATA_DIR` | `~/tasl_data` |
| task registry (shared with collect.py) | `TASL_TASKS_STORE` | `~/rlinf_data/tasks_store.json` |
| layout masks (shared with collect.py) | `RLINF_LAYOUT_DIR` | `~/rlinf_data/layouts` |
| checkpoints scanned by the switcher | — | `~/ckpts/<group>[/<step>]/params` (+ `config.txt` = serve config name) |
| serve log / gripper audit | `SERVE_LOG`, `GRIP_AUDIT_PATH` | `~/.tasl/` of whoever runs the dashboard |

First-time migration on tasl-1 (data stays where it is; the new defaults just point at it):

```bash
mkdir -p ~/tasl_data && cd ~/tasl_data
ln -s ~/RLinf/tasl/eval_episodes eval_episodes
ln -s ~/RLinf/saved_demo         saved_demo
ln -s ~/RLinf/tasl/VLA-PatchLen-cp VLA-PatchLen-cp
mv ~/RLinf/tasl/tasks_store.json ~/rlinf_data/tasks_store.json && ln -s ~/rlinf_data/tasks_store.json ~/RLinf/tasl/tasks_store.json
```

## Serve configs (src/openpi/training/config.py)

| checkpoint group | `config.txt` | image mode | prompt |
|---|---|---|---|
| `pi05_droid`, `pi05_droid_franka_lora_*` (DROID base) | `pi05_droid` / `pi05_droid_franka_lora` | pad (letterbox) | task text |
| `cotrain_pbc_v2`, `cotrain_cfg_v2` (AXIS cotrain bases) | `pi05_cotrain_franka_serve` | **crop** (tick `pbc`) | task text; CFG: tick `CFG quality` → `"\nQuality: N"` appended |

## Run

```bash
cd <openpi>/examples/tasl_franka/eval
sudo bash launch/start_openpi_full.sh    # NUC container → FCI prompt → serve + dashboard (root)
sudo bash launch/start_openpi.sh         # serve (pi05_droid) + dashboard only
sudo bash launch/openpi-stop.sh
# user-mode dashboard (serve is spawned per selected checkpoint):
PYTHONPATH=$PWD:$HOME/.local/lib/python3.10/site-packages:<openpi>/packages/openpi-client/src \
  /usr/bin/python3 dashboards/openpi.py --port 8003 --policy-host 127.0.0.1 --policy-port 8000
```

The dashboard's dependencies (Flask, zerorpc, pyzed …) are the system python 3.10 + `~/.local`; the policy server
runs in this checkout's `.venv` (JAX). Building that venv on tasl-1 is slow (WiFi) — do it once, off-hours.

## Known failure modes (2026-09-03/04)

| symptom | cause | fix |
|---|---|---|
| `spawn failed: Permission denied: .../serve_ckpt.log` | dashboard alternately run as root and as the user | logs now live in `~/.tasl/` per identity; nothing to do |
| OOD layout masks 404 | layout files created by a root-run dashboard (mode 600) | `sudo chown -R $USER ~/rlinf_data` |
| eval hangs, `ZED … grab FAILURE` | ZED 2i USB dropped | replug the camera, restart the dashboard |
| robot/gripper unavailable after a robot reboot | polymetis driver in the NUC container died | Desk: FCI active → dashboard "NUC restart" |
| `eval running, stop first` with serve idle | Start pressed with no serve (client retried forever) | guarded: Start is refused until a checkpoint is ready |
