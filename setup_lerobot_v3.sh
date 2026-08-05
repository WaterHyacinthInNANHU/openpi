#!/bin/bash
# Overlay LeRobot v3.0 into the openpi venv so it can read v3.0 datasets
# (e.g. Devon018/Franka-Datasets-v2, codebase_version "v3.0").
#
# WHY THIS IS A SCRIPT AND NOT pyproject/uv.lock:
#   openpi (ours + upstream main) pins LeRobot v2.1. LeRobot's first v3.0 release
#   is v0.4.0, whose CORE hard-deps rerun-sdk (viz only). rerun-sdk forces numpy>=2
#   (openpi needs numpy<2) AND ships only manylinux_2_31 wheels, but this cluster is
#   glibc 2.28 (manylinux_2_28) with no rerun sdist -> rerun is UNINSTALLABLE here.
#   So a full `uv sync` of lerobot v0.4.4 can never resolve/install. We therefore
#   overlay lerobot v0.4.4 with --no-deps (skipping rerun) plus the one extra import
#   dep it needs. datasets 3.6 + av 17 + torchcodec 0.4.0 already in the venv are
#   sufficient to READ v3.0.
#
# RE-RUN THIS after any `uv sync`, which reverts lerobot to ../lerobot_local (v2.1).
#
# RUNTIME: v3.0 decodes video via torchcodec, which needs FFmpeg .so on
# LD_LIBRARY_PATH. Use `module load ffmpeg/4.3.0` (decoder4). Do NOT use ffmpeg/5.0:
# it is 5.0 (not >=5.1) and torchcodec's decoder5 fails with
# `undefined symbol: av_channel_layout_default`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIRTUAL_ENV="${HERE}/.venv"

uv pip install --no-deps 'lerobot @ git+https://github.com/huggingface/lerobot@v0.4.4'
uv pip install 'accelerate==1.14.0'

# lerobot 0.4.4 requires datasets>=4.0,<5 (it emits the new 'List' feature type; datasets 3.6.0
# only has 'Sequence'/'LargeList', so a full LeRobotDataset load raises
# "Feature type 'List' not found"). uv sync leaves datasets at openpi's pinned 3.6.0, so bump it
# here. --no-deps to avoid churning numpy (openpi pins numpy<2; datasets 4.0.0 works under --no-deps).
uv pip install --no-deps 'datasets==4.0.0'

# lerobot 0.4.4's _query_hf_dataset tries a column-first lookup that ALWAYS raises
# on datasets 3.6.0, but only after materialising the whole column -> O(total_frames)
# wasted per sample (measured 1078x). Reorder it row-first. See patch_lerobot_query.py.
"${VIRTUAL_ENV}/bin/python" "${HERE}/patch_lerobot_query.py"

"${VIRTUAL_ENV}/bin/python" - <<'PY'
import importlib.metadata as m
import lerobot.datasets.lerobot_dataset as L
print("lerobot", m.version("lerobot"), "| CODEBASE_VERSION", L.CODEBASE_VERSION)
assert L.CODEBASE_VERSION == "v3.0", "expected v3.0-capable lerobot"
print("OK: openpi venv can now read LeRobot v3.0 datasets")
PY
