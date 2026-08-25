#!/usr/bin/env bash
# Upload the two 5-step checkpoint repos to HF (params+assets only, ~30G each).
# Resumable: re-run to continue an interrupted upload.
set -u
export PATH=$HOME/.local/bin:$PATH
cd /data1/vla-reasoning || exit 1

echo "[$(date +%H:%M)] uploading 5k repo..."
hf upload Litian2002/pi05-droid-franka-lora-5k hf-upload-5k --repo-type dataset
echo "[$(date +%H:%M)] 5k repo done (exit $?)"

echo "[$(date +%H:%M)] uploading calibra50 repo..."
hf upload Litian2002/pi05-droid-franka-lora-5k-calibra50 hf-upload-calibra50 --repo-type dataset
echo "[$(date +%H:%M)] calibra50 repo done (exit $?)"

echo "[$(date +%H:%M)] ALL UPLOADS COMPLETE"
