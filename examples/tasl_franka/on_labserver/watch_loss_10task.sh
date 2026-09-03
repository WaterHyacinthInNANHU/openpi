#!/usr/bin/env bash
# 读 10task 训练的 loss。
# 为什么不直接看 /tmp/train_10task.log:openpi 的 `pbar.write(f"Step {step}: ...")`
# 走的是块缓冲的 stdout,重定向到文件后要攒满 ~4KB 才落盘,所以文本日志里的 loss
# 会滞后几千步。wandb 的 offline 数据是每次 log 就落盘的,读它才实时。
set -u
RUN=$(ls -dt /data1/Franka_RealRobot/openpi/wandb/offline-run-* | head -1)
UVPY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages \
"$UVPY" - "$RUN" <<"PY"
import sys, glob
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
f = glob.glob(sys.argv[1] + "/*.wandb")[0]
ds = datastore.DataStore(); ds.open_for_scan(f)
rows = []
while True:
    d = ds.scan_data()
    if d is None: break
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        m = {tuple(i.nested_key)[0] if i.nested_key else i.key: i.value_json for i in r.history.item}
        if "loss" in m:
            rows.append((int(float(m["_step"])), float(m["loss"]), float(m.get("grad_norm", "nan"))))
rows.sort()
print("run:", sys.argv[1].split("/")[-1], "| 已记录", len(rows), "个点")
print("  step      loss    grad_norm")
for s, l, g in rows[-15:]:
    print(f"  {s:>6}  {l:8.4f}  {g:8.3f}")
PY
