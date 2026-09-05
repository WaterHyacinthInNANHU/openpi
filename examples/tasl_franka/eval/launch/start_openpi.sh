#!/usr/bin/env bash
# Bring up the FR3 openpi-eval dashboard (:8003) and its full chain.
#
# RUN THIS ON THE DESKTOP (TASL-1). No prior context needed:
#     ./start_openpi.sh             # it re-runs itself under sudo
# It checks the robot, guards the GPU, ensures the openpi serve_policy (:8000)
# is up, and launches the eval dashboard. Then open the Tailscale URL printed at the end.
#
# Prereqs it does NOT do for you (it will tell you if missing):
#   • FR3 powered + FCI active in Desk
#   • NUC1 droid-nuc-fr3 container up (zerorpc :4242)  — see the error if down
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ensure_root "$0" "$@"

# The openpi dashboard itself runs on the host, but preflight's camera probe and
# holder reap both talk to this container — start_collect.sh / teleop.sh / eval.sh
# all ensure it first and this script was the one that did not, so a container
# that had exited turned every launch into a bogus "ZEDs not SDK-visible".
# zed_serials() now falls back to the host probe, making this belt AND braces.
ensure_rlinf_container

# Release this dashboard's ZED handles and CUDA context before probing cameras
# or checking GPU availability. Otherwise a rerun can reset cameras still in
# use and reject its own dashboard as a competing compute process.
step "Stop any existing openpi dashboard (idempotent re-run)"
desk "pkill -TERM -f '[/_]openpi[.]py' || true"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then sleep 2; fi

preflight_robot openpi

step "Backend: serve_policy :8000"
if port_open 8000; then
  ok "serve_policy already up on :8000 (its GPU use is expected — no guard needed)"
else
  step "  :8000 down — GPU guard before starting serve (serve + training co-resident froze the box)"
  gpu_free || die "a compute process is using the Desktop GPU — do NOT start serve_policy on top of training (it froze the box). Stop training (or run it on the cluster), then retry."
  ok "GPU free — starting serve_policy (pi05_droid)"
  desk "cd $OPENPI_DIR && setsid .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_droid --policy.dir=$DESKTOP_HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid </dev/null >> $DESKTOP_HOME/_serve_policy.log 2>&1 &"
  wait_http 8000 60 || die "serve_policy did not come up on :8000 within ~2min (check $DESKTOP_HOME/_serve_policy.log)"
  ok "serve_policy up on :8000"
fi

step "Launch openpi dashboard :8003 (--policy-host 127.0.0.1 --policy-port 8000)"
WRIST_FLAG=""
if [[ "${ALLOW_MISSING_WRIST:-}" == 1 ]]; then
  WRIST_FLAG="--allow-missing-wrist"
  warn "dashboard starts with $WRIST_FLAG — wrist view is a BLACK frame"
fi
desk "cd $TASL && ulimit -n 8192 && LC_ALL=C.UTF-8 PYTHONPATH=$TASL:$SITE_PKGS:$OPENPI_DIR/packages/openpi-client/src setsid /usr/bin/python3 dashboards/openpi.py --port 8003 --policy-host 127.0.0.1 --policy-port 8000 $WRIST_FLAG </dev/null >> $LOG_DIR/openpi.log 2>&1 &"
wait_http 8003 || die "openpi dashboard did not answer on :8003 (see $LOG_DIR/openpi.log)"
ok "READY — openpi dashboard at http://$TS_IP:8003 (Tailscale; robot-net IP if TS offline)"
