"""Numpy-only relative-EEF action math (no openpi/flax deps).

Kept dependency-free so the offline corpus column-add (which runs many parallel workers)
can import it without pulling in flax/jax. `axis_franka_policy` re-exports these.

Convention CONFIRMED against LIBERO's robosuite OSC_POSE (benchmark survey 2026-08-02):
per-step delta, WORLD-frame (`input_ref_frame=base`) position delta, orientation delta as a
3-D axis-angle vector, gripper absolute, 7-D action `[dx,dy,dz, dax,day,daz, gripper]`.
ALL quaternions here are **xyzw** (quat[3]=w) -- callers holding MuJoCo wxyz must reorder first.
"""
from __future__ import annotations

import numpy as np


def quat_xyzw_to_axisangle(quat) -> np.ndarray:
    """Axis-angle (rotation vector) of a unit xyzw quaternion (robosuite `_quat2axisangle`)."""
    q = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:  # ~zero rotation
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * np.arccos(w)) / den).astype(np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quats) of an xyzw quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def hemisphere_align(quats_xyzw) -> np.ndarray:
    """Put a SEQUENCE of xyzw quaternions in one continuous hemisphere.

    `canonical_xyzw` picks the sign per frame (w>=0). At a top-down gripper the rotation is ~180 deg
    and w sits at ~0, so that per-frame rule flips arbitrarily between the +v and -v representations
    of the SAME orientation -- and the axis-angle it feeds the policy jumps by 2*pi.

    MEASURED on the held-out corpus: `state_eef[3]` swings the full -pi..+pi with std 2.979 (task
    1889) and shows 6.28 rad single-step jumps on 0.8-3.8% of frames in EVERY task. LIBERO's own
    state, by contrast, sits in a narrow band -- std 0.354, q01..q99 = 1.53..3.26. Aligning each
    frame to the previous one's hemisphere reproduces that: std 2.979 -> 0.278, range 2.01..3.22.

    The DELTA path does not need this: `eef_pose_to_delta_actions` applies shortest-arc to
    `q_rel`, so a source sign flip cancels there. This is a STATE defect only.
    """
    q = np.asarray(quats_xyzw, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def canonical_xyzw(q) -> np.ndarray:
    """Sign-canonicalise a unit xyzw quaternion to w>=0 (robosuite mat2quat / LIBERO
    convention), so its axis-angle norm stays <= pi. MuJoCo does NOT canonicalise sign, so
    the STATE path must apply this or top-down grippers (w~0) land at norm>pi, a region
    LIBERO states never occupy -> degraded orientation-state transfer."""
    q = np.asarray(q, dtype=np.float64)
    return -q if q[3] < 0 else q


def episode_eef_columns(eef_pose_wxyz, gripper, episode_index, frame_index,
                        *, continuous_rotation: bool = False):
    """Per-frame relative-EEF `action_eef`(N,7) + `state_eef`(N,8) from an absolute MuJoCo
    **wxyz** eef_pose sequence, grouped by episode and ordered by frame_index within each.

      action_eef = [Δpos(world), Δaxis-angle(world), gripper_next]; the LAST frame of each
        episode holds still (zero delta) but KEEPS its own gripper (not 0) so end-of-episode
        chunk padding never teaches "open at the end".
      state_eef  = [pos(3), axis-angle(3), gripper, gripper] (8-D, LIBERO-shaped).
        `continuous_rotation=True` aligns each episode's quaternions to one hemisphere
        (see `hemisphere_align`) instead of the per-frame w>=0 rule, which flips by 2*pi
        at a top-down gripper. Default False so existing corpora rebuild byte-identical.

    Pure numpy; testable without parquet. Row order of the outputs matches the input rows.
    """
    pose = np.asarray(eef_pose_wxyz, dtype=np.float64)  # (N,7) [pos3, w,x,y,z]
    grip = np.asarray(gripper, dtype=np.float64)
    ep = np.asarray(episode_index)
    fr = np.asarray(frame_index)
    n = pose.shape[0]
    xyzw = pose[:, [0, 1, 2, 4, 5, 6, 3]]  # wxyz -> xyzw
    act = np.zeros((n, 7), dtype=np.float32)
    ste = np.zeros((n, 8), dtype=np.float32)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(fr[idx], kind="stable")]  # frame order within the episode
        p, g = xyzw[idx], grip[idx]
        if len(idx) > 1:
            act[idx[:-1]] = eef_pose_to_delta_actions(p, g)
        act[idx[-1]] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g[-1]]  # hold still, keep gripper
        q_seq = hemisphere_align(p[:, 3:7]) if continuous_rotation else p[:, 3:7]
        aa = np.stack([quat_xyzw_to_axisangle(q if continuous_rotation else canonical_xyzw(q))
                       for q in q_seq])
        ste[idx] = np.concatenate([p[:, :3], aa, g[:, None], g[:, None]], axis=1).astype(np.float32)
    return act, ste


def eef_pose_to_delta_actions(eef_pose, gripper) -> np.ndarray:
    """Per-frame relative-EEF delta actions from an absolute eef_pose sequence.

    `eef_pose` is (T, 7) = [pos(3), quat_xyzw(4)] per frame; `gripper` is (T,). Returns
    (T-1, 7) = [Δpos(3, world frame), Δaxis-angle(3), gripper_target]. Δrotation is the
    world-frame relative rotation `q_delta = q_{t+1} · conj(q_t)`, sign-normalised (w≥0) to
    the shortest arc; the action's gripper is the NEXT frame's closedness (the command).
    """
    pose = np.asarray(eef_pose, dtype=np.float64)
    grip = np.asarray(gripper, dtype=np.float64)
    t = pose.shape[0]
    out = np.zeros((max(t - 1, 0), 7), dtype=np.float32)
    for i in range(t - 1):
        out[i, :3] = pose[i + 1, :3] - pose[i, :3]
        q_rel = _quat_mul(pose[i + 1, 3:7], _quat_conj(pose[i, 3:7]))
        n = np.linalg.norm(q_rel)
        if n > 0:
            q_rel = q_rel / n
        if q_rel[3] < 0:  # shortest-arc: q and -q are the same rotation
            q_rel = -q_rel
        out[i, 3:6] = quat_xyzw_to_axisangle(q_rel)
        out[i, 6] = grip[i + 1]
    return out
