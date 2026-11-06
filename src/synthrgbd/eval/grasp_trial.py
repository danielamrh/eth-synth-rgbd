"""Execute a planned top-down grasp with the floating parallel-jaw gripper in MuJoCo.

Sequence: place the gripper above the grasp (open to the planned width, at the planned yaw),
descend, close with limited force, lift, hold. Success = at least one part ends more than
`lift_success` above its settled height. The same settled state is restored for every trial,
so all sensor conditions see exactly the same physical scene.
"""
import mujoco
import numpy as np

from synthrgbd.scene.cell import CLOSE_OVERSHOOT, FINGER_LEN, MAX_OPEN, PARK, part_body_ids

TRIAL = {
    "approach_clearance": 0.06,   # [m] start this far above the estimated object top
    "descend_s": 0.6,
    "close_s": 0.5,
    "lift": 0.12,
    "lift_s": 0.6,
    "hold_s": 1.0,
    "lift_success": 0.05,
}


def _set(model, data, name, value):
    data.ctrl[model.actuator(name).id] = value


def _q(model, data, joint, value=None):
    adr = model.jnt_qposadr[model.joint(joint).id]
    if value is not None:
        data.qpos[adr] = value
    return data.qpos[adr]


def _run(model, data, seconds, ramp=None, on_step=None):
    """Step for `seconds`; ramp = (actuator, start, end) interpolates its target linearly.
    on_step(model, data) is called after every step (used only to record videos)."""
    n = max(1, int(seconds / model.opt.timestep))
    for i in range(n):
        if ramp is not None:
            act, a, b = ramp
            _set(model, data, act, a + (b - a) * (i + 1) / n)
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(model, data)


def execute_grasp(model, data, settled_state: np.ndarray, scene: dict, grasp, p: dict = TRIAL,
                  on_step=None) -> dict:
    """Run one trial from `settled_state` (mj state vector). Returns outcome and diagnostics."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, settled_state, spec)
    mujoco.mj_forward(model, data)
    bodies = part_body_ids(model, scene)
    z0 = data.xpos[bodies, 2].copy()

    # Gripper origin = finger root; tips are FINGER_LEN below it. Joint values are offsets from PARK.
    gx, gy = grasp.x - PARK[0], grasp.y - PARK[1]
    z_start = grasp.z_top + p["approach_clearance"] + FINGER_LEN - PARK[2]
    z_grasp = grasp.z_tip + FINGER_LEN - PARK[2]
    half = min(grasp.width, MAX_OPEN) / 2
    for joint, act, val in [("gx", "ax", gx), ("gy", "ay", gy), ("gz", "az", z_start), ("gyaw", "ayaw", grasp.yaw)]:
        _q(model, data, joint, val)
        _set(model, data, act, val)
    for joint in ("fl", "fr"):
        _q(model, data, joint, half)
    _set(model, data, "agrip", half)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    _run(model, data, p["descend_s"], ("az", z_start, z_grasp), on_step=on_step)
    _run(model, data, 0.1, on_step=on_step)
    z_reached = _q(model, data, "gz") + PARK[2] - FINGER_LEN
    _run(model, data, p["close_s"], ("agrip", half, -CLOSE_OVERSHOOT), on_step=on_step)
    opening = 2 * _q(model, data, "fl")
    _run(model, data, p["lift_s"], ("az", z_grasp, z_grasp + p["lift"]), on_step=on_step)
    _run(model, data, p["hold_s"], on_step=on_step)

    rise = data.xpos[bodies, 2] - z0
    lifted = rise > p["lift_success"]
    return {
        "success": bool(lifted.any()),
        "n_lifted": int(lifted.sum()),
        "lifted": [scene["parts"][i]["name"] for i in np.nonzero(lifted)[0]],
        "tip_blocked_mm": float((z_reached - grasp.z_tip) * 1000),   # > 0: fingers stopped early
        "closed_opening_mm": float(opening * 1000),                   # ~0: closed on nothing
    }


def settled_state(model, data) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    s = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, s, spec)
    return s


def oracle_profile_grasp(model, data, scene: dict, k: int):
    """Grasp from the ground-truth pose of profile k (gripper sanity check, not a perception result).

    Closes across the profile's local y axis at its centre; None if the profile does not lie flat.
    """
    from synthrgbd.perception.grasp_planner import Grasp
    p = scene["parts"][k]
    if p["kind"] != "profile":
        return None
    b = model.body(p["name"]).id
    R = data.xmat[b].reshape(3, 3)
    long_axis = R[:, 0]
    if abs(long_axis[2]) > 0.2:            # standing or leaning
        return None
    across = np.array([-long_axis[1], long_axis[0]])
    across /= np.linalg.norm(across)
    pos = data.xpos[b]
    # Top of the box: centre height plus the largest vertical half-extent.
    half = np.array([p["L"], p["a"], p["a"]]) / 2
    z_top = pos[2] + float(np.abs(R[2, :]) @ half)
    yaw = float(np.arctan2(-across[0], across[1]))
    return Grasp(float(pos[0]), float(pos[1]), float(max(0.003, z_top - 0.015)), yaw,
                 p["a"] + 0.024, float(z_top), 0)
