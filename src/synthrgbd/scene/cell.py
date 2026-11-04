"""Procedural manufacturing-cell scenes: aluminium profiles and L-brackets on a table,
a D435 on a fixed mount above the cell, and a floating parallel-jaw gripper for grasp trials.

A scene is a plain dict (JSON-serialisable) so it can be stored as a label file and rebuilt.
Parts are dropped from above and settled by physics, so resting poses and contacts between
parts are realistic instead of hand-placed.
"""
import mujoco
import numpy as np

from synthrgbd.render.mujoco_renderer import camera_xml_attrs
from synthrgbd.scene.builder import D435_F, D435_H, D435_W

PARK = np.array([0.0, 0.0, 1.2])   # gripper rest position, above and behind every sampled camera
FINGER_LEN = 0.05                   # finger tip is this far below the gripper origin
MAX_OPEN = 0.085                    # total jaw opening [m], typical of a 2F-85 class gripper
CLOSE_OVERSHOOT = 0.01             # closing target below the joint limit -> force-limited squeeze

DEFAULTS = {
    "n_parts": [3, 6],
    "p_bracket": 0.4,
    "region_xy": [0.12, 0.09],          # half extents of the drop region on the table [m]
    "profile": {"section": [0.02, 0.03], "length": [0.08, 0.20]},
    "bracket": {"thickness": 0.005, "width": [0.03, 0.04], "leg": [0.04, 0.06]},
    "camera": {"distance": [0.35, 0.60], "tilt_deg": [0.0, 35.0], "target_jitter": 0.03},
    "settle_s": 3.0,
}


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _random_quat(rng):
    q = rng.standard_normal(4)
    return (q / np.linalg.norm(q)).tolist()


def sample_scene(rng: np.random.Generator, cfg: dict = DEFAULTS) -> dict:
    """Draw part shapes, drop poses and the camera mount. Settling happens in settle_scene."""
    n = int(rng.integers(cfg["n_parts"][0], cfg["n_parts"][1] + 1))
    parts, z = [], 0.0
    for k in range(n):
        if rng.random() < cfg["p_bracket"]:
            b = cfg["bracket"]
            part = {"kind": "bracket", "t": b["thickness"], "w": _u(rng, b["width"]),
                    "l1": _u(rng, b["leg"]), "l2": _u(rng, b["leg"]), "quat": _random_quat(rng)}
            extent = max(part["l1"], part["l2"], part["w"])
        else:
            p = cfg["profile"]
            yaw = rng.uniform(0, 2 * np.pi)
            part = {"kind": "profile", "a": float(rng.choice(p["section"])), "L": _u(rng, p["length"]),
                    "quat": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]}
            extent = part["a"]
        # Stagger drop heights so parts never start interpenetrating.
        z += extent + 0.01
        rx, ry = cfg["region_xy"]
        part["pos"] = [_u(rng, (-rx, rx)), _u(rng, (-ry, ry)), z]
        part["name"] = f"part_{k}"
        parts.append(part)

    c = cfg["camera"]
    target = np.array([_u(rng, (-c["target_jitter"], c["target_jitter"])),
                       _u(rng, (-c["target_jitter"], c["target_jitter"])), 0.0])
    dist, tilt, az = _u(rng, c["distance"]), np.deg2rad(_u(rng, c["tilt_deg"])), rng.uniform(0, 2 * np.pi)
    pos = target + dist * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    return {"parts": parts, "camera": {"pos": pos.tolist(), "target": target.tolist(), "azimuth": float(az)}}


def _camera_xyaxes(pos, target, az):
    """MuJoCo camera axes: x right, y up, looking along -z. x is kept horizontal."""
    fwd = np.asarray(target) - np.asarray(pos)
    fwd /= np.linalg.norm(fwd)
    x = np.array([-np.sin(az), np.cos(az), 0.0])     # perpendicular to the azimuth plane, so to fwd
    y = np.cross(-fwd, x)
    return np.concatenate([x, y])


def _part_xml(p):
    q = " ".join(f"{v:.6f}" for v in p["quat"])
    pos = " ".join(f"{v:.5f}" for v in p["pos"])
    if p["kind"] == "profile":
        geoms = f'<geom type="box" size="{p["L"] / 2} {p["a"] / 2} {p["a"] / 2}" material="alu"/>'
    else:
        t, w, l1, l2 = p["t"], p["w"], p["l1"], p["l2"]
        geoms = (f'<geom type="box" size="{l1 / 2} {w / 2} {t / 2}" pos="{l1 / 2} 0 {t / 2}" material="alu"/>'
                 f'<geom type="box" size="{t / 2} {w / 2} {l2 / 2}" pos="{t / 2} 0 {l2 / 2}" material="alu"/>')
    # Extrusions are hollow; ~40 % fill of aluminium (2700 kg/m^3) -> ~1100 kg/m^3 effective.
    return (f'<body name="{p["name"]}" pos="{pos}" quat="{q}"><freejoint/>'
            f'{geoms}</body>')


GRIPPER_XML = f"""
    <body name="gripper" pos="{PARK[0]} {PARK[1]} {PARK[2]}" gravcomp="1">
      <joint name="gx" type="slide" axis="1 0 0" damping="50"/>
      <joint name="gy" type="slide" axis="0 1 0" damping="50"/>
      <joint name="gz" type="slide" axis="0 0 1" damping="50"/>
      <joint name="gyaw" type="hinge" axis="0 0 1" damping="5"/>
      <geom name="palm" type="box" size="0.02 0.06 0.01" pos="0 0 0.01" rgba=".2 .2 .2 1" mass="0.3"/>
      <body name="finger_l" gravcomp="1">
        <joint name="fl" type="slide" axis="0 1 0" range="0 {MAX_OPEN / 2}" damping="5"/>
        <geom name="pad_l" type="box" size="0.01 0.005 {FINGER_LEN / 2}" pos="0 0.005 {-FINGER_LEN / 2}"
              friction="1.2 0.02 0.001" condim="4" rgba=".2 .2 .2 1" mass="0.05"/>
      </body>
      <body name="finger_r" gravcomp="1">
        <joint name="fr" type="slide" axis="0 -1 0" range="0 {MAX_OPEN / 2}" damping="5"/>
        <geom name="pad_r" type="box" size="0.01 0.005 {FINGER_LEN / 2}" pos="0 -0.005 {-FINGER_LEN / 2}"
              friction="1.2 0.02 0.001" condim="4" rgba=".2 .2 .2 1" mass="0.05"/>
      </body>
    </body>"""

ACTUATOR_XML = f"""
  <equality><joint joint1="fr" joint2="fl"/></equality>
  <actuator>
    <position name="ax" joint="gx" kp="3000" kv="100"/>
    <position name="ay" joint="gy" kp="3000" kv="100"/>
    <position name="az" joint="gz" kp="3000" kv="100" forcerange="-40 40"/>
    <position name="ayaw" joint="gyaw" kp="50" kv="5"/>
    <!-- Closing target lies beyond the joint limit, so the jaws squeeze with the force limit
         (40 N, low end of industrial 2-finger grippers) independent of part width. -->
    <position name="agrip" joint="fl" kp="5000" kv="20" ctrlrange="{-CLOSE_OVERSHOOT} {MAX_OPEN / 2}" forcerange="-40 40"/>
  </actuator>"""


def scene_xml(scene: dict) -> str:
    K = np.array([[D435_F, 0, (D435_W - 1) / 2], [0, D435_F, (D435_H - 1) / 2], [0, 0, 1]])
    cam = scene["camera"]
    xy = " ".join(f"{v:.6f}" for v in _camera_xyaxes(cam["pos"], cam["target"], cam["azimuth"]))
    pos = " ".join(f"{v:.5f}" for v in cam["pos"])
    parts = "\n    ".join(_part_xml(p) for p in scene["parts"])
    return f"""
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="{D435_W}" offheight="{D435_H}"/></visual>
  <default><geom friction="0.8 0.01 0.001" solref="0.004 1"/></default>
  <asset>
    <material name="alu" rgba=".75 .76 .78 1" specular="1" shininess="1"/>
    <material name="table" rgba=".35 .35 .38 1" specular=".05"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 1.5" dir="0 0.2 -1"/>
    <camera name="cam" pos="{pos}" xyaxes="{xy}" {camera_xml_attrs(K, D435_W, D435_H)}/>
    <geom name="table" type="plane" size="1 1 .1" material="table"/>
    {parts}
    {GRIPPER_XML}
  </worldbody>
  {ACTUATOR_XML}
</mujoco>"""


def build(scene: dict):
    model = mujoco.MjModel.from_xml_string(scene_xml(scene))
    data = mujoco.MjData(model)
    open_gripper(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def open_gripper(model, data):
    data.qpos[model.jnt_qposadr[model.joint("fl").id]] = MAX_OPEN / 2
    data.qpos[model.jnt_qposadr[model.joint("fr").id]] = MAX_OPEN / 2
    data.ctrl[model.actuator("agrip").id] = MAX_OPEN / 2


def part_body_ids(model, scene) -> list[int]:
    return [model.body(p["name"]).id for p in scene["parts"]]


def settle(model, data, seconds: float) -> float:
    """Let dropped parts come to rest. Returns the largest part speed at the end [m/s]."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return float(np.abs(data.qvel[:model.nv - 6]).max()) if model.nv > 6 else 0.0


def part_labels(model, data, scene) -> list[dict]:
    """Ground-truth labels per part: class, 6D pose (world), geom ids (for instance masks)."""
    out = []
    for p, bid in zip(scene["parts"], part_body_ids(model, scene)):
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        out.append({"name": p["name"], "class": p["kind"], "body_id": int(bid), "geom_ids": geoms,
                    "pos": data.xpos[bid].tolist(), "quat": data.xquat[bid].tolist()})
    return out
