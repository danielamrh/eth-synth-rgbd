"""Short video: the same scene and settled state, grasped from ideal depth and from simulated D435 depth.

Per scene: the depth image each planner sees (invalid = black) with the chosen cluster and the
planned jaw line, then both grasps executed side by side in MuJoCo (slow motion). Scenes are those
of a finished grasp run where condition A succeeds and B fails (or given with --seeds).

    pip install -e ".[video]"
    python scripts/grasp_video.py --tag main --n 3          # writes videos/grasp_main.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from synthrgbd.eval.grasp_trial import execute_grasp, settled_state
from synthrgbd.perception.grasp_planner import PARAMS, plan_grasp
from synthrgbd.render.mujoco_renderer import render_gbuffer
from synthrgbd.scene import cell
from synthrgbd.scene.builder import D435_H, D435_W

ROOT = Path(__file__).resolve().parents[1]
PW, PH = 640, 360            # one panel (848x480 scaled to 640 wide, height a multiple of 8)
FPS, SLOWMO = 30, 2.0        # video frames per second, slow-motion factor
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (245, 245, 245), (150, 150, 150)
GREEN, RED, BLUE, ORANGE = (60, 180, 75), (215, 60, 50), (42, 120, 214), (235, 104, 52)


def _sensors(metal_ids):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from grasp_experiment import make_sensors, sensor_for
    return make_sensors(metal_ids), sensor_for


def _text(img, s, xy, scale=0.55, color=WHITE, thick=1):
    """Text on a dark box (readable on any background)."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = xy
    x0, y0, x1, y1 = max(x - 5, 0), max(y - h - 6, 0), min(x + w + 5, img.shape[1]), min(y + base + 4, img.shape[0])
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def depth_panel(depth, gbuf, region, grasp, label, ok):
    v = depth[depth > 0]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    img[depth <= 0] = 0
    if region is not None:
        cnt, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, ORANGE, 2)
    if grasp is not None:
        # Jaw line: both finger positions at the grasp height, projected into the image.
        d = np.array([-np.sin(grasp.yaw), np.cos(grasp.yaw), 0.0]) * grasp.width / 2
        c = np.array([grasp.x, grasp.y, grasp.z_top])
        T = np.linalg.inv(gbuf.T_world_cam)
        pts = []
        for p in (c - d, c + d):
            q = T[:3, :3] @ p + T[:3, 3]
            pts.append((int(gbuf.K[0, 0] * q[0] / q[2] + gbuf.K[0, 2]), int(gbuf.K[1, 1] * q[1] / q[2] + gbuf.K[1, 2])))
        cv2.line(img, pts[0], pts[1], WHITE, 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 6, WHITE, -1, cv2.LINE_AA)
    img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_NEAREST)
    _text(img, label, (12, 28), 0.7, WHITE, 2)
    _text(img, "black = no depth   orange = chosen cluster   white = jaws", (12, PH - 12), 0.5, WHITE)
    if ok is not None:
        _text(img, "lifted" if ok else "failed", (PW - 110, 28), 0.7, GREEN if ok else RED, 2)
    return img


class Recorder:
    def __init__(self, renderer, model, grasp):
        self.r = renderer
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [grasp.x, grasp.y, max(grasp.z_top - 0.02, 0.0)]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.42, -28, np.degrees(grasp.yaw) + 90
        self.every = max(1, int(round(1 / (FPS * SLOWMO) / model.opt.timestep)))
        self.i, self.frames = 0, []

    def __call__(self, model, data):
        if self.i % self.every == 0:
            self.r.update_scene(data, self.cam)
            self.frames.append(self.r.render().copy())
        self.i += 1


def scene_frames(seed, cond_a, cond_b, params, settle_s):
    scene = cell.sample_scene(np.random.default_rng(seed))
    model, data = cell.build(scene)
    cell.settle(model, data, settle_s)
    state = settled_state(model, data)
    gbuf = render_gbuffer(model, data, "cam", D435_W, D435_H)
    metal_ids = [model.material("alu").id]
    sensors, sensor_for = _sensors(metal_ids)
    stills, clips, results = [], [], []
    renderer = mujoco.Renderer(model, PH, PW)
    for cond in (cond_a, cond_b):
        s = sensor_for(cond, sensors, metal_ids)
        depth = gbuf.depth.copy() if s is None else s.apply(gbuf, np.random.default_rng([seed, 1]))
        g, info = plan_grasp(depth, gbuf, np.random.default_rng([seed, 2]), params=params)
        rec = Recorder(renderer, model, g) if g is not None else None
        res = execute_grasp(model, data, state, scene, g, on_step=rec) if g is not None else {"success": False}
        results.append(res["success"])
        stills.append(depth_panel(depth, gbuf, info.get("region"), g, f"{cond} depth", res["success"]))
        clips.append(rec.frames if rec else [])
    renderer.close()
    return stills, clips, results


def compose(stills, clips, results, seed, cond_a, cond_b, writer):
    W, H = 2 * PW, 2 * PH + 40
    def frame(top, bottom):
        f = np.full((H, W, 3), 18, np.uint8)
        _text(f, f"scene {seed}: same scene, same settled state, same grasp planner; only the depth differs",
              (12, 27), 0.6, WHITE, 1)
        f[40:40 + PH, :PW], f[40:40 + PH, PW:] = top
        f[40 + PH:, :PW], f[40 + PH:, PW:] = bottom
        return f
    blank = np.zeros((PH, PW, 3), np.uint8)
    for _ in range(int(2.5 * FPS)):
        writer.append_data(frame(stills, (blank, blank)))
    n = max(len(c) for c in clips)
    for i in range(n + int(1.5 * FPS)):
        bottom = []
        for c, name, ok in zip(clips, (cond_a, cond_b), results):
            img = c[min(i, len(c) - 1)].copy() if c else blank.copy()
            _text(img, f"{name}: executed in MuJoCo ({SLOWMO:g}x slow motion)", (12, 28), 0.55, WHITE, 1)
            if i >= len(c) and c:
                _text(img, "lifted" if ok else "failed", (PW - 110, 28), 0.7, GREEN if ok else RED, 2)
            bottom.append(img)
        writer.append_data(frame(stills, bottom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--a", default="ideal")
    ap.add_argument("--b", default="d435")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    run = ROOT / "results" / "grasp" / a.tag
    cfg = json.loads((run / "run_config.json").read_text())["config"]
    params = {**PARAMS, **(cfg.get("planner") or {})}
    seeds = a.seeds
    if not seeds:
        ok = {}
        for line in open(run / "trials.jsonl"):
            r = json.loads(line)
            if r["pipeline"] == "depth" and r["condition"] in (a.a, a.b):
                ok.setdefault(r["seed"], {})[r["condition"]] = r["success"]
        seeds = [s for s in sorted(ok) if ok[s].get(a.a) and ok[s].get(a.b) is False][:a.n]
    out = ROOT / "videos" / f"grasp_{a.tag}.mp4"
    out.parent.mkdir(exist_ok=True)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8) as w:
        for s in seeds:
            stills, clips, results = scene_frames(s, a.a, a.b, params, cfg["settle_s"])
            print(f"seed {s}: {a.a} {'ok' if results[0] else 'fail'}, {a.b} {'ok' if results[1] else 'fail'}")
            compose(stills, clips, results, s, a.a, a.b, w)
    print("wrote", out)


if __name__ == "__main__":
    main()
