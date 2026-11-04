"""Grasp experiment: same scenes, different depth sensor models, physical grasp outcome.

    python scripts/grasp_experiment.py                  # full run (configs/grasp_experiment.yaml)
    python scripts/grasp_experiment.py --n 20 --tag pilot
    python scripts/grasp_experiment.py --dev            # dev scenes, ideal depth only

Per scene: sample parts and camera mount, settle the parts by physics, render one G-buffer,
then for every condition apply the sensor model, plan a grasp, execute it from the same
settled state. Writes one JSON line per (scene, pipeline, condition) to results/grasp/<tag>/.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def make_sensors(metal_ids):
    from synthrgbd.sensor.baselines import build_baselines
    from synthrgbd.sensor.d435 import build_d435
    full = build_d435(metal_ids)
    s = {"ideal": None, "d435": full}
    for name in full.names:
        s[f"no_{name}"] = full.without(name)
    s.update(build_baselines())
    s["metal_x0.5"] = build_d435(metal_ids, metal_scale=0.5)
    s["metal_x2"] = build_d435(metal_ids, metal_scale=2.0)
    s["metal_corr0"] = build_d435(metal_ids, metal_corr_px=0.0)
    s["metal_corr6"] = build_d435(metal_ids, metal_corr_px=6.0)
    return s


def _ref_grasp(gbuf, mask, seed):
    """Grasp the planner makes for one part from ideal depth and its true mask (error reference)."""
    from synthrgbd.perception.grasp_planner import plan_grasp
    g, _ = plan_grasp(gbuf.depth, gbuf, np.random.default_rng(seed), instance_masks=[mask])
    return g


def _angle_diff(a, b):
    """Difference of two closing directions; a jaw is symmetric, so modulo 180 deg."""
    d = (a - b) % np.pi
    return float(np.degrees(min(d, np.pi - d)))


def run_scene(args):
    seed, cfg, conditions, mask_conditions, save_dir = args
    from synthrgbd.eval.grasp_trial import execute_grasp, settled_state
    from synthrgbd.perception.grasp_planner import plan_grasp
    from synthrgbd.render.mujoco_renderer import render_gbuffer
    from synthrgbd.scene import cell
    from synthrgbd.scene.builder import D435_H, D435_W

    t0 = time.time()
    scene = cell.sample_scene(np.random.default_rng(seed))
    model, data = cell.build(scene)
    speed = cell.settle(model, data, cfg["settle_s"])
    state = settled_state(model, data)
    gbuf = render_gbuffer(model, data, "cam", D435_W, D435_H)
    labels = cell.part_labels(model, data, scene)
    metal_ids = [model.material("alu").id]
    sensors = make_sensors(metal_ids)
    masks = [np.isin(gbuf.instance, lab["geom_ids"]) for lab in labels]
    part_of_pixel = np.full(gbuf.instance.shape, -1, np.int32)
    for k, m in enumerate(masks):
        part_of_pixel[m] = k

    if save_dir is not None:
        np.savez_compressed(save_dir / f"scene_{seed}.npz", rgb=gbuf.rgb,
                            depth_mm=np.round(gbuf.depth * 1000).astype(np.uint16),
                            part=part_of_pixel, material=gbuf.material, K=gbuf.K, T_world_cam=gbuf.T_world_cam)
        (save_dir / f"scene_{seed}.json").write_text(json.dumps({"scene": scene, "parts": labels}, indent=1))

    rows, cache, ref_cache = [], {}, {}
    jobs = [("depth", c) for c in conditions] + [("masks", c) for c in mask_conditions]
    for pipeline, cond in jobs:
        sensor = sensors[cond]
        # Same noise seed for every condition of a scene -> paired comparison.
        depth = gbuf.depth.copy() if sensor is None else sensor.apply(gbuf, np.random.default_rng([seed, 1]))
        grasp, info = plan_grasp(depth, gbuf, np.random.default_rng([seed, 2]),
                                 instance_masks=masks if pipeline == "masks" else None)
        row = {"seed": seed, "pipeline": pipeline, "condition": cond, "n_parts": len(scene["parts"]),
               "settle_speed": speed, "valid_frac_parts": float((depth[part_of_pixel >= 0] > 0).mean()),
               "plan": info["reason"]}
        if grasp is None:
            row.update({"success": False, "n_lifted": 0})
            rows.append(row)
            continue
        key = tuple(np.round([grasp.x, grasp.y, grasp.z_tip, grasp.yaw, grasp.width, grasp.z_top], 5))
        if key not in cache:
            cache[key] = execute_grasp(model, data, state, scene, grasp)
        row.update(cache[key])
        row["grasp"] = grasp.as_dict()
        # Which part did the planner aim at (majority of the chosen cluster's pixels)?
        ids = part_of_pixel[info["region"]]
        ids = ids[ids >= 0]
        target = int(np.bincount(ids).argmax()) if len(ids) else -1
        row["target"] = target
        row["cluster_purity"] = float((ids == target).mean()) if len(ids) else 0.0
        if target >= 0:
            if target not in ref_cache:
                ref_cache[target] = _ref_grasp(gbuf, masks[target], seed)
            ref = ref_cache[target]
            if ref is not None:
                row["err"] = {"xy_mm": float(np.hypot(grasp.x - ref.x, grasp.y - ref.y) * 1000),
                              "yaw_deg": _angle_diff(grasp.yaw, ref.yaw),
                              "z_top_mm": float((grasp.z_top - ref.z_top) * 1000),
                              "width_mm": float((grasp.width - ref.width) * 1000)}
        rows.append(row)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "grasp_experiment.yaml"))
    ap.add_argument("--n", type=int, help="override number of scenes")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dev", action="store_true", help="dev scenes, ideal depth only")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.dev:
        seeds = [cfg["dev_seed"] + i for i in range(a.n or cfg["n_dev_scenes"])]
        conditions, mask_conditions, tag = ["ideal"], ["ideal"], "dev"
    else:
        seeds = [cfg["seed"] + i for i in range(a.n or cfg["n_scenes"])]
        conditions, mask_conditions, tag = cfg["conditions"], cfg["mask_conditions"], a.tag

    out = ROOT / "results" / "grasp" / tag
    img_dir = out / "examples"
    img_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, cfg, conditions, mask_conditions, img_dir if i < cfg["save_images"] else None)
            for i, s in enumerate(seeds)]
    (out / "run_config.json").write_text(json.dumps({"config": cfg, "seeds": seeds, "conditions": conditions,
                                                     "mask_conditions": mask_conditions}, indent=1))
    t0 = time.time()
    with open(out / "trials.jsonl", "w") as f, ProcessPoolExecutor(a.workers) as ex:
        for i, (rows, dt) in enumerate(ex.map(run_scene, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            ok = {r["condition"]: r["success"] for r in rows if r["pipeline"] == "depth"}
            print(f"[{i + 1}/{len(jobs)}] seed {rows[0]['seed']} {dt:.1f}s  "
                  f"ideal={ok.get('ideal')} d435={ok.get('d435')}  elapsed {time.time() - t0:.0f}s", flush=True)
    print("wrote", out / "trials.jsonl")


if __name__ == "__main__":
    main()
