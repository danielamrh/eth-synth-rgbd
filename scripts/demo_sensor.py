"""Render the demo scene and show ideal depth, full D435 depth and each effect on its own."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from synthrgbd.render.mujoco_renderer import render_gbuffer
from synthrgbd.scene.builder import D435_H, D435_W, demo_scene_xml
from synthrgbd.sensor import SensorModel
from synthrgbd.sensor.d435 import build_d435


def main(seed: int = 0):
    model = mujoco.MjModel.from_xml_string(demo_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = render_gbuffer(model, data, "cam", D435_W, D435_H)
    alu = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "alu")
    sensor = build_d435([alu])

    full = sensor.apply(g, np.random.default_rng(seed))
    singles = {n: SensorModel([e for e in sensor.effects if e[0] == n]).apply(g, np.random.default_rng(seed))
               for n in sensor.names}

    metal = g.material == alu
    print(f"invalid on metal: ideal {np.mean(g.depth[metal] == 0):.3f}, D435 model {np.mean(full[metal] == 0):.3f}")
    print(f"invalid on table: D435 model {np.mean(full[~metal & (g.instance >= 0)] == 0):.3f}")

    panels = [("RGB", None), ("ideal depth", g.depth), ("D435 model (all effects)", full)] + list(singles.items())
    vmin, vmax = np.percentile(g.depth[g.depth > 0], [1, 99])
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for ax, (title, img) in zip(axes.flat, panels):
        if img is None:
            ax.imshow(g.rgb)
        else:
            ax.imshow(np.where(img > 0, img, np.nan), vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("white = invalid (depth 0)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/demo_sensor.png", dpi=90)
    print("saved figures/demo_sensor.png")


if __name__ == "__main__":
    main()
