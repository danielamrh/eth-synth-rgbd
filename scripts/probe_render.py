"""Smoke test: render the probe scene and save RGB / depth / instance mask."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from synthrgbd.render.mujoco_renderer import render_gbuffer
from synthrgbd.scene.builder import probe_scene_xml


def main():
    model = mujoco.MjModel.from_xml_string(probe_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = render_gbuffer(model, data, "cam")

    print(f"mujoco {mujoco.__version__}  rgb {g.rgb.shape}  "
          f"depth [{g.depth.min():.3f}, {g.depth.max():.3f}] m  "
          f"instances {np.unique(g.instance).tolist()}  materials {np.unique(g.material).tolist()}")

    out = Path("figures/probe.png")
    out.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    ax[0].imshow(g.rgb); ax[0].set_title("RGB")
    im = ax[1].imshow(g.depth); ax[1].set_title("ideal depth [m]"); fig.colorbar(im, ax=ax[1])
    ax[2].imshow(g.instance); ax[2].set_title("instance (geom id)")
    ax[3].imshow((g.normals + 1) / 2); ax[3].set_title("normals (camera frame)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
