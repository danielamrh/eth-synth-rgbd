# synthrgbd: synthetic RGB-D of aluminium parts with a realistic depth camera model

Work in progress for the ETH Zurich doctoral assignment (humanoid robotics, 3D perception).

Pipeline: MuJoCo renders an ideal *G-buffer* (RGB, z-depth, instance and material ids,
normals, intrinsics, pose). A composable sensor model turns ideal depth into the depth a
reference active-stereo camera would report, with every effect switchable for ablations.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

On Linux the renderer uses headless EGL (`MUJOCO_GL=egl`). On Windows and macOS
MuJoCo's default OpenGL backend is used.

## Run

```bash
python scripts/probe_render.py   # renders the probe scene to figures/probe.png
python scripts/demo_sensor.py    # ideal vs simulated D435 depth, one panel per effect
pytest                           # unit tests
```

### Calibration on real metal parts (ROBI)

The metal and edge effects are calibrated on real RealSense D415 depth of metal parts from
ROBI (Yang et al., IROS 2021, CC BY-NC-SA 4.0). The fetch script pulls only one scene
(~60 MB) out of the 3.7 GB archive via HTTP range requests.

```bash
python scripts/robi_fetch.py --object Zigzag --scene 4
python scripts/robi_d415_metal.py --object Zigzag --scene 4   # ~10 min, writes results/robi/
python scripts/robi_blob_size.py --scene 4                     # dropout blob size (corr_px)
python scripts/robi_edge_steps.py --scene 4 --step 2           # edge dropout vs depth step
python scripts/robi_fetch.py --object Zigzag --scene 5         # holdout scene
python scripts/robi_grasp_check.py --scene 5                   # planner on real vs simulated depth, ~6 min
```

### Grasp experiment

Procedural cells (3-6 aluminium profiles and L-brackets, settled by physics, D435 on a randomised
fixed mount). Each scene is rendered once; every sensor condition (ideal, full D435 model,
D435 minus one effect, simpler noise models from the literature, metal sensitivity) is applied to
the same G-buffer, a top-down grasp is planned from depth and executed with a parallel-jaw gripper
in MuJoCo from the same settled state. Labels (part class, 6D pose, instance masks) and RGB-D of
the first 20 scenes are stored in `results/grasp/<tag>/examples/`.

```bash
python scripts/grasp_experiment.py --tag main      # 300 scenes x 16 runs, ~25 min on 6 cores
python scripts/grasp_report.py --tag main          # table, results/grasp/main/summary.json, figures/grasp_main.png
python scripts/grasp_experiment.py --tag clear6mm --clearance 0.006   # gripper sensitivity: 6 mm instead of 12 mm per side
python scripts/grasp_sensitivity.py --tag main     # which model part matters (after grasp_report.py)
python scripts/grasp_camera_pose.py --tag main     # camera-model gap vs camera distance and tilt
pip install -e ".[video]" && python scripts/grasp_video.py --tag clear6mm --n 3   # videos/grasp_clear6mm.mp4
```

## Layout

| Path | Content |
|------|---------|
| `configs/` | camera (every parameter with source), scene randomisation, experiment |
| `src/synthrgbd/render/` | MuJoCo to `GBuffer` |
| `src/synthrgbd/sensor/` | `SensorModel` pipeline and effects (`effect(depth, gbuf, params, rng) -> depth`, 0 = invalid) |
| `src/synthrgbd/scene/` | MJCF scene construction (`cell.py`: procedural cells, gripper, labels) |
| `src/synthrgbd/perception/` | depth-based grasp planner |
| `src/synthrgbd/eval/` | grasp execution in MuJoCo |
| `scripts/` | one entry point per step |
| `ai_log.md` | log of AI tool use for the report |
