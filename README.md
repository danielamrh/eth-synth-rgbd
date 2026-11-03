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
```

## Layout

| Path | Content |
|------|---------|
| `configs/` | camera (every parameter with source), scene randomisation, experiment |
| `src/synthrgbd/render/` | MuJoCo to `GBuffer` |
| `src/synthrgbd/sensor/` | `SensorModel` pipeline and effects (`effect(depth, gbuf, params, rng) -> depth`, 0 = invalid) |
| `src/synthrgbd/scene/` | MJCF scene construction |
| `src/synthrgbd/labels/`, `perception/`, `eval/` | dataset writer, grasp estimation, evaluation (to come) |
| `scripts/` | one entry point per step |
| `ai_log.md` | log of AI tool use for the report |
