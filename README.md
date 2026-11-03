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
pytest                           # unit tests
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
