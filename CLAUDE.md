# Project: ETH Zurich doctoral assignment (synthetic RGB-D perception)

Candidate: Daniel Amrhein. Deadline: **Sunday 8 Nov 2026, 23:59 CET**. Submit to jpetrik@ethz.ch (PDF report + code, e.g. GitHub link).
Language of report and code comments: English. Chat with Daniel in German, concise.

## Assignment (summary of the invitation)
Robot in a manufacturing cell picks/assembles parts made of aluminium profiles and brackets with an RGB-D camera as main sensor.
1. Choose a commercially available RGB-D camera as reference. Build a small pipeline that generates **labelled synthetic RGB-D images** of such a scene (any simulator/renderer). Do not use ideal depth: model the camera's realistic imperfections on these parts. Explain which effects are included, which left out, and why.
2. Design and run **one small experiment** that shows whether the camera model changes the outcome of a downstream perception or grasping step. Report quantitative results **including what did not work**.
3. At most **one page**: the riskiest assumption when perception models trained mainly on synthetic data must work on a real robot in a factory, and what to do in the first six months of a doctorate to reduce that risk.
4. Deliverables: **PDF report, max. 8 pages (mandatory)**, runnable code (GitHub), optional videos/sketches.
5. AI tools are allowed. Add a short note on how they were used, **including one real place where Daniel disagreed with or corrected a tool's suggestion**. Never invent this: keep `ai_log.md` and append real disagreements/corrections when they happen.
No single expected solution. They assess approach, technical reasoning and judgement.

## Decisions so far
- Simulator: **MuJoCo** (3.17.0, venv in `.venv`, `pip install -e ".[dev]"`). `MUJOCO_GL=egl` only on Linux; on Windows the default backend is used (set in `render/mujoco_renderer.py`). MuJoCo has no ray tracing, so specular/missing-depth effects on aluminium must be modelled in our own sensor model.
- Architecture: renderer -> `GBuffer` (rgb, z-depth, instance, material, normals, K, T_world_cam in OpenCV convention, aux) -> `SensorModel` (list of named effects `effect(depth, gbuf, params, rng) -> depth`, 0 = invalid, `.without(name)` for ablations) -> perception -> eval.
- Camera: **Intel RealSense D435** (decided 2026-11-03). Main source [R25] Rustler et al., IEEE Access 2025 (arXiv:2501.07421): recommends D435 for table-top object perception up to ~1 m; gives range, FOV, 848x480 setting, plane SD/bias (Table 3), public D435 recordings for validating the noise model. [R25] has no baseline/focal/subpixel values and no metal parts. [DS] Intel D400 datasheet Rev. 019 (Oct 2024) gives baseline 50 mm, imager HFOV 90 deg (-> f = 424 px at 848x480), Min-Z 195 mm, factory depth quality spec; no subpixel value. Aluminium dropouts need a further source (e.g. Halmetschlager-Funek 2019, ROBI). All values with source in `configs/camera_d435.yaml`. Do not guess numbers.
- Plan: (1) simple NumPy sensor model first (noise and quantisation in disparity space so sigma_z ~ z^2 follows, dropouts at grazing angles and on specular/anodised aluminium, edge flying/dropout at silhouettes, stereo shadow via second render at the right imager, min/max range); (2) optional upgrade: render stereo pair + dot pattern and run OpenCV SGBM; compare both models in the experiment if time allows.
- Main experiment (proposal): simple grasp-point estimation from depth (plane fit + centroid), executed in MuJoCo, success rate for ideal depth vs sensor model vs single effects ablated (>= ~200 scenes per condition, fixed seeds, confidence intervals). Optional: small segmentation net trained on ideal vs modelled depth, tested on held-out noise parameters; also RGB vs depth vs RGB-D under lighting shifts (active IR depth is almost independent of visible light).
- Reality check (optional): ROBI dataset (real reflective metal parts, Ensenso N35 + RealSense D415, ground-truth depth) if it is downloadable.
- Sensor model implemented (`sensor/d435.py`, order: stereo_shadow, range, metal_dropout, edge_artifacts, metal_depth_error, disparity_noise). Metal/edge tables measured on ROBI Zigzag scene 4 (D415, CAD-rendered GT): dropout 9-16 % below 40 deg, 60 % at 80-90 deg; bias up to +2.2 mm; head-on dropout higher than at 10-20 deg (saturation branch). Spread is taken in excess of the 1.32 mm best-angle floor (GT alignment). Open: holdout check on ROBI scene 5, blob size (corr_px) not measured.
- Left out and justified in report: multipath, temperature drift, motion blur, rolling shutter.
- Fresh work: Daniel's earlier ACT visual-robustness project is already mentioned in his cover letter, so this task must not reuse it as the main idea.

## Rules for Claude
- **Git: Claude never commits.** Daniel reviews diffs and commits himself. Never add `Co-Authored-By` or other AI attribution lines to commits or PRs. AI use is disclosed in the report and `ai_log.md` instead.
- Do not invent sensor parameters, citations or results. Mark unverified numbers clearly and ask Daniel to verify them.
- Every claim in the report must be backed by a measured result or a cited source.
- Keep code simple and reproducible: fixed seeds, one command per experiment, README that runs on a fresh machine. Add tests where cheap.
- Daniel must understand and be able to explain every part (an interview follows). Explain design choices briefly in plain German when asked.
- Daniel writes the one-page risk statement himself. Claude gives feedback only.

## Schedule (Tue 3 Nov to Sun 8 Nov)
Tue/Wed: camera sources, scene + labels, depth pipeline. Thu: sensor model + validation. Fri: grasp experiment + ablation. Sat: optional net, report, risk page. Sun: clean repo, fresh-machine test, submit by ~18:00.

## Open
- GPU available on Daniel's laptop? (affects SAPIEN/Blender options)
- Subpixel RMS 0.08 px is back-calculated from [R25] only; verify (Intel stereo tuning white paper, fit on [R25] D435 plane data).
- MuJoCo camera must use f = 424 px at 848x480 (derived from datasheet Table 3-13), not the 87 deg depth FOV.
