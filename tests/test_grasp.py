import numpy as np
import pytest

from synthrgbd.eval.grasp_trial import execute_grasp, oracle_profile_grasp, settled_state
from synthrgbd.perception.grasp_planner import fit_table, plan_grasp
from synthrgbd.render.mujoco_renderer import render_gbuffer
from synthrgbd.scene import cell
from synthrgbd.scene.builder import D435_H, D435_W
from synthrgbd.sensor.baselines import build_baselines


def _single_profile(yaw=0.4, a=0.03, L=0.15, xy=(0.03, -0.02)):
    return {"parts": [{"kind": "profile", "a": a, "L": L, "name": "part_0",
                       "quat": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], "pos": [xy[0], xy[1], a / 2 + 0.005]}],
            "camera": {"pos": [0.0, 0.0, 0.5], "target": [0.0, 0.0, 0.0], "azimuth": 0.0}}


def test_fit_table_recovers_tilted_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (5000, 2))
    z = 0.1 * xy[:, 0] + 0.02 + rng.normal(0, 0.001, 5000)
    pts = np.c_[xy, z]
    pts[:500, 2] += 0.05                                    # 10 % outliers (objects)
    n, d = fit_table(pts, rng, 200, 0.003)
    assert n[2] > 0 and n @ np.array([-0.1, 0, 1]) / np.linalg.norm([-0.1, 0, 1]) == pytest.approx(1, abs=1e-3)


def test_planner_finds_isolated_profile_and_gripper_lifts_it():
    sc = _single_profile()
    m, d = cell.build(sc)
    cell.settle(m, d, 1.0)
    state = settled_state(m, d)
    g = render_gbuffer(m, d, "cam", D435_W, D435_H)
    grasp, info = plan_grasp(g.depth, g, np.random.default_rng(0))
    assert info["reason"] == "ok"
    assert np.hypot(grasp.x - 0.03, grasp.y + 0.02) < 0.003           # centre within 3 mm
    across = np.array([-np.sin(0.4), np.cos(0.4)])                     # perpendicular to the long axis
    closing = np.array([-np.sin(grasp.yaw), np.cos(grasp.yaw)])
    assert abs(across @ closing) == pytest.approx(1, abs=0.01)          # closes across the profile
    assert grasp.z_top == pytest.approx(0.03, abs=0.002)
    assert execute_grasp(m, d, state, sc, grasp)["success"]


def test_oracle_grasp_succeeds_and_off_target_grasp_fails():
    sc = _single_profile(yaw=1.2, a=0.02, L=0.1)
    m, d = cell.build(sc)
    cell.settle(m, d, 1.0)
    state = settled_state(m, d)
    og = oracle_profile_grasp(m, d, sc, 0)
    assert execute_grasp(m, d, state, sc, og)["success"]
    og.x += 0.2                                                        # grasp empty table
    assert not execute_grasp(m, d, state, sc, og)["success"]


def test_baselines_keep_every_pixel_and_match_amplitude():
    sc = _single_profile()
    m, d = cell.build(sc)
    g = render_gbuffer(m, d, "cam", D435_W, D435_H)
    for name, model in build_baselines().items():
        out = model.apply(g, np.random.default_rng(1))
        assert np.array_equal(out > 0, g.depth > 0), name             # no holes in the simple models
    out = build_baselines()["iid_gauss"].apply(g, np.random.default_rng(1))
    diff = (out - g.depth)[g.depth > 0]
    assert diff.std() == pytest.approx(0.5 ** 2 * 0.08 / (424 * 0.05), rel=0.05)
