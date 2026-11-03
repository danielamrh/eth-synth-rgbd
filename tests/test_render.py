import mujoco
import numpy as np
import pytest

from synthrgbd.render.mujoco_renderer import render_gbuffer
from synthrgbd.scene.builder import probe_scene_xml


@pytest.fixture(scope="module")
def probe():
    model = mujoco.MjModel.from_xml_string(probe_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, render_gbuffer(model, data, "cam")


def test_shapes_and_labels(probe):
    model, _, g = probe
    assert g.rgb.shape == (480, 640, 3)
    assert g.depth.shape == g.instance.shape == g.material.shape == (480, 640)
    profile = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "profile")
    assert profile in np.unique(g.instance)
    assert np.all(g.depth[g.instance < 0] == 0)
    assert np.all(g.depth[g.instance >= 0] > 0)


def test_projection_matches_segmentation_and_depth(probe):
    """Project the profile's top-face centre with K and T; pixel must show the profile at that depth."""
    model, data, g = probe
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "profile")
    p_world = data.geom_xpos[gid] + np.array([0, 0, model.geom_size[gid][2]])
    T_cam_world = np.linalg.inv(g.T_world_cam)
    p_cam = T_cam_world[:3, :3] @ p_world + T_cam_world[:3, 3]
    u, v, _ = g.K @ (p_cam / p_cam[2])
    u, v = int(round(u)), int(round(v))
    assert g.instance[v, u] == gid
    assert g.depth[v, u] == pytest.approx(p_cam[2], abs=2e-3)


def test_explicit_intrinsics_with_offset_principal_point():
    """A small sphere must appear where K projects it, for a non-centred principal point."""
    from synthrgbd.render.mujoco_renderer import camera_xml_attrs
    W, H = 320, 240
    K = np.array([[300.0, 0, 150.3], [0, 300.0, 127.6], [0, 0, 1]])
    xml = f"""<mujoco><visual><global offwidth="{W}" offheight="{H}"/></visual><worldbody>
      <camera name="c" pos="0 0 1" {camera_xml_attrs(K, W, H)}/>
      <geom name="s" type="sphere" size=".03" pos="0.08 -0.05 0"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = render_gbuffer(model, data, "c", W, H)
    np.testing.assert_allclose(g.K, K, atol=1e-6)
    p_cam = np.linalg.inv(g.T_world_cam) @ np.array([0.08, -0.05, 0, 1])
    u, v, _ = K @ (p_cam[:3] / p_cam[2])
    vs, us = np.nonzero(g.instance == 0)
    assert us.mean() == pytest.approx(u, abs=0.25)
    assert vs.mean() == pytest.approx(v, abs=0.25)


def test_table_normal_points_to_camera(probe):
    model, _, g = probe
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    up_cam = np.linalg.inv(g.T_world_cam)[:3, :3] @ np.array([0, 0, 1.0])
    mask = g.instance == table
    mean_n = g.normals[mask].mean(axis=0)
    assert np.dot(mean_n / np.linalg.norm(mean_n), up_cam) > 0.99
