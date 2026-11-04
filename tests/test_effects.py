import numpy as np
import pytest

from synthrgbd.gbuffer import GBuffer
from synthrgbd.sensor.effects.stereo import disparity_noise_and_quantisation, stereo_shadow
from synthrgbd.sensor.effects.surface import edge_artifacts, metal_depth_error, metal_dropout
from synthrgbd.sensor.fields import correlated_uniform

F, B = 424.0, 0.05


def _gbuf(depth, instance=None, material=None, normals=None):
    h, w = depth.shape
    K = np.array([[F, 0, (w - 1) / 2], [0, F, (h - 1) / 2], [0, 0, 1]])
    if normals is None:
        normals = np.zeros((h, w, 3), np.float32)
        normals[..., 2] = -1.0
    return GBuffer(rgb=np.zeros((h, w, 3), np.uint8), depth=depth.astype(np.float32),
                   instance=np.zeros((h, w), np.int32) if instance is None else instance,
                   material=np.zeros((h, w), np.int32) if material is None else material,
                   normals=normals, K=K, T_world_cam=np.eye(4))


def test_stereo_shadow_band_left_of_box_and_left_border():
    z = np.full((4, 400), 1.0, np.float32)
    z[:, 200:250] = 0.5
    out = stereo_shadow(z, _gbuf(z), {"baseline_m": B}, np.random.default_rng(0))
    d_bg, d_box = F * B / 1.0, F * B / 0.5
    hidden = np.nonzero(out[0] == 0)[0]
    # left invalid band: u < d_bg
    occl_start = 200 - (d_box - d_bg)
    assert hidden.min() == 0 and np.all(out[0, int(np.ceil(d_bg)) + 1:int(occl_start) - 1] > 0)
    # occlusion band of width d_box - d_bg directly left of the box
    occl = hidden[hidden > d_bg + 1]
    assert occl.max() == 199
    assert len(occl) == pytest.approx(d_box - d_bg, abs=1.5)
    assert np.all(out[0, 200:250] == 0.5)


def test_disparity_noise_matches_z_squared_law_and_quantises():
    z0 = 0.6
    z = np.full((480, 848), z0, np.float32)
    p = {"baseline_m": B, "sigma_px": 0.08, "corr_px": 0.0, "step_px": 0.032, "depth_unit_m": 0.0001}
    out = disparity_noise_and_quantisation(z, _gbuf(z), p, np.random.default_rng(1))
    expected = z0 ** 2 * 0.08 / (F * B)
    assert out.std() == pytest.approx(expected, rel=0.1)
    # output disparities lie on the 0.032 px grid (up to the 0.1 mm depth unit)
    d = F * B / out[out > 0]
    assert np.median(np.abs(d / 0.032 - np.round(d / 0.032))) < 0.2


def test_correlated_uniform_has_uniform_marginals():
    u = correlated_uniform((200, 300), 3.0, np.random.default_rng(2))
    assert (u < 0.2).mean() == pytest.approx(0.2, abs=1e-3)


def test_metal_dropout_rate_follows_table_and_spares_other_materials():
    z = np.full((300, 300), 0.5, np.float32)
    material = np.zeros((300, 300), np.int32)
    material[:, 150:] = 1                      # right half is metal
    p = {"material_ids": [1], "angle_deg": [0, 90], "dropout_rate": [0.3, 0.3], "corr_px": 2.0}
    out = metal_dropout(z, _gbuf(z, material=material), p, np.random.default_rng(3))
    assert np.all(out[:, :150] == 0.5)
    assert (out[:, 150:] == 0).mean() == pytest.approx(0.3, abs=0.03)


def test_metal_effects_leave_silhouette_band_to_edge_artifacts():
    """Tables are measured on interior pixels; applying them in the band would double count."""
    z = np.full((100, 100), 0.5, np.float32)
    inst = np.full((100, 100), -1, np.int32)
    inst[20:80, 20:80] = 0
    material = np.ones((100, 100), np.int32)
    g = _gbuf(z, instance=inst, material=material)
    p = {"material_ids": [1], "angle_deg": [0, 90], "dropout_rate": [1.0, 1.0], "corr_px": 0.0, "edge_px": 3}
    out = metal_dropout(z, g, p, np.random.default_rng(6))
    assert np.all(out[17:23, 30:70] == 0.5)        # band around the top silhouette untouched
    assert np.all(out[30:70, 30:70] == 0.0)        # interior fully dropped (rate 1)


def test_metal_depth_error_bias_and_spread():
    z = np.full((300, 300), 0.5, np.float32)
    material = np.ones((300, 300), np.int32)
    p = {"material_ids": [1], "angle_deg": [0, 90], "bias_mm": [2.0, 2.0], "sigma_mm": [1.0, 1.0], "corr_px": 0.0}
    out = metal_depth_error(z, _gbuf(z, material=material), p, np.random.default_rng(4))
    err_mm = (out - z) * 1000
    assert err_mm.mean() == pytest.approx(2.0, abs=0.05)
    assert err_mm.std() == pytest.approx(1.0, rel=0.05)


def test_edge_artifacts_only_touch_silhouette_band():
    z = np.full((100, 100), 1.0, np.float32)
    inst = np.full((100, 100), -1, np.int32)
    z[30:70, 30:70] = 0.5
    inst[30:70, 30:70] = 0
    p = {"width_px": 2, "p_drop": 0.3, "p_fly": 0.2}
    out = edge_artifacts(z, _gbuf(z, instance=inst), p, np.random.default_rng(5))
    changed = out != z
    assert changed[33:67, 33:67].sum() == 0 and changed[:27].sum() == 0
    band = np.zeros_like(changed)
    band[28:72, 28:72] = True
    band[32:68, 32:68] = False
    assert changed[band].mean() == pytest.approx(0.5, abs=0.07)
    flying = changed & (out > 0)
    assert np.all((out[flying] >= 0.5) & (out[flying] <= 1.0))
