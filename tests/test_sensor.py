import numpy as np
import pytest

from synthrgbd.gbuffer import GBuffer
from synthrgbd.sensor import SensorModel
from synthrgbd.sensor.effects.stereo import (depth_to_disparity, disparity_to_depth,
                                             range_limits)


def _gbuf(depth):
    h, w = depth.shape
    return GBuffer(rgb=np.zeros((h, w, 3), np.uint8), depth=depth,
                   instance=np.zeros((h, w), np.int32), material=np.zeros((h, w), np.int32),
                   normals=np.zeros((h, w, 3), np.float32), K=np.eye(3), T_world_cam=np.eye(4))


def test_disparity_roundtrip_keeps_invalid():
    z = np.array([[0.0, 0.3, 1.0, 2.5]], np.float32)
    back = disparity_to_depth(depth_to_disparity(z, 400.0, 0.05), 400.0, 0.05)
    np.testing.assert_allclose(back, z, rtol=1e-6)


def test_range_limits():
    z = np.array([[0.1, 0.5, 5.0]], np.float32)
    out = range_limits(z, _gbuf(z), {"min_z_m": 0.2, "max_z_m": 3.0}, np.random.default_rng(0))
    np.testing.assert_array_equal(out, [[0.0, 0.5, 0.0]])


def test_pipeline_order_ablation_and_input_untouched():
    z = np.ones((2, 2), np.float32)
    add = lambda d, g, p, r: d + p["k"]
    model = SensorModel([("a", add, {"k": 1.0}), ("b", add, {"k": 10.0})])
    g = _gbuf(z)
    assert model.apply(g, np.random.default_rng(0))[0, 0] == 12.0
    assert model.without("b").apply(g, np.random.default_rng(0))[0, 0] == 2.0
    assert g.depth[0, 0] == 1.0
    with pytest.raises(KeyError):
        model.without("missing")
