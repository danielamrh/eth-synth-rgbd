"""Simpler depth noise models from the literature, as rivals to the D435 model.

All of them perturb every valid pixel but never remove one, which is the main difference to the
D435 model (holes on metal and at silhouettes, valid-but-biased depth on metal).
"""
from pathlib import Path

import numpy as np
import yaml

from synthrgbd.gbuffer import GBuffer
from synthrgbd.sensor.d435 import _v, load_config
from synthrgbd.sensor.fields import correlated_noise
from synthrgbd.sensor.pipeline import SensorModel

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "noise_baselines.yaml"


def iid_gauss(depth, gbuf: GBuffer, params, rng):
    out = depth.copy()
    valid = depth > 0
    out[valid] += params["sigma_m"] * rng.standard_normal(int(valid.sum()))
    return out


def z2_gauss(depth, gbuf: GBuffer, params, rng):
    out = depth.copy()
    valid = depth > 0
    sigma = depth[valid] ** 2 * params["sigma_px"] / (gbuf.K[0, 0] * params["baseline_m"])
    out[valid] += sigma * rng.standard_normal(int(valid.sum()))
    return out


def dexnet_noise(depth, gbuf: GBuffer, params, rng):
    out = depth.copy()
    valid = depth > 0
    alpha = rng.gamma(params["gamma_shape"], params["gamma_scale"])
    eps = params["gp_sigma_m"] * correlated_noise(depth.shape, params["gp_bandwidth_px"] / np.sqrt(2), rng)
    out[valid] = alpha * depth[valid] + eps[valid]
    return out


def build_baselines(config_path=DEFAULT_CONFIG, camera_config=None) -> dict[str, SensorModel]:
    c = yaml.safe_load(Path(config_path).read_text())
    cam = load_config(camera_config) if camera_config else load_config()
    sigma_px = _v(cam["stereo"]["subpixel_rms_px"])
    B = _v(cam["stereo"]["baseline_m"])
    f = _v(cam["intrinsics"]["f_px"])
    z_ref = c["iid_gauss"]["z_ref_m"]
    d = c["dexnet"]
    return {
        "iid_gauss": SensorModel([("iid_gauss", iid_gauss, {"sigma_m": z_ref ** 2 * sigma_px / (f * B)})]),
        "z2_gauss": SensorModel([("z2_gauss", z2_gauss, {"sigma_px": sigma_px, "baseline_m": B})]),
        "dexnet": SensorModel([("dexnet", dexnet_noise, {
            "gamma_shape": d["gamma_shape"], "gamma_scale": d["gamma_scale"],
            "gp_sigma_m": d["gp_sigma_m"], "gp_bandwidth_px": d["gp_bandwidth_px"]})]),
    }
