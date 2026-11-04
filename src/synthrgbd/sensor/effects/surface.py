"""Material- and geometry-dependent artefacts, calibrated on real D415 data on metal parts (ROBI).

The angle-dependent tables (dropout rate, bias, spread) are measured with
scripts/robi_d415_metal.py and stored in configs/camera_d435.yaml. Effects look them up with
linear interpolation over the incidence angle, so the same definitions are used for measuring
and simulating (synthrgbd.sensor.fields).
"""
import cv2
import numpy as np

from synthrgbd.gbuffer import GBuffer
from synthrgbd.sensor.fields import correlated_noise, correlated_uniform, edge_band, incidence_deg


def _metal_mask(gbuf: GBuffer, params: dict) -> np.ndarray:
    """Metal pixels outside the silhouette band: the tables were measured on interior pixels only,
    and the band is handled by edge_artifacts (otherwise band pixels would be counted twice)."""
    metal = np.isin(gbuf.material, params["material_ids"])
    if params.get("edge_px", 0) > 0:
        metal &= ~edge_band(gbuf.instance, params["edge_px"])
    return metal


def metal_dropout(depth: np.ndarray, gbuf: GBuffer, params: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """Invalidate metal pixels with probability p(incidence angle), in spatially correlated blobs.

    The measured p(theta) has two branches: higher at head-on incidence (specular saturation)
    and rising again at grazing angles (too little light returns) [K17 Sec. 4.2.5, 4.2.8].
    """
    metal = _metal_mask(gbuf, params) & (depth > 0)
    if not metal.any():
        return depth
    theta = incidence_deg(gbuf)
    p = params.get("scale", 1.0) * np.interp(theta, params["angle_deg"], params["dropout_rate"])
    u = correlated_uniform(depth.shape, params["corr_px"], rng)
    out = depth.copy()
    out[metal & (u < p)] = 0.0
    return out


def metal_depth_error(depth: np.ndarray, gbuf: GBuffer, params: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """Valid-but-wrong depth on metal: angle-dependent bias plus correlated spread.

    Real D415 data on metal overestimates depth increasingly with incidence angle; this matters
    more for grasping than holes, because nothing marks these pixels as unreliable.
    """
    metal = _metal_mask(gbuf, params) & (depth > 0)
    if not metal.any():
        return depth
    theta = incidence_deg(gbuf)
    bias = np.interp(theta, params["angle_deg"], params["bias_mm"]) / 1000.0
    sigma = np.interp(theta, params["angle_deg"], params["sigma_mm"]) / 1000.0
    n = correlated_noise(depth.shape, params["corr_px"], rng)
    out = depth.copy()
    out[metal] += (params.get("scale", 1.0) * (bias + sigma * n))[metal]
    return out


def edge_artifacts(depth: np.ndarray, gbuf: GBuffer, params: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Dropouts and flying pixels in a band around object silhouettes.

    In the band, a pixel is dropped with p_drop; of the remaining, a fraction p_fly gets a depth
    mixed between the nearest foreground and background depth in its neighbourhood.
    """
    band = edge_band(gbuf.instance, params["width_px"]) & (depth > 0)
    if not band.any():
        return depth
    k = np.ones((2 * params["width_px"] + 1,) * 2, np.uint8)
    near = cv2.erode(np.where(depth > 0, depth, np.inf).astype(np.float32), k)
    far = cv2.dilate(depth.astype(np.float32), k)
    r = rng.random(depth.shape)
    drop = band & (r < params["p_drop"])
    fly = band & ~drop & (r < params["p_drop"] + params["p_fly"]) & np.isfinite(near)
    out = depth.copy()
    mix = rng.random(depth.shape).astype(np.float32)
    out[fly] = near[fly] + mix[fly] * (far[fly] - near[fly])
    out[drop] = 0.0
    return out
