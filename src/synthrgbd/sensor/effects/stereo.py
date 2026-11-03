"""Depth <-> disparity for an active stereo camera, and effects in disparity space.

Working in disparity d = f * B / z makes the z^2 growth of depth error fall out
of a constant disparity error instead of being imposed by hand [K17 Eq. 2].
The left imager is the reference; the right imager sits at +B along the camera x axis,
so a point at column u appears at u - d in the right image.
"""
import numpy as np

from synthrgbd.gbuffer import GBuffer
from synthrgbd.sensor.fields import correlated_noise


def depth_to_disparity(depth: np.ndarray, f_px: float, baseline_m: float) -> np.ndarray:
    d = np.zeros_like(depth)
    valid = depth > 0
    d[valid] = f_px * baseline_m / depth[valid]
    return d


def disparity_to_depth(disp: np.ndarray, f_px: float, baseline_m: float) -> np.ndarray:
    z = np.zeros_like(disp)
    valid = disp > 0
    z[valid] = f_px * baseline_m / disp[valid]
    return z


def range_limits(depth: np.ndarray, gbuf: GBuffer, params: dict,
                 rng: np.random.Generator) -> np.ndarray:
    """Invalidate depth outside [min_z_m, max_z_m]."""
    out = depth.copy()
    out[(out < params["min_z_m"]) | (out > params["max_z_m"])] = 0.0
    return out


def stereo_shadow(depth: np.ndarray, gbuf: GBuffer, params: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """Invalidate pixels the right imager cannot see (occlusion) or that fall outside its image.

    Scanline test on the ideal geometry: pixel u maps to x_r = u - d(u) in the right image.
    It is visible iff no pixel further right maps to the same or a smaller x_r, i.e.
    x_r(u) < min_{u' > u} x_r(u'). Pixels with x_r < 0 form the left invalid band [DS Sec. 4.5].
    """
    f = gbuf.K[0, 0]
    d = depth_to_disparity(gbuf.depth, f, params["baseline_m"])
    w = d.shape[1]
    x_r = np.arange(w, dtype=np.float64)[None, :] - d
    right_min = np.minimum.accumulate(x_r[:, ::-1], axis=1)[:, ::-1]
    right_min_excl = np.concatenate([right_min[:, 1:], np.full((d.shape[0], 1), np.inf)], axis=1)
    hidden = (x_r > right_min_excl) | (x_r < 0)
    out = depth.copy()
    out[hidden] = 0.0
    return out


def disparity_noise_and_quantisation(depth: np.ndarray, gbuf: GBuffer, params: dict,
                                     rng: np.random.Generator) -> np.ndarray:
    """Spatially correlated disparity noise, disparity quantisation, then depth-unit quantisation.

    sigma_px: disparity RMS ("subpixel") [BKM 3e, PRJ Sec. 3]; corr_px: noise correlation length;
    step_px: disparity quantisation step [PRJ Sec. 3]; depth_unit_m: output depth unit [BKM 8a].
    """
    f, B = gbuf.K[0, 0], params["baseline_m"]
    valid = depth > 0
    d = depth_to_disparity(depth, f, B)
    d = d + params["sigma_px"] * correlated_noise(d.shape, params["corr_px"], rng)
    d = np.round(d / params["step_px"]) * params["step_px"]
    d[~valid | (d <= 0)] = 0.0
    z = disparity_to_depth(d, f, B)
    unit = params["depth_unit_m"]
    return (np.round(z / unit) * unit).astype(np.float32)
