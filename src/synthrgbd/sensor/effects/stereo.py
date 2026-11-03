"""Depth <-> disparity for an active stereo camera, and effects in disparity space.

Working in disparity d = f * B / z makes the z^2 growth of depth error fall out
of a constant disparity error instead of being imposed by hand.
"""
import numpy as np

from synthrgbd.gbuffer import GBuffer


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


def disparity_noise_and_quantisation(depth: np.ndarray, gbuf: GBuffer, params: dict,
                                     rng: np.random.Generator) -> np.ndarray:
    """Gaussian disparity noise (sigma_px) followed by subpixel quantisation (step_px)."""
    raise NotImplementedError("Thursday: sensor model")


def stereo_shadow(depth: np.ndarray, gbuf: GBuffer, params: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """Invalidate pixels not visible from the second imager (needs gbuf.aux['depth_right'])."""
    raise NotImplementedError("Thursday: sensor model")
