"""Per-pixel helper fields shared by the sensor effects and the ROBI evaluation.

Using the same definitions for measuring real data (scripts/robi_d415_metal.py) and for
simulating it keeps "edge pixel" and "incidence angle" consistent between both.
"""
import cv2
import numpy as np

from synthrgbd.gbuffer import GBuffer


def correlated_noise(shape, corr_px: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-std Gaussian field, smoothed with a Gaussian of sigma corr_px pixels.

    Stereo noise is spatially smooth ("bumps", "egg carton"), not i.i.d. per pixel.
    """
    g = rng.standard_normal(shape).astype(np.float32)
    if corr_px > 0:
        g = cv2.GaussianBlur(g, (0, 0), corr_px)
        g /= g.std() + 1e-12
    return g


def correlated_uniform(shape, corr_px: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially correlated field with exactly uniform marginals on (0, 1).

    Thresholding it at p drops a fraction p of pixels, but in blobs instead of salt-and-pepper.
    """
    g = correlated_noise(shape, corr_px, rng).ravel()
    ranks = np.empty(g.size)
    ranks[np.argsort(g, kind="stable")] = np.arange(g.size)
    return ((ranks + 0.5) / g.size).reshape(shape)


def edge_band(instance: np.ndarray, width: int) -> np.ndarray:
    """Pixels within `width` px of a change in instance id (object/object or object/background)."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    inst = (instance + 1).astype(np.float32)
    return cv2.dilate(inst, k) != cv2.erode(inst, k)


def incidence_deg(gbuf: GBuffer) -> np.ndarray:
    """Angle between the viewing ray and the surface normal, in degrees (0 = head-on)."""
    h, w = gbuf.depth.shape
    K = gbuf.K
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    cos = np.abs(np.sum(gbuf.normals * ray, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
