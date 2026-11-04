"""Top-down parallel-jaw grasp planning from a single depth image.

Deliberately simple and explainable, typical of a model-free bin-picking baseline:
  1. back-project valid depth to world points (camera extrinsics are calibrated),
  2. fit the table plane with RANSAC, keep points more than `min_height` above it,
  3. split them into clusters: connected image regions without depth jumps
     (or, in the `masks` variant, the ground-truth instance masks = a perfect segmenter),
  4. take the highest cluster that fits the gripper; grasp at the median xy of its points,
     close across the minor PCA axis, open to the measured width plus clearance,
     put the finger tips `grip_depth` below the local top surface (points under the jaws).
Every step uses only depth, so holes, flying pixels and biased depth all reach the grasp.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from synthrgbd.gbuffer import GBuffer

PARAMS = {
    "min_height": 0.006,    # [m] above the table plane counts as object
    "jump": 0.008,          # [m] depth step between neighbouring pixels that splits clusters
    "min_pixels": 150,      # smaller clusters are noise
    "clearance": 0.012,     # [m] extra opening on each side
    "max_open": 0.085,      # [m] gripper stroke
    "grip_depth": 0.015,    # [m] finger tips below the top surface
    "tip_floor": 0.003,     # [m] never aim the tips lower than this above the table
    "local_halfwidth": 0.015,  # [m] points within this distance of the jaw plane set the grip height
    "ransac_iters": 200,
    "ransac_tol": 0.003,
}


@dataclass
class Grasp:
    x: float
    y: float
    z_tip: float       # world height of the finger tips at the grasp
    yaw: float         # gripper closes along (-sin yaw, cos yaw)
    width: float       # jaw opening before closing
    z_top: float       # estimated top of the object, for the approach
    n_points: int

    def as_dict(self):
        return asdict(self)


def backproject(depth: np.ndarray, gbuf: GBuffer) -> np.ndarray:
    """World coordinates per pixel (H, W, 3); invalid pixels give NaN."""
    h, w = depth.shape
    K = gbuf.K
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = np.where(depth > 0, depth, np.nan)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    T = gbuf.T_world_cam
    return pc @ T[:3, :3].T + T[:3, 3]


def fit_table(points: np.ndarray, rng: np.random.Generator, iters: int, tol: float):
    """RANSAC plane (n, d) with n pointing up, n.p + d = 0, refined by least squares on inliers."""
    best, best_count = None, -1
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs(points @ n - n @ s[0]) < tol).sum())
        if count > best_count:
            best, best_count = (n, -n @ s[0]), count
    n, d = best
    inl = points[np.abs(points @ n + d) < tol]
    c = inl.mean(0)
    n = np.linalg.eigh(np.cov((inl - c).T))[1][:, 0]   # normal = smallest-variance direction
    if n[2] < 0:
        n = -n
    return n, -n @ c


def clusters_from_depth(height: np.ndarray, depth: np.ndarray, p: dict) -> list[np.ndarray]:
    """Connected regions above the table, split at depth jumps (touching parts may merge)."""
    obj = np.nan_to_num(height, nan=-1.0) > p["min_height"]
    z = np.where(depth > 0, depth, 0).astype(np.float32)
    jump = np.zeros_like(obj)
    jump[:, 1:] |= np.abs(np.diff(z, axis=1)) > p["jump"]
    jump[1:, :] |= np.abs(np.diff(z, axis=0)) > p["jump"]
    n, lab, stats, _ = cv2.connectedComponentsWithStats((obj & ~jump).astype(np.uint8), connectivity=4)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= p["min_pixels"]]
    return [lab == k for k in keep]


def clusters_from_masks(height, instance_masks: list[np.ndarray], p: dict) -> list[np.ndarray]:
    obj = np.nan_to_num(height, nan=-1.0) > p["min_height"]
    return [m & obj for m in instance_masks]


def plan_grasp(depth: np.ndarray, gbuf: GBuffer, rng: np.random.Generator,
               instance_masks: list[np.ndarray] | None = None, params: dict = PARAMS,
               plane: tuple[np.ndarray, float] | None = None, min_points: int = 1000):
    """Returns (Grasp or None, info dict). info explains a failure to plan; on success
    info["region"] is the pixel mask of the chosen cluster (used only for evaluation).
    plane = (n, d) skips the RANSAC table fit (used where the support surface is known)."""
    p = params
    P = backproject(depth, gbuf)
    valid = np.isfinite(P[..., 2])
    pts = P[valid]
    if len(pts) < min_points:
        return None, {"reason": "too_few_points"}
    if plane is None:
        sub = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
        n, d = fit_table(sub, rng, p["ransac_iters"], p["ransac_tol"])
    else:
        n, d = plane
    height = np.full(depth.shape, np.nan)
    height[valid] = pts @ n + d

    if instance_masks is None:
        regions = clusters_from_depth(height, depth, p)
    else:
        regions = clusters_from_masks(height, instance_masks, p)

    cands = []
    for r in regions:
        if r.sum() < p["min_pixels"]:
            continue
        h = height[r]
        cands.append((float(np.percentile(h, 95)), r))
    if not cands:
        return None, {"reason": "no_cluster"}
    cands.sort(key=lambda c: -c[0])

    for z_top_h, r in cands:
        q = P[r]
        # Project onto the table plane frame: in-plane xy (world xy, plane is near-horizontal).
        xy = q[:, :2]
        c = np.median(xy, axis=0)
        evals, evecs = np.linalg.eigh(np.cov((xy - c).T))
        minor = evecs[:, 0]
        proj = (xy - c) @ minor
        lo, hi = np.percentile(proj, [2, 98])
        width = (hi - lo) + 2 * p["clearance"]
        if width > p["max_open"]:
            continue
        c = c + minor * (lo + hi) / 2           # centre between the two measured faces
        # Grip height from the points under the jaws, not from the whole cluster: a part leaning on
        # another one is highest at one end, and the global top would put the jaws above its centre.
        major = evecs[:, 1]
        near = np.abs((xy - c) @ major) < p["local_halfwidth"]
        h_loc = height[r][near] if near.sum() >= 20 else height[r]
        z_top_h = float(np.percentile(h_loc, 95))
        table_z = -(d + n[0] * c[0] + n[1] * c[1]) / n[2]
        z_top = table_z + z_top_h
        z_tip = table_z + max(p["tip_floor"], z_top_h - p["grip_depth"])
        yaw = float(np.arctan2(-minor[0], minor[1]))
        return Grasp(float(c[0]), float(c[1]), float(z_tip), yaw, float(width), float(z_top), int(r.sum())), \
            {"reason": "ok", "n_clusters": len(cands), "region": r}
    return None, {"reason": "nothing_fits_gripper", "n_clusters": len(cands)}
