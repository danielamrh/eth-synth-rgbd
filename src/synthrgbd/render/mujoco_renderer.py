"""Render a GBuffer from a MuJoCo camera."""
import os
import sys

# EGL (headless GPU) exists only on Linux; Windows/macOS use MuJoCo's default backend.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from synthrgbd.gbuffer import GBuffer

# MuJoCo cameras look along -z with y up; OpenCV looks along +z with y down.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics(model: mujoco.MjModel, cam_id: int, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics from MuJoCo's vertical field of view (square pixels)."""
    fy = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    return np.array([[fy, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1]])


def camera_pose(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ _MJ_TO_CV
    T[:3, 3] = data.cam_xpos[cam_id]
    return T


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame unit normals from back-projected depth (central differences).

    Normals point towards the camera. Pixels at depth edges get noisy normals;
    that is acceptable because the edge effect handles silhouettes separately.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], axis=-1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    flip = np.sum(n * P, axis=-1) > 0
    n[flip] *= -1
    return n.astype(np.float32)


def render_gbuffer(model: mujoco.MjModel, data: mujoco.MjData, camera: str,
                   width: int = 640, height: int = 480) -> GBuffer:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"camera '{camera}' not found")
    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(data, camera)
        rgb = r.render()
        r.enable_depth_rendering()
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()

    is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    instance = np.where(is_geom, seg[..., 0], -1).astype(np.int32)
    material = np.where(instance >= 0, model.geom_matid[np.maximum(instance, 0)], -1).astype(np.int32)
    # Background shows the far clipping plane; a real camera returns nothing there.
    depth[instance < 0] = 0.0
    K = intrinsics(model, cam_id, width, height)
    normals = normals_from_depth(depth, K)
    normals[instance < 0] = 0.0
    return GBuffer(rgb=rgb, depth=depth, instance=instance, material=material,
                   normals=normals, K=K,
                   T_world_cam=camera_pose(data, cam_id))
