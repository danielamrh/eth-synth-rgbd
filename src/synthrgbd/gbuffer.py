"""Ideal render output: the interface between renderer and sensor model."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GBuffer:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, z-depth in metres along the optical axis
    instance: np.ndarray     # (H, W) int32, MuJoCo geom id, -1 = background
    material: np.ndarray     # (H, W) int32, MuJoCo material id, -1 = none
    normals: np.ndarray      # (H, W, 3) float32, unit normals in camera frame (OpenCV)
    K: np.ndarray            # (3, 3) intrinsics
    T_world_cam: np.ndarray  # (4, 4) camera pose, OpenCV convention (x right, y down, z forward)
    aux: dict = field(default_factory=dict)  # extra renders, e.g. depth from the second imager

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape
