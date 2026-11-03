"""Composable depth sensor model.

Each effect is a function ``effect(depth, gbuf, params, rng) -> depth`` that may
only degrade depth (add noise, set pixels to 0 = invalid). Effects run in the
order given, so physically earlier stages (geometry: shadow, range) come before
matching artefacts and noise.
"""
from typing import Callable

import numpy as np

from synthrgbd.gbuffer import GBuffer

Effect = Callable[[np.ndarray, GBuffer, dict, np.random.Generator], np.ndarray]


class SensorModel:
    def __init__(self, effects: list[tuple[str, Effect, dict]]):
        names = [name for name, _, _ in effects]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate effect names: {names}")
        self.effects = effects

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.effects]

    def without(self, *names: str) -> "SensorModel":
        """Same model with the given effects removed (for ablations)."""
        unknown = set(names) - set(self.names)
        if unknown:
            raise KeyError(f"unknown effects: {sorted(unknown)}")
        return SensorModel([e for e in self.effects if e[0] not in names])

    def apply(self, gbuf: GBuffer, rng: np.random.Generator) -> np.ndarray:
        depth = gbuf.depth.copy()
        for _, effect, params in self.effects:
            depth = effect(depth, gbuf, params, rng)
        return depth
