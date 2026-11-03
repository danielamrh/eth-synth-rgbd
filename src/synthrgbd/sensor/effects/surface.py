"""Material- and geometry-dependent dropouts (aluminium is the hard case)."""
import numpy as np

from synthrgbd.gbuffer import GBuffer


def grazing_dropout(depth: np.ndarray, gbuf: GBuffer, params: dict,
                    rng: np.random.Generator) -> np.ndarray:
    """Dropout probability rising with the angle between view ray and surface normal."""
    raise NotImplementedError("Thursday: sensor model")


def specular_dropout(depth: np.ndarray, gbuf: GBuffer, params: dict,
                     rng: np.random.Generator) -> np.ndarray:
    """Dropout on specular materials where the projector's mirror reflection misses the imager."""
    raise NotImplementedError("Thursday: sensor model")


def edge_artifacts(depth: np.ndarray, gbuf: GBuffer, params: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Flying pixels / dropouts in a band around depth discontinuities."""
    raise NotImplementedError("Thursday: sensor model")
