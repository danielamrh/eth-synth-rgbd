"""Assemble the RealSense D435 sensor model from configs/camera_d435.yaml.

Order follows the physics: geometry first (what the right imager cannot see, range),
then material/silhouette failures, then matching noise and quantisation of the output.
"""
from pathlib import Path

import yaml

from synthrgbd.sensor.effects.stereo import (disparity_noise_and_quantisation, range_limits,
                                             stereo_shadow)
from synthrgbd.sensor.effects.surface import edge_artifacts, metal_depth_error, metal_dropout
from synthrgbd.sensor.pipeline import SensorModel

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "camera_d435.yaml"


def _v(entry):
    """Unwrap {value: x, source: ...} entries; plain values pass through."""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


def load_config(path=DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_d435(metal_material_ids, config_path=DEFAULT_CONFIG) -> SensorModel:
    """D435 model; metal_material_ids are the renderer's material ids treated as metal."""
    c = load_config(config_path)
    st, rg, ef = c["stereo"], c["range"], c["effects"]
    B = _v(st["baseline_m"])
    metal = ef["metal"]
    edge_px = _v(ef["edges"]["width_px"])
    return SensorModel([
        ("stereo_shadow", stereo_shadow, {"baseline_m": B}),
        # No hard maximum range in the datasheet; far depth degrades via the z^2 noise law instead.
        ("range", range_limits, {"min_z_m": _v(rg["min_z_m"]), "max_z_m": _v(rg.get("max_z_m", float("inf")))}),
        ("metal_dropout", metal_dropout, {
            "material_ids": list(metal_material_ids), "angle_deg": metal["angle_deg"],
            "dropout_rate": metal["dropout_rate"], "corr_px": _v(metal["corr_px"]), "edge_px": edge_px}),
        ("edge_artifacts", edge_artifacts, {
            "width_px": edge_px, "p_drop": _v(ef["edges"]["p_drop"]),
            "p_fly": _v(ef["edges"]["p_fly"])}),
        ("metal_depth_error", metal_depth_error, {
            "material_ids": list(metal_material_ids), "angle_deg": metal["angle_deg"],
            "bias_mm": metal["bias_mm"], "sigma_mm": metal["sigma_mm"], "corr_px": _v(metal["corr_px"]),
            "edge_px": edge_px}),
        ("disparity_noise", disparity_noise_and_quantisation, {
            "baseline_m": B, "sigma_px": _v(st["subpixel_rms_px"]), "corr_px": _v(st["noise_correlation_px"]),
            "step_px": _v(st["disparity_step_px"]), "depth_unit_m": _v(st["depth_unit_m"])}),
    ])
