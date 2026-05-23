from __future__ import annotations

import json
import logging
import math
from typing import Any

from langchain_core.tools import tool


logger = logging.getLogger(__name__)

DEFAULT_ASCENT_METERS_PER_FLAT_KM = 100.0


def _finite_float(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def calculate_flat_kilometers(
    distance_km: float,
    elevation_gain_m: float,
    ascent_m_per_flat_km: float = DEFAULT_ASCENT_METERS_PER_FLAT_KM,
) -> dict[str, Any]:
    """Calculate rough equivalent flat kilometers for a trail route."""
    distance = _finite_float("distance_km", distance_km)
    elevation = _finite_float("elevation_gain_m", elevation_gain_m)
    conversion = _finite_float("ascent_m_per_flat_km", ascent_m_per_flat_km)

    if distance < 0:
        raise ValueError("distance_km must be non-negative.")
    if elevation < 0:
        raise ValueError("elevation_gain_m must be non-negative.")
    if conversion <= 0:
        raise ValueError("ascent_m_per_flat_km must be greater than zero.")

    elevation_flat_km = elevation / conversion
    flat_km = distance + elevation_flat_km
    return {
        "distance_km": round(distance, 2),
        "elevation_gain_m": round(elevation, 1),
        "ascent_m_per_flat_km": round(conversion, 1),
        "elevation_flat_km": round(elevation_flat_km, 2),
        "flat_km_equivalent": round(flat_km, 2),
        "formula": "distance_km + elevation_gain_m / ascent_m_per_flat_km",
        "assumption": "Rough heuristic: 100 m ascent counts as 1 flat km by default.",
    }


@tool("flat_kilometer_equivalent", parse_docstring=True)
def flat_kilometer_equivalent(
    distance_km: float,
    elevation_gain_m: float,
    ascent_m_per_flat_km: float = DEFAULT_ASCENT_METERS_PER_FLAT_KM,
) -> str:
    """Calculate rough equivalent flat kilometers from distance and elevation gain.

    Uses ``distance_km + elevation_gain_m / ascent_m_per_flat_km``. The default
    conversion treats 100 meters of ascent as one extra flat kilometer. This is a
    simple training-load/course-effort heuristic, not a precise pacing model.

    Args:
        distance_km: Route distance in kilometers.
        elevation_gain_m: Positive elevation gain/ascent in meters.
        ascent_m_per_flat_km: Meters of ascent counted as one flat kilometer.
            Defaults to 100.
    """
    logger.info(
        "[effort] tool_call distance_km=%r elevation_gain_m=%r ascent_m_per_flat_km=%r",
        distance_km,
        elevation_gain_m,
        ascent_m_per_flat_km,
    )
    try:
        return json.dumps(
            calculate_flat_kilometers(
                distance_km,
                elevation_gain_m,
                ascent_m_per_flat_km,
            )
        )
    except ValueError as exc:
        return json.dumps(
            {
                "error": f"Flat kilometer calculation failed: {exc}",
                "category": "validation",
                "retryable": False,
                "inputs": {
                    "distance_km": distance_km,
                    "elevation_gain_m": elevation_gain_m,
                    "ascent_m_per_flat_km": ascent_m_per_flat_km,
                },
            },
            default=str,
        )


EFFORT_TOOLS = [flat_kilometer_equivalent]
