"""Food preparation recommendation logic."""

from __future__ import annotations

import math


FOOD_BUFFER_RATE = 0.05


def calculate_food_requirement(predicted_attendance: int) -> tuple[int, int, str]:
    """Return portions, potential surplus, and risk for a 5% planning buffer.

    Potential surplus is an estimate. Actual food waste can only be measured
    after the meal has been served.
    """
    try:
        predicted = int(predicted_attendance)
    except (TypeError, ValueError) as exc:
        raise ValueError("predicted_attendance must be a whole number") from exc

    if predicted < 0:
        raise ValueError("predicted_attendance cannot be negative")

    recommended_food = math.ceil(predicted * (1 + FOOD_BUFFER_RATE))
    potential_surplus = recommended_food - predicted
    surplus_ratio = potential_surplus / predicted if predicted else 0.0

    # A small tolerance prevents ceil() from turning a 5% planning policy into
    # Medium risk solely because a fractional portion was rounded upward.
    if surplus_ratio <= 0.055:
        wastage_risk = "Low"
    elif surplus_ratio <= 0.10:
        wastage_risk = "Medium"
    else:
        wastage_risk = "High"

    return recommended_food, potential_surplus, wastage_risk
