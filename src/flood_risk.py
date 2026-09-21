# ============================================================
# FLOOD RISK ENGINE
# Rule-based risk assessment for SIH prototype
# ============================================================

from typing import Dict, Any


def calculate_flood_risk(
    predicted_rain_mm: float,
    heavy_probability: float,
    current_rain_mm: float,
    rain_3h_mm: float,
    rain_6h_mm: float,
    rain_24h_mm: float,
) -> Dict[str, Any]:
    """
    Calculate a prototype flood-risk score.

    IMPORTANT:
    This is NOT a trained flood classifier.

    The score combines:
        - predicted next-hour rainfall
        - heavy-rain probability
        - current rainfall
        - 3-hour rainfall accumulation
        - 6-hour rainfall accumulation
        - 24-hour rainfall accumulation
        - rainfall persistence

    Returns:
        Dictionary containing:
            risk_level
            score
            predicted_rain_mm
            heavy_probability
            action
            factors
    """

    # ========================================================
    # 1. NORMALIZE INPUTS
    # ========================================================

    try:
        predicted_rain_mm = max(0.0, float(predicted_rain_mm))
    except (TypeError, ValueError):
        predicted_rain_mm = 0.0

    try:
        heavy_probability = float(heavy_probability)
    except (TypeError, ValueError):
        heavy_probability = 0.0

    heavy_probability = min(
        1.0,
        max(0.0, heavy_probability)
    )

    try:
        current_rain_mm = max(0.0, float(current_rain_mm))
    except (TypeError, ValueError):
        current_rain_mm = 0.0

    try:
        rain_3h_mm = max(0.0, float(rain_3h_mm))
    except (TypeError, ValueError):
        rain_3h_mm = 0.0

    try:
        rain_6h_mm = max(0.0, float(rain_6h_mm))
    except (TypeError, ValueError):
        rain_6h_mm = 0.0

    try:
        rain_24h_mm = max(0.0, float(rain_24h_mm))
    except (TypeError, ValueError):
        rain_24h_mm = 0.0

    # ========================================================
    # 2. INITIALIZE
    # ========================================================

    score = 0.0
    factors = []

    # ========================================================
    # 3. PREDICTED NEXT-HOUR RAINFALL
    #
    # Maximum contribution: 35 points
    # ========================================================

    if predicted_rain_mm >= 20.0:

        score += 35.0

        factors.append(
            f"Very high predicted rainfall "
            f"({predicted_rain_mm:.2f} mm/hour)"
        )

    elif predicted_rain_mm >= 10.0:

        score += 28.0

        factors.append(
            f"High predicted rainfall "
            f"({predicted_rain_mm:.2f} mm/hour)"
        )

    elif predicted_rain_mm >= 5.0:

        score += 18.0

        factors.append(
            f"Elevated predicted rainfall "
            f"({predicted_rain_mm:.2f} mm/hour)"
        )

    elif predicted_rain_mm >= 2.0:

        score += 10.0

        factors.append(
            f"Moderate predicted rainfall "
            f"({predicted_rain_mm:.2f} mm/hour)"
        )

    elif predicted_rain_mm > 0.0:

        score += min(
            5.0,
            predicted_rain_mm * 2.0
        )

        factors.append(
            f"Light predicted rainfall "
            f"({predicted_rain_mm:.2f} mm/hour)"
        )

    # ========================================================
    # 4. HEAVY-RAIN PROBABILITY
    #
    # Maximum contribution: 25 points
    # ========================================================

    if heavy_probability >= 0.80:

        score += 25.0

        factors.append(
            f"Very high heavy-rain probability "
            f"({heavy_probability:.1%})"
        )

    elif heavy_probability >= 0.60:

        score += 20.0

        factors.append(
            f"High heavy-rain probability "
            f"({heavy_probability:.1%})"
        )

    elif heavy_probability >= 0.40:

        score += 14.0

        factors.append(
            f"Moderate heavy-rain probability "
            f"({heavy_probability:.1%})"
        )

    elif heavy_probability >= 0.20:

        score += 7.0

        factors.append(
            f"Non-negligible heavy-rain probability "
            f"({heavy_probability:.1%})"
        )

    # ========================================================
    # 5. CURRENT RAINFALL
    #
    # Maximum contribution: 10 points
    # ========================================================

    if current_rain_mm >= 10.0:

        score += 10.0

        factors.append(
            f"Very high current rainfall "
            f"({current_rain_mm:.2f} mm)"
        )

    elif current_rain_mm >= 5.0:

        score += 8.0

        factors.append(
            f"High current rainfall "
            f"({current_rain_mm:.2f} mm)"
        )

    elif current_rain_mm >= 2.0:

        score += 5.0

        factors.append(
            f"Moderate current rainfall "
            f"({current_rain_mm:.2f} mm)"
        )

    elif current_rain_mm > 0.0:

        score += min(
            3.0,
            current_rain_mm
        )

    # ========================================================
    # 6. 3-HOUR ACCUMULATION
    #
    # Maximum contribution: 10 points
    # ========================================================

    if rain_3h_mm >= 30.0:

        score += 10.0

        factors.append(
            f"Very high 3-hour accumulation "
            f"({rain_3h_mm:.2f} mm)"
        )

    elif rain_3h_mm >= 15.0:

        score += 7.0

        factors.append(
            f"High 3-hour accumulation "
            f"({rain_3h_mm:.2f} mm)"
        )

    elif rain_3h_mm >= 5.0:

        score += 4.0

        factors.append(
            f"Elevated 3-hour accumulation "
            f"({rain_3h_mm:.2f} mm)"
        )

    elif rain_3h_mm > 0.0:

        score += min(
            2.0,
            rain_3h_mm * 0.2
        )

    # ========================================================
    # 7. 6-HOUR ACCUMULATION
    #
    # Maximum contribution: 8 points
    # ========================================================

    if rain_6h_mm >= 50.0:

        score += 8.0

        factors.append(
            f"Very high 6-hour accumulation "
            f"({rain_6h_mm:.2f} mm)"
        )

    elif rain_6h_mm >= 25.0:

        score += 6.0

        factors.append(
            f"High 6-hour accumulation "
            f"({rain_6h_mm:.2f} mm)"
        )

    elif rain_6h_mm >= 10.0:

        score += 3.0

        factors.append(
            f"Elevated 6-hour accumulation "
            f"({rain_6h_mm:.2f} mm)"
        )

    elif rain_6h_mm > 0.0:

        score += min(
            1.5,
            rain_6h_mm * 0.1
        )

    # ========================================================
    # 8. 24-HOUR ACCUMULATION
    #
    # Maximum contribution: 12 points
    # ========================================================

    if rain_24h_mm >= 100.0:

        score += 12.0

        factors.append(
            f"Very high 24-hour accumulation "
            f"({rain_24h_mm:.2f} mm)"
        )

    elif rain_24h_mm >= 50.0:

        score += 9.0

        factors.append(
            f"High 24-hour accumulation "
            f"({rain_24h_mm:.2f} mm)"
        )

    elif rain_24h_mm >= 25.0:

        score += 5.0

        factors.append(
            f"Elevated 24-hour accumulation "
            f"({rain_24h_mm:.2f} mm)"
        )

    elif rain_24h_mm > 0.0:

        score += min(
            3.0,
            rain_24h_mm * 0.06
        )

    # ========================================================
    # 9. PERSISTENCE
    #
    # Maximum contribution: 10 points
    # ========================================================

    if (
        current_rain_mm >= 5.0
        and predicted_rain_mm >= 5.0
    ):

        score += 10.0

        factors.append(
            "Persistent rainfall expected to continue"
        )

    elif (
        current_rain_mm >= 2.0
        and predicted_rain_mm >= 2.0
    ):

        score += 5.0

        factors.append(
            "Rainfall is expected to continue"
        )

    elif (
        rain_3h_mm >= 5.0
        and predicted_rain_mm >= 2.0
    ):

        score += 3.0

        factors.append(
            "Recent rainfall combined with "
            "continued rainfall forecast"
        )

    # ========================================================
    # 10. SCORE NORMALIZATION
    # ========================================================

    score = min(
        100.0,
        max(0.0, score)
    )

    # ========================================================
    # 11. RISK CATEGORY
    #
    # 0 - 14     LOW
    # 15 - 34    MODERATE
    # 35 - 74    HIGH
    # 75 - 100   CRITICAL
    #
    # Prototype thresholds only.
    # These are NOT official flood-warning thresholds.
    # ========================================================

    if score >= 75.0:

        risk_level = "CRITICAL"

        action = (
            "Critical rainfall/flood-risk conditions. "
            "Immediate attention is recommended. "
            "Closely monitor rainfall, accumulation, "
            "and local flood conditions."
        )

    elif score >= 35.0:

        risk_level = "HIGH"

        action = (
            "High rainfall/flood risk. "
            "Monitor conditions closely and prepare "
            "appropriate warnings and response measures."
        )

    elif score >= 15.0:

        risk_level = "MODERATE"

        action = (
            "Moderate rainfall/flood risk. "
            "Continue monitoring rainfall intensity, "
            "recent accumulation, and the next-hour forecast."
        )

    else:

        risk_level = "LOW"

        action = (
            "Low immediate rainfall/flood risk "
            "based on available model inputs."
        )

    # ========================================================
    # 12. DEFAULT FACTOR
    # ========================================================

    if not factors:

        factors.append(
            "No significant rainfall risk indicators detected."
        )

    # Remove duplicate factors while preserving order.
    factors = list(
        dict.fromkeys(factors)
    )

    # ========================================================
    # 13. RETURN RESULT
    # ========================================================

    return {
        "risk_level": risk_level,
        "score": round(score, 2),
        "predicted_rain_mm": round(
            predicted_rain_mm,
            2
        ),
        "heavy_probability": round(
            heavy_probability,
            4
        ),
        "action": action,
        "factors": factors,
    }