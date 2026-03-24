"""
Earth pressure coefficients — Rankine and Coulomb theories.

References:
- IS 9527 (Part 1): Design of Sheet Pile Walls
- Bowles, Foundation Analysis and Design, 5th Ed
- Das, Principles of Geotechnical Engineering
"""

import math
try:
    from engine.models import PressureTheory
except ImportError:
    from models import PressureTheory


def ka_rankine(phi_deg: float) -> float:
    """
    Rankine active earth pressure coefficient.
    Ka = tan²(45 - φ/2)
    
    Args:
        phi_deg: Effective friction angle in degrees
    Returns:
        Ka value
    """
    if phi_deg == 0:
        return 1.0
    phi = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi / 2) ** 2


def kp_rankine(phi_deg: float) -> float:
    """
    Rankine passive earth pressure coefficient.
    Kp = tan²(45 + φ/2)
    
    Args:
        phi_deg: Effective friction angle in degrees
    Returns:
        Kp value
    """
    if phi_deg == 0:
        return 1.0
    phi = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi / 2) ** 2


def ka_coulomb(phi_deg: float, delta_deg: float, alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """
    Coulomb active earth pressure coefficient.
    
    Ka = sin²(α + φ) / [sin²(α) × sin(α - δ) × (1 + √(sin(φ+δ)×sin(φ-β) / sin(α-δ)×sin(α+β)))²]
    
    Args:
        phi_deg:   Effective friction angle (degrees)
        delta_deg: Wall friction angle (degrees)
        alpha_deg: Wall inclination from horizontal (90 = vertical)
        beta_deg:  Backfill slope angle (0 = horizontal)
    Returns:
        Ka value
    """
    if phi_deg == 0:
        return 1.0

    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    # Numerator
    num = math.sin(alpha + phi) ** 2

    # Terms in denominator
    sin2_alpha = math.sin(alpha) ** 2
    sin_alpha_minus_delta = math.sin(alpha - delta)

    # Check feasibility: φ - β must be ≥ 0
    if phi_deg < beta_deg:
        raise ValueError(f"φ ({phi_deg}°) must be ≥ β ({beta_deg}°) for Coulomb active")

    inner_sqrt_num = math.sin(phi + delta) * math.sin(phi - beta)
    inner_sqrt_den = math.sin(alpha - delta) * math.sin(alpha + beta)

    if inner_sqrt_den == 0:
        raise ValueError("Division by zero in Coulomb Ka — check wall/backfill angles")

    sqrt_term = math.sqrt(inner_sqrt_num / inner_sqrt_den)
    bracket = (1 + sqrt_term) ** 2

    den = sin2_alpha * sin_alpha_minus_delta * bracket

    return num / den


def kp_coulomb(phi_deg: float, delta_deg: float, alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """
    Coulomb passive earth pressure coefficient.
    
    Kp = sin²(α - φ) / [sin²(α) × sin(α + δ) × (1 - √(sin(φ+δ)×sin(φ+β) / sin(α+δ)×sin(α+β)))²]
    
    Note: Coulomb Kp overestimates passive pressure for δ > φ/3.
    For high δ values, log-spiral or Caquot-Kerisel tables are more appropriate.
    Use with caution — this module flags a warning when δ > φ/3.
    
    Args:
        phi_deg:   Effective friction angle (degrees)
        delta_deg: Wall friction angle (degrees)
        alpha_deg: Wall inclination from horizontal (90 = vertical)
        beta_deg:  Backfill slope angle (0 = horizontal)
    Returns:
        Kp value
    """
    if phi_deg == 0:
        return 1.0

    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    # Warning for high wall friction
    if delta_deg > phi_deg / 3.0:
        import warnings
        warnings.warn(
            f"δ ({delta_deg}°) > φ/3 ({phi_deg/3:.1f}°): Coulomb Kp overestimates passive pressure. "
            f"Consider using Caquot-Kerisel tables or log-spiral method.",
            UserWarning
        )

    num = math.sin(alpha - phi) ** 2

    sin2_alpha = math.sin(alpha) ** 2
    sin_alpha_plus_delta = math.sin(alpha + delta)

    inner_sqrt_num = math.sin(phi + delta) * math.sin(phi + beta)
    inner_sqrt_den = math.sin(alpha + delta) * math.sin(alpha + beta)

    if inner_sqrt_den == 0:
        raise ValueError("Division by zero in Coulomb Kp — check wall/backfill angles")

    sqrt_term = math.sqrt(inner_sqrt_num / inner_sqrt_den)
    bracket = (1 - sqrt_term) ** 2

    if bracket == 0:
        raise ValueError("Denominator bracket is zero in Coulomb Kp — check angles")

    den = sin2_alpha * sin_alpha_plus_delta * bracket

    return num / den


def get_ka(phi_deg: float, theory: PressureTheory, delta_deg: float = 0.0,
           alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Get Ka based on selected theory."""
    if theory == PressureTheory.RANKINE:
        return ka_rankine(phi_deg)
    else:
        return ka_coulomb(phi_deg, delta_deg, alpha_deg, beta_deg)


def get_kp(phi_deg: float, theory: PressureTheory, delta_deg: float = 0.0,
           alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Get Kp based on selected theory."""
    if theory == PressureTheory.RANKINE:
        return kp_rankine(phi_deg)
    else:
        return kp_coulomb(phi_deg, delta_deg, alpha_deg, beta_deg)


# === Caquot-Kerisel Kp lookup (interpolated) ===
# More accurate than Coulomb for passive pressure with wall friction.
# Source: Caquot & Kerisel (1948), tabulated in Bowles Table 11-4

# Table: Kp values for δ/φ ratios and φ values
# Format: _CK_KP_TABLE[phi_deg][(delta/phi ratio)] = Kp
# Common ratios: 0, 1/3, 1/2, 2/3, 1.0

_CK_KP_TABLE = {
    # phi: {delta/phi: Kp}
    20: {0.0: 2.04, 0.33: 2.48, 0.50: 2.75, 0.67: 3.01, 1.0: 3.70},
    25: {0.0: 2.46, 0.33: 3.15, 0.50: 3.59, 0.67: 4.07, 1.0: 5.34},
    30: {0.0: 3.00, 0.33: 4.09, 0.50: 4.80, 0.67: 5.63, 1.0: 7.95},
    35: {0.0: 3.69, 0.33: 5.45, 0.50: 6.65, 0.67: 8.11, 1.0: 12.6},
    40: {0.0: 4.60, 0.33: 7.50, 0.50: 9.60, 0.67: 12.3, 1.0: 21.8},
    45: {0.0: 5.83, 0.33: 10.8, 0.50: 14.6, 0.67: 19.8, 1.0: 41.4},
}


def kp_caquot_kerisel(phi_deg: float, delta_deg: float) -> float:
    """
    Caquot-Kerisel passive pressure coefficient (interpolated from tables).
    More accurate than Coulomb for δ > 0.
    Valid for vertical wall (α=90°) and horizontal backfill (β=0°).
    
    Args:
        phi_deg:   Effective friction angle (degrees)
        delta_deg: Wall friction angle (degrees)
    Returns:
        Kp value (interpolated)
    """
    if phi_deg == 0:
        return 1.0

    # Clamp delta/phi ratio
    if phi_deg > 0:
        ratio = min(abs(delta_deg) / phi_deg, 1.0)
    else:
        ratio = 0.0

    # Get bounding phi values from table
    available_phis = sorted(_CK_KP_TABLE.keys())

    if phi_deg <= available_phis[0]:
        phi_low = phi_high = available_phis[0]
    elif phi_deg >= available_phis[-1]:
        phi_low = phi_high = available_phis[-1]
    else:
        for i in range(len(available_phis) - 1):
            if available_phis[i] <= phi_deg <= available_phis[i + 1]:
                phi_low = available_phis[i]
                phi_high = available_phis[i + 1]
                break

    def _interp_ratio(phi_key, r):
        """Interpolate Kp for a given phi at ratio r."""
        ratios = sorted(_CK_KP_TABLE[phi_key].keys())
        kp_vals = [_CK_KP_TABLE[phi_key][rt] for rt in ratios]

        if r <= ratios[0]:
            return kp_vals[0]
        if r >= ratios[-1]:
            return kp_vals[-1]

        for i in range(len(ratios) - 1):
            if ratios[i] <= r <= ratios[i + 1]:
                t = (r - ratios[i]) / (ratios[i + 1] - ratios[i])
                return kp_vals[i] + t * (kp_vals[i + 1] - kp_vals[i])
        return kp_vals[-1]

    kp_low = _interp_ratio(phi_low, ratio)

    if phi_low == phi_high:
        return kp_low

    kp_high = _interp_ratio(phi_high, ratio)

    # Interpolate between phi values
    t = (phi_deg - phi_low) / (phi_high - phi_low)
    return kp_low + t * (kp_high - kp_low)
