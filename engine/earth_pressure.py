"""
Earth pressure computation through layered soil profile.

Computes at each depth increment:
- Effective vertical stress (σ'v) accounting for soil layers and water table
- Active earth pressure (σ'ah) 
- Passive earth pressure (σ'ph)
- Hydrostatic water pressure (u)
- Surcharge-induced lateral pressure

Sign convention:
- Pressures acting toward the wall (active side) are POSITIVE
- Pressures acting away from the wall (passive side) are POSITIVE
- The net pressure diagram is computed separately in the analysis module

References:
- IS 9527 (Part 1): Sheet pile walls
- IS 14458 (Part 1): Retaining walls
- Bowles, Foundation Analysis and Design, 5th Ed, Ch 11
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import sys, os
_engine_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.join(_engine_dir, '..')
if _engine_dir not in sys.path: sys.path.insert(0, _engine_dir)
if _app_dir not in sys.path: sys.path.insert(0, _app_dir)

try:
    from engine.models import (
        SoilLayer, WaterTable, Surcharge, ProjectInput,
        PressureTheory, SurchargeType
    )
    from engine.coefficients import get_ka, get_kp, kp_caquot_kerisel
except ImportError:
    from models import (
        SoilLayer, WaterTable, Surcharge, ProjectInput,
        PressureTheory, SurchargeType
    )
    from coefficients import get_ka, get_kp, kp_caquot_kerisel


@dataclass
class PressurePoint:
    """Computed pressure values at a single depth."""
    depth: float                 # m below GL
    sigma_v_total: float         # kPa, total vertical stress
    u: float                     # kPa, pore water pressure
    sigma_v_eff: float           # kPa, effective vertical stress
    Ka: float                    # active coefficient at this depth
    Kp: float                    # passive coefficient at this depth
    sigma_ah_eff: float          # kPa, effective active horizontal stress
    sigma_ph_eff: float          # kPa, effective passive horizontal stress
    sigma_ah_total: float        # kPa, total active horizontal stress (eff + water)
    sigma_ph_total: float        # kPa, total passive horizontal stress (eff + water)
    q_lateral: float             # kPa, surcharge-induced lateral pressure
    layer_name: str              # which soil layer this point is in
    c_eff: float                 # kPa, cohesion of layer at this depth
    phi_eff: float               # degrees, friction angle at this depth


@dataclass
class PressureProfile:
    """Complete pressure profile along the wall height."""
    points: List[PressurePoint]
    excavation_depth: float
    embedment_depth: float
    water_table_depth: float
    theory: PressureTheory
    tension_crack_depth: float = 0.0  # m, depth of tension crack (active side)

    @property
    def depths(self) -> List[float]:
        return [p.depth for p in self.points]

    @property
    def active_pressures(self) -> List[float]:
        """Total active pressure (effective + water + surcharge)."""
        return [p.sigma_ah_total + p.q_lateral for p in self.points]

    @property
    def passive_pressures(self) -> List[float]:
        """Total passive pressure (effective + water)."""
        return [p.sigma_ph_total for p in self.points]

    @property
    def active_eff(self) -> List[float]:
        return [p.sigma_ah_eff for p in self.points]

    @property
    def passive_eff(self) -> List[float]:
        return [p.sigma_ph_eff for p in self.points]

    @property
    def water_pressures(self) -> List[float]:
        return [p.u for p in self.points]

    @property
    def surcharge_pressures(self) -> List[float]:
        return [p.q_lateral for p in self.points]

    def get_at_depth(self, z: float, tol: float = 0.001) -> Optional[PressurePoint]:
        """Get pressure point closest to specified depth."""
        for p in self.points:
            if abs(p.depth - z) < tol:
                return p
        # Find nearest
        closest = min(self.points, key=lambda p: abs(p.depth - z))
        if abs(closest.depth - z) < self.points[1].depth - self.points[0].depth:
            return closest
        return None

    def summary(self) -> str:
        """Print summary table."""
        lines = []
        lines.append(f"{'Depth':>6} │ {'σv_eff':>8} │ {'u':>8} │ {'Ka':>6} │ {'σah_eff':>8} │ "
                      f"{'Kp':>6} │ {'σph_eff':>8} │ {'q_lat':>7} │ {'Layer'}")
        lines.append(f"{'(m)':>6} │ {'(kPa)':>8} │ {'(kPa)':>8} │ {'':>6} │ {'(kPa)':>8} │ "
                      f"{'':>6} │ {'(kPa)':>8} │ {'(kPa)':>7} │ ")
        lines.append("─" * 95)

        # Print at key depths + every 1m
        key_depths = set()
        # Layer boundaries
        cum_depth = 0
        for p in self.points:
            if p.depth == 0 or abs(p.depth % 1.0) < 0.05 or abs(p.depth - self.excavation_depth) < 0.05:
                key_depths.add(round(p.depth, 3))

        for p in self.points:
            if round(p.depth, 3) in key_depths:
                lines.append(
                    f"{p.depth:6.2f} │ {p.sigma_v_eff:8.2f} │ {p.u:8.2f} │ {p.Ka:6.3f} │ "
                    f"{p.sigma_ah_eff:8.2f} │ {p.Kp:6.3f} │ {p.sigma_ph_eff:8.2f} │ "
                    f"{p.q_lateral:7.2f} │ {p.layer_name}"
                )

        return "\n".join(lines)


def _get_layer_at_depth(soil_layers: List[SoilLayer], z: float) -> Tuple[SoilLayer, float]:
    """
    Find which soil layer contains depth z.
    
    Returns:
        (layer, depth_within_layer)
    """
    cumulative = 0.0
    for layer in soil_layers:
        if z <= cumulative + layer.thickness + 1e-9:
            return layer, z - cumulative
        cumulative += layer.thickness

    # If z exceeds total soil depth, return last layer
    return soil_layers[-1], z - (cumulative - soil_layers[-1].thickness)


def _compute_vertical_stress(soil_layers: List[SoilLayer], wt: WaterTable,
                              z: float) -> Tuple[float, float, float]:
    """
    Compute total vertical stress, pore pressure, and effective vertical stress at depth z.
    
    Accounts for:
    - Multiple soil layers with different unit weights
    - Water table position (different γ above and below WT)
    
    Returns:
        (sigma_v_total, u, sigma_v_eff) in kPa
    """
    sigma_v_total = 0.0
    depth_processed = 0.0
    gamma_w = wt.gamma_w

    for layer in soil_layers:
        layer_top = depth_processed
        layer_bot = depth_processed + layer.thickness

        if z <= layer_top:
            break

        # Depth range within this layer that we need to integrate
        z_top = layer_top
        z_bot = min(z, layer_bot)
        dz_layer = z_bot - z_top

        if dz_layer <= 0:
            depth_processed = layer_bot
            continue

        # Split by water table
        wt_depth = wt.depth_behind_wall

        if wt_depth >= z_bot:
            # Entire segment above water table
            sigma_v_total += layer.gamma * dz_layer
        elif wt_depth <= z_top:
            # Entire segment below water table
            sigma_v_total += layer.gamma_sat * dz_layer
        else:
            # Water table crosses this segment
            dz_above = wt_depth - z_top
            dz_below = z_bot - wt_depth
            sigma_v_total += layer.gamma * dz_above + layer.gamma_sat * dz_below

        depth_processed = layer_bot

    # Pore water pressure
    if z <= wt.depth_behind_wall:
        u = 0.0
    else:
        u = gamma_w * (z - wt.depth_behind_wall)

    sigma_v_eff = sigma_v_total - u

    return sigma_v_total, u, sigma_v_eff


def _compute_surcharge_lateral(surcharges: List[Surcharge], z: float, Ka: float) -> float:
    """
    Compute lateral pressure at depth z due to surcharges.
    
    - Uniform surcharge: Δσ_h = Ka × q (constant with depth)
    - Line load: Boussinesq solution (2Q/π) × z×x² / (x²+z²)²  ... simplified
    - Strip load: Standard elasticity solution
    
    For deep excavation design, uniform surcharge is most common.
    
    Args:
        surcharges: List of surcharge objects
        z: Depth below ground level (m)
        Ka: Active earth pressure coefficient at this depth
    Returns:
        Total lateral pressure from all surcharges (kPa)
    """
    q_lateral = 0.0

    for s in surcharges:
        if s.surcharge_type == SurchargeType.UNIFORM:
            # Simple: Δσ_h = Ka × q
            q_lateral += Ka * s.magnitude

        elif s.surcharge_type == SurchargeType.LINE:
            # Boussinesq line load solution (per unit length of wall)
            # Δσ_h = (2Q/π) × z × x² / (x² + z²)²
            # where Q = line load (kN/m), x = offset from wall
            x = max(s.offset, 0.1)  # avoid singularity at x=0
            if z > 0:
                q_lateral += (2 * s.magnitude / math.pi) * (z * x ** 2) / (x ** 2 + z ** 2) ** 2
            # For z=0, contribution is 0

        elif s.surcharge_type == SurchargeType.STRIP:
            # Strip load at offset 'a' with width 'b'
            # Using standard elasticity: Δσ_h = (q/π)(α - sin(α)cos(α+2β))
            # Simplified for vertical wall: use influence chart approach
            # For now, treat as equivalent uniform over influenced zone
            a = s.offset          # near edge offset
            b = s.width           # strip width
            if z > 0:
                alpha1 = math.atan2(a, z)
                alpha2 = math.atan2(a + b, z)
                beta_angle = alpha2 - alpha1
                # Approximate: q_h = (q/π)(β - sin(β)cos(β + 2α1))
                q_h = (s.magnitude / math.pi) * (
                    beta_angle - math.sin(beta_angle) * math.cos(beta_angle + 2 * alpha1)
                )
                q_lateral += max(q_h, 0)

    return q_lateral


def compute_pressure_profile(project: ProjectInput) -> PressureProfile:
    """
    Main computation: generate full pressure profile along wall height.
    
    Computes active and passive pressures at every dz increment from
    GL to the toe of the wall (excavation depth + embedment).
    
    Active pressure: σ'ah = Ka × σ'v - 2c'√Ka  (Rankine)
    Passive pressure: σ'ph = Kp × σ'v + 2c'√Kp  (Rankine)
    
    Tension crack: If σ'ah < 0, it means the soil is in tension.
    For design, we set σ'ah = 0 in the tension zone.
    Tension crack depth: z_c = (2c')/(γ × √Ka)
    
    Returns:
        PressureProfile with computed values at each depth increment
    """
    project.validate()

    dz = project.dz
    total_depth = project.total_wall_height
    wt = project.water_table
    theory = project.pressure_theory

    points = []
    tension_crack_depth = 0.0
    found_positive_active = False

    # Generate depth array
    n_points = int(total_depth / dz) + 1
    depths = [i * dz for i in range(n_points)]
    # Ensure exact excavation depth is included
    if project.excavation_depth not in depths:
        depths.append(project.excavation_depth)
    depths = sorted(set([round(d, 6) for d in depths]))

    for z in depths:
        if z > total_depth + 1e-9:
            continue

        # Get layer properties at this depth
        layer, _ = _get_layer_at_depth(project.soil_layers, z)

        # Vertical stress
        sigma_v_total, u, sigma_v_eff = _compute_vertical_stress(
            project.soil_layers, wt, z
        )

        # Earth pressure coefficients
        Ka = get_ka(layer.phi_eff, theory, layer.delta)
        Kp = get_kp(layer.phi_eff, theory, layer.delta)

        # Effective horizontal pressures
        # Active: σ'ah = Ka × σ'v - 2c'√Ka
        c = layer.c_eff
        sigma_ah_eff = Ka * sigma_v_eff - 2.0 * c * math.sqrt(Ka)

        # Handle tension crack
        if sigma_ah_eff < 0 and not found_positive_active:
            sigma_ah_eff = 0.0  # Set to zero in tension zone
            tension_crack_depth = z
        else:
            if sigma_ah_eff > 0:
                found_positive_active = True

        # Passive: σ'ph = Kp × σ'v + 2c'√Kp
        sigma_ph_eff = Kp * sigma_v_eff + 2.0 * c * math.sqrt(Kp)

        # Total horizontal pressures (effective + water)
        sigma_ah_total = sigma_ah_eff + u
        sigma_ph_total = sigma_ph_eff + u

        # Surcharge lateral pressure
        q_lateral = _compute_surcharge_lateral(project.surcharges, z, Ka)

        point = PressurePoint(
            depth=z,
            sigma_v_total=sigma_v_total,
            u=u,
            sigma_v_eff=sigma_v_eff,
            Ka=Ka,
            Kp=Kp,
            sigma_ah_eff=sigma_ah_eff,
            sigma_ph_eff=sigma_ph_eff,
            sigma_ah_total=sigma_ah_total,
            sigma_ph_total=sigma_ph_total,
            q_lateral=q_lateral,
            layer_name=layer.name,
            c_eff=c,
            phi_eff=layer.phi_eff,
        )
        points.append(point)

    return PressureProfile(
        points=points,
        excavation_depth=project.excavation_depth,
        embedment_depth=project.embedment_depth,
        water_table_depth=wt.depth_behind_wall,
        theory=theory,
        tension_crack_depth=tension_crack_depth,
    )


def compute_net_pressure(profile: PressureProfile) -> List[Tuple[float, float]]:
    """
    Compute net pressure diagram for wall design.
    
    Above excavation level: only active pressure acts (toward wall)
    Below excavation level: active on retained side, passive on excavation side
    
    Net pressure = Active_total + surcharge - Passive_total (below exc. level)
    
    Convention: positive = net pressure pushing wall into excavation
    
    Returns:
        List of (depth, net_pressure) tuples
    """
    exc_depth = profile.excavation_depth
    net_pressures = []

    for p in profile.points:
        if p.depth <= exc_depth:
            # Above excavation: only active side
            net = p.sigma_ah_eff + p.u + p.q_lateral
        else:
            # Below excavation: active - passive
            # Active side: full active + water + surcharge from retained side
            active_total = p.sigma_ah_eff + p.u + p.q_lateral
            # Passive side: passive + water on excavation side
            # Water pressure on passive side depends on WT in excavation
            # For simplicity here, assume balanced water (net water = 0 below WT if same on both sides)
            # TODO: handle differential water head properly
            passive_total = p.sigma_ph_eff
            net = active_total - passive_total

        net_pressures.append((p.depth, net))

    return net_pressures
