"""
Module 5: Anchored Wall Analysis

For excavations in rock or where wall toe sits on hard stratum:
  - Sheet pile / soldier pile driven to rock surface
  - Lateral support from ground anchors (rock anchors / soil anchors)
  - Below wall toe: rock face self-supporting (drapery / shotcrete / rock bolts)

Structural model:
  - Continuous beam from wall top to toe
  - Discrete pin supports at each anchor level and at toe
  - Cantilever overhang above first anchor
  - Active pressure + water + surcharge as distributed load

Solution method:
  - Beam finite element (Euler-Bernoulli)
  - N elements along wall length
  - Pin support at each anchor/toe node (w=0)
  - Solve for displacements, reactions, BM, SF

Anchor design:
  - Force from wall analysis → tendon force (accounting for inclination)
  - Free length: beyond active failure wedge
  - Bond length: from allowable bond stress in rock/soil
  - Tendon sizing: bar or strand capacity

References:
  - BS 8081:2015 Ground Anchors
  - IS 9527 (Part 1):1981
  - FHWA-IF-99-015 Ground Anchors and Anchored Systems
  - Bowles 5th Ed., Chapter 11
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import sys, os
# Allow both standalone and package usage
_engine_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.join(_engine_dir, '..')
if _engine_dir not in sys.path: sys.path.insert(0, _engine_dir)
if _app_dir not in sys.path: sys.path.insert(0, _app_dir)

try:
    from engine.models import ProjectInput, SoilLayer, WaterTable
    from engine.earth_pressure import compute_pressure_profile
except ImportError:
    from models import ProjectInput, SoilLayer, WaterTable
    from earth_pressure import compute_pressure_profile


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Anchor:
    """
    Definition of a ground anchor.
    
    Three types supported:
    
    1. "prestressed" — Active anchor. Strand/bar tendon with free length + bond 
       zone. Prestressed to design load after grouting. Controls wall movement.
       Design per BS 8081 / EN 1537.
       
    2. "rebar" — Passive anchor (rock bolt / dowel). Plain rebar grouted full 
       length into drilled hole. No prestress. Force mobilized by wall movement.
       Cheap, fast, common in Mumbai CWR. Design per IS 456 (working stress for 
       rebar) and bond per empirical rock-grout values.
       
    3. "sda" — Self Drilling Anchor. Hollow bar acts as drill rod + anchor.
       Sacrificial drill bit. Grout through hollow core. No casing needed.
       Brands: DYWI Drill (DSI), Ischebeck TITAN, MAI, IBO.
       Design: manufacturer UTS + grout-ground bond.
    """
    # ── Common parameters ──
    level: float                    # m below GL (anchor head location on wall)
    anchor_type: str = "rebar"     # "prestressed", "rebar", or "sda"
    inclination: float = 20.0      # degrees below horizontal
    horizontal_spacing: float = 3.0  # m c/c along wall
    label: str = ""
    
    # Rock/soil bond parameters
    bond_stress: float = 200.0     # kPa, allowable grout-ground bond stress
    drill_diameter: float = 0.15   # m, anchor hole diameter
    
    # FOS
    fos_tendon: float = 1.67      # FOS on tendon/rebar capacity
    fos_bond: float = 2.5         # FOS on bond (BS 8081 for prestressed)
    
    # ── Prestressed-specific ──
    tendon_type: str = "strand"    # "strand" or "bar" (for prestressed only)
    prestress_ratio: float = 0.6   # lock-off load as fraction of tendon UTS
    
    # ── Rebar-specific ──
    rebar_dia: float = 25.0        # mm, rebar diameter
    rebar_fy: float = 500.0        # MPa, yield strength (Fe500)
    rebar_count: int = 1           # number of bars per anchor
    rebar_grade: str = "Fe500"     # for reporting
    
    # ── SDA-specific ──
    sda_size: str = "R32"          # R25, R32, R38, R51
    sda_uts: float = 0.0           # kN, will be set from size if 0
    sda_yield: float = 0.0         # kN, will be set from size if 0
    sda_od: float = 0.0            # mm, outer diameter
    
    def __post_init__(self):
        """Set SDA properties from size designation."""
        SDA_CATALOG = {
            # size: (OD_mm, UTS_kN, Yield_kN, drill_dia_mm)
            "R25": (25.0, 250, 200, 42),
            "R32": (32.0, 360, 280, 51),
            "R38": (38.0, 500, 400, 57),
            "R51": (51.0, 800, 630, 76),
        }
        
        if self.anchor_type == "sda" and self.sda_size in SDA_CATALOG:
            od, uts, yld, dd = SDA_CATALOG[self.sda_size]
            if self.sda_uts == 0:
                self.sda_uts = uts
            if self.sda_yield == 0:
                self.sda_yield = yld
            if self.sda_od == 0:
                self.sda_od = od
            # SDA drill diameter from catalog (sacrificial bit is larger than bar)
            self.drill_diameter = dd / 1000.0
        
        # Rebar: default FOS is lower (IS 456 working stress)
        if self.anchor_type == "rebar" and self.fos_tendon == 1.67:
            self.fos_tendon = 1.15  # IS 456 partial safety for steel in WS
            self.fos_bond = 2.0     # common for passive anchors


@dataclass
class AnchorDesignResult:
    """Result of anchor design for a single anchor."""
    label: str
    level: float                   # m below GL
    
    # Forces
    reaction_horizontal: float     # kN/m, from wall analysis
    reaction_per_anchor: float     # kN, = horizontal × spacing
    tendon_force: float            # kN, = reaction / cos(inclination)
    
    # Free length
    free_length: float             # m, beyond active wedge
    
    # Bond length
    bond_length_required: float    # m, from bond capacity
    
    # Total length
    total_length: float            # m, free + bond
    
    # Tendon
    tendon_type: str
    tendon_area_required: float    # mm²
    tendon_recommendation: str     # e.g., "2 × 15.2mm strands"
    
    # Capacity check
    fos_bond_actual: float
    fos_tendon_actual: float
    status: str
    
    notes: List[str] = field(default_factory=list)


@dataclass
class AnchoredWallResult:
    """Complete result of anchored wall analysis."""
    # Wall geometry
    wall_height: float             # m (from top to toe)
    toe_level: float               # m below GL
    
    # Internal forces
    depths: np.ndarray             # m
    bending_moments: np.ndarray    # kN·m/m
    shear_forces: np.ndarray       # kN/m
    deflections: np.ndarray        # mm
    
    # Design values
    max_moment: float              # kN·m/m
    max_moment_depth: float        # m
    max_shear: float               # kN/m
    max_deflection: float          # mm
    
    # Support reactions
    anchor_reactions: List[float]  # kN/m, horizontal reaction at each anchor
    toe_reaction: float            # kN/m, horizontal reaction at toe
    
    # Anchor designs
    anchor_designs: List[AnchorDesignResult]
    
    # Applied loading
    total_active_force: float      # kN/m
    
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Beam FE Solver
# ─────────────────────────────────────────────

def _beam_element_stiffness(L, EI):
    """
    4×4 stiffness matrix for Euler-Bernoulli beam element.
    DOFs: [w1, θ1, w2, θ2]
    """
    k = EI / L**3
    return k * np.array([
        [ 12,    6*L,   -12,    6*L],
        [ 6*L,   4*L**2, -6*L,  2*L**2],
        [-12,   -6*L,    12,   -6*L],
        [ 6*L,   2*L**2, -6*L,  4*L**2],
    ])


def _equivalent_nodal_loads_udl(q, L):
    """
    Equivalent nodal loads for uniform distributed load q (kN/m per m run)
    on a beam element of length L.
    Returns [F1, M1, F2, M2] (consistent loading).
    """
    return np.array([
        q * L / 2,
        q * L**2 / 12,
        q * L / 2,
        -q * L**2 / 12,
    ])


def _equivalent_nodal_loads_linear(q1, q2, L):
    """
    Equivalent nodal loads for linearly varying load from q1 to q2.
    q1 at node 1, q2 at node 2.
    """
    return np.array([
        L * (7*q1 + 3*q2) / 20,
        L**2 * (3*q1 + 2*q2) / 60,
        L * (3*q1 + 7*q2) / 20,
        -L**2 * (2*q1 + 3*q2) / 60,
    ])


def solve_anchored_wall_beam(
    wall_height: float,
    pressure_at_depths: List[Tuple[float, float]],  # [(depth, total_pressure_kPa), ...]
    anchor_levels: List[float],   # depths of anchor heads
    toe_level: float,             # depth of wall toe
    EI: float = 50000.0,          # kN·m² per m run (default ~ LARSSEN 4 sheet pile)
    n_elements: int = 50,
    point_loads: List[Tuple[float, float, float]] = None,  # [(depth, force_kN/m, moment_kNm/m), ...]
    anchor_stiffnesses: List[Tuple[float, float]] = None,  # [(level, k_kN/m_per_m), ...]
    toe_stiffness: float = None,  # kN/m per m run (None = rigid)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], float]:
    """
    Solve anchored wall as beam on discrete supports.
    
    Args:
        wall_height: total wall length from top (0) to toe
        pressure_at_depths: active pressure distribution (depth, pressure) pairs
        anchor_levels: depths where anchors provide lateral support
        toe_level: depth of wall toe (last support)
        EI: flexural rigidity of wall per m run
        n_elements: number of beam elements
        point_loads: optional list of (depth, horizontal_force, moment) applied at
                     specific depths. Used for wind barrier loads, crane loads, etc.
        
    Returns:
        depths, moments, shears, deflections, anchor_reactions, toe_reaction
    """
    n_nodes = n_elements + 1
    L_elem = wall_height / n_elements
    n_dof = 2 * n_nodes  # 2 DOF per node: w, θ
    
    # ── Global stiffness matrix ──
    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)
    
    for i in range(n_elements):
        # Element stiffness
        k_e = _beam_element_stiffness(L_elem, EI)
        
        # Assembly
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for a in range(4):
            F_idx = dofs[a]
            for b in range(4):
                K_global[dofs[a], dofs[b]] += k_e[a, b]
    
    # ── Load vector from pressure distribution ──
    # Interpolate pressure at each node
    depths_array = np.linspace(0, wall_height, n_nodes)
    
    # Build pressure at each node by interpolation
    p_depths = np.array([p[0] for p in pressure_at_depths])
    p_values = np.array([p[1] for p in pressure_at_depths])
    pressures = np.interp(depths_array, p_depths, p_values)
    
    # Apply as linearly varying load on each element
    for i in range(n_elements):
        q1 = pressures[i]
        q2 = pressures[i + 1]
        f_e = _equivalent_nodal_loads_linear(q1, q2, L_elem)
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for a in range(4):
            F_global[dofs[a]] += f_e[a]
    
    # ── Point loads (wind barrier, crane, etc.) ──
    if point_loads:
        for (p_depth, p_force, p_moment) in point_loads:
            node_idx = int(round(p_depth / L_elem))
            node_idx = max(0, min(node_idx, n_nodes - 1))
            # DOF 2*node = transverse force, DOF 2*node+1 = moment
            F_global[2 * node_idx] += p_force
            F_global[2 * node_idx + 1] += p_moment
    
    # ── Boundary conditions ──
    # Spring supports at anchors (stiffness-based, not rigid pins)
    # Toe support: rigid or spring depending on connection
    
    # Default anchor stiffness if not provided:
    # Passive rebar in CWR: ~5-20 kN/mm per anchor
    # Prestressed in good rock: ~20-50 kN/mm per anchor
    # Convert to kN/m per m run: k_spring = k_anchor / (spacing × 1000) 
    # with horizontal component: × cos(inclination)
    
    support_levels = sorted(anchor_levels + [toe_level])
    support_nodes = []
    
    for s_level in support_levels:
        node_idx = int(round(s_level / L_elem))
        node_idx = max(0, min(node_idx, n_nodes - 1))
        support_nodes.append(node_idx)
    
    # Add spring stiffness at support nodes
    # anchor_stiffnesses: list of (level, k in kN/m per m run)
    # If None, use rigid supports (backward compatible)
    if anchor_stiffnesses is not None:
        # Spring supports — add stiffness to diagonal
        for (a_level, k_spring) in anchor_stiffnesses:
            node_idx = int(round(a_level / L_elem))
            node_idx = max(0, min(node_idx, n_nodes - 1))
            # k_spring is in kN/m per m run → directly add to K[w_dof, w_dof]
            K_global[2 * node_idx, 2 * node_idx] += k_spring
        
        # Toe: grouted bench — relatively stiff
        toe_node_idx = int(round(toe_level / L_elem))
        toe_node_idx = max(0, min(toe_node_idx, n_nodes - 1))
        if toe_stiffness is not None:
            K_global[2 * toe_node_idx, 2 * toe_node_idx] += toe_stiffness
        else:
            # Default: rigid toe (grouted bench on rock)
            K_global[2 * toe_node_idx, 2 * toe_node_idx] += 1e8
        
        # No constrained DOFs — all springs
        constrained_dofs = []
    else:
        # Original rigid pin behavior
        constrained_dofs = [2 * n for n in support_nodes]
    
    # ── Solve ──
    free_dofs = [d for d in range(n_dof) if d not in constrained_dofs]
    
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F_global[free_dofs]
    
    # Check conditioning
    if np.linalg.cond(K_ff) > 1e15:
        for cd in constrained_dofs:
            K_global[cd, cd] += 1e10
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    
    u_f = np.linalg.solve(K_ff, F_f)
    
    # Full displacement vector
    u_global = np.zeros(n_dof)
    for i, d in enumerate(free_dofs):
        u_global[d] = u_f[i]
    
    # ── Extract reactions ──
    # For spring supports, reaction = k × displacement
    anchor_reactions = []
    if anchor_stiffnesses is not None:
        for (a_level, k_spring) in anchor_stiffnesses:
            node_idx = int(round(a_level / L_elem))
            node_idx = max(0, min(node_idx, n_nodes - 1))
            reaction = k_spring * u_global[2 * node_idx]
            anchor_reactions.append(reaction)
        
        toe_node = int(round(toe_level / L_elem))
        toe_node = max(0, min(toe_node, n_nodes - 1))
        k_toe = toe_stiffness if toe_stiffness else 1e8
        toe_reaction = k_toe * u_global[2 * toe_node]
    else:
        R_global = K_global @ u_global - F_global
        for s_level in anchor_levels:
            node_idx = int(round(s_level / L_elem))
            node_idx = max(0, min(node_idx, n_nodes - 1))
            anchor_reactions.append(R_global[2 * node_idx])
        
        toe_node = int(round(toe_level / L_elem))
        toe_node = max(0, min(toe_node, n_nodes - 1))
        toe_reaction = R_global[2 * toe_node]
    
    # ── Internal forces ──
    moments = np.zeros(n_nodes)
    shears = np.zeros(n_nodes)
    
    for i in range(n_elements):
        # Element displacements
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        u_e = u_global[dofs]
        
        k_e = _beam_element_stiffness(L_elem, EI)
        
        # Element forces (internal)
        q1 = pressures[i]
        q2 = pressures[i + 1]
        f_fixed = _equivalent_nodal_loads_linear(q1, q2, L_elem)
        
        f_int = k_e @ u_e - f_fixed
        
        # Convention: positive shear = upward on left face
        # f_int = [V1, M1, V2, M2] in element local coords
        # At node i: shear = -V1, moment = -M1
        # At node i+1: shear = V2, moment = M2
        
        if i == 0:
            shears[0] = f_int[0]
            moments[0] = f_int[1]
        
        shears[i + 1] = -f_int[2]
        moments[i + 1] = f_int[3]
    
    # Deflections (in mm)
    deflections = np.array([u_global[2 * i] * 1000 for i in range(n_nodes)])
    
    return depths_array, moments, shears, deflections, anchor_reactions, toe_reaction


# ─────────────────────────────────────────────
# Active Pressure on Wall (0 to toe only)
# ─────────────────────────────────────────────

def get_wall_pressure_distribution(
    project: ProjectInput,
    toe_level: float,
) -> List[Tuple[float, float]]:
    """
    Get active pressure distribution from GL to wall toe.
    Returns list of (depth, total_horizontal_pressure) in kPa.
    
    Total pressure = effective active pressure + water pressure.
    """
    profile = compute_pressure_profile(project)
    
    result = []
    for pt in profile.points:
        if pt.depth > toe_level + 0.05:
            break
        # Total active pressure = σ'ah + u
        total_p = pt.sigma_ah_eff + pt.u
        # Add surcharge component (already included in sigma_ah_eff via σv)
        result.append((pt.depth, max(total_p, 0)))
    
    # Ensure we have the toe point
    if result and result[-1][0] < toe_level - 0.05:
        # Interpolate to exact toe level
        last_d, last_p = result[-1]
        if len(result) >= 2:
            prev_d, prev_p = result[-2]
            slope = (last_p - prev_p) / (last_d - prev_d) if last_d != prev_d else 0
            toe_p = last_p + slope * (toe_level - last_d)
            result.append((toe_level, max(toe_p, 0)))
    
    return result


# ─────────────────────────────────────────────
# Anchor Design
# ─────────────────────────────────────────────

def design_anchor(
    anchor: Anchor,
    reaction_per_m: float,     # kN/m, horizontal reaction from wall analysis
    excavation_depth: float,
    soil_layers: List[SoilLayer],
    wall_toe_level: float,
) -> AnchorDesignResult:
    """
    Design a single ground anchor — dispatches to type-specific design.
    
    Supports: "prestressed", "rebar", "sda"
    """
    notes = []
    
    # ── Common: Forces ──
    reaction_per_anchor = reaction_per_m * anchor.horizontal_spacing
    alpha_rad = math.radians(anchor.inclination)
    tendon_force = reaction_per_anchor / math.cos(alpha_rad)
    
    notes.append(f"Type: {anchor.anchor_type.upper()}")
    notes.append(f"Horizontal reaction: {reaction_per_m:.1f} kN/m × "
                 f"{anchor.horizontal_spacing:.1f}m = {reaction_per_anchor:.1f} kN/anchor")
    notes.append(f"Bar/tendon force (at {anchor.inclination}°): {tendon_force:.1f} kN")
    
    # ── Common: Active wedge geometry ──
    cum_depth = 0
    phi_at_anchor = 30.0
    for layer in soil_layers:
        layer_top = cum_depth
        layer_bot = cum_depth + layer.thickness
        cum_depth = layer_bot
        if layer_top <= anchor.level < layer_bot:
            phi_at_anchor = layer.phi_eff
            break
    
    wedge_angle = math.radians(45 + phi_at_anchor / 2)
    depth_below_exc = excavation_depth - anchor.level
    horizontal_to_wedge = depth_below_exc / math.tan(wedge_angle) if depth_below_exc > 0 else 0
    
    # ── Dispatch to type-specific design ──
    if anchor.anchor_type == "prestressed":
        return _design_prestressed(anchor, reaction_per_m, reaction_per_anchor,
                                   tendon_force, alpha_rad, horizontal_to_wedge, notes)
    elif anchor.anchor_type == "rebar":
        return _design_rebar(anchor, reaction_per_m, reaction_per_anchor,
                             tendon_force, alpha_rad, notes)
    elif anchor.anchor_type == "sda":
        return _design_sda(anchor, reaction_per_m, reaction_per_anchor,
                           tendon_force, alpha_rad, notes)
    else:
        raise ValueError(f"Unknown anchor type: {anchor.anchor_type}. "
                         f"Use 'prestressed', 'rebar', or 'sda'.")


# ─────────────────────────────────────────────
# Prestressed (Active) Anchor Design
# ─────────────────────────────────────────────

def _design_prestressed(anchor, reaction_per_m, reaction_per_anchor,
                        tendon_force, alpha_rad, horizontal_to_wedge, notes):
    """
    Prestressed (active) anchor: strand or bar tendon.
    Free length beyond active wedge + grouted bond zone.
    Locked off at design load. Per BS 8081 / EN 1537.
    """
    # ── Free length ──
    free_length_horizontal = horizontal_to_wedge + 1.5  # 1.5m beyond wedge
    free_length = free_length_horizontal / math.cos(alpha_rad)
    free_length = max(free_length, 3.0)  # min 3m per BS 8081
    
    notes.append(f"Active wedge: {horizontal_to_wedge:.1f}m from wall")
    notes.append(f"Free length: {free_length:.1f}m (min 3.0m per BS 8081)")
    
    # ── Bond length ──
    bond_length = (tendon_force * anchor.fos_bond) / (
        math.pi * anchor.drill_diameter * anchor.bond_stress)
    bond_length = max(bond_length, 3.0)
    bond_length = math.ceil(bond_length * 2) / 2  # round to 0.5m
    
    bond_capacity = math.pi * anchor.drill_diameter * anchor.bond_stress * bond_length
    fos_bond = bond_capacity / tendon_force if tendon_force > 0 else 999
    
    notes.append(f"Bond: {anchor.bond_stress:.0f} kPa, Ø{anchor.drill_diameter*1000:.0f}mm hole")
    notes.append(f"Bond length: {bond_length:.1f}m (capacity: {bond_capacity:.1f} kN, FOS: {fos_bond:.2f})")
    
    # ── Tendon sizing ──
    if anchor.tendon_type == "strand":
        strand_uts = 260.7  # kN, 15.2mm 7-wire Grade 1860
        strand_area = 140   # mm²
        n_strands = max(1, math.ceil(tendon_force / (strand_uts / anchor.fos_tendon)))
        tendon_area = n_strands * strand_area
        tendon_capacity = n_strands * strand_uts
        fos_tendon = tendon_capacity / tendon_force if tendon_force > 0 else 999
        lock_off = tendon_force
        recommendation = (f"{n_strands} × 15.2mm strand (Gr.1860), "
                          f"lock-off: {lock_off:.0f} kN")
    else:
        bar_options = [(25,245,491),(32,402,804),(36,509,1018),(40,628,1257),(50,981,1963)]
        selected = bar_options[-1]
        for dia, uts, area in bar_options:
            if uts / anchor.fos_tendon >= tendon_force:
                selected = (dia, uts, area)
                break
        dia, uts, area = selected
        tendon_area = area
        tendon_capacity = uts
        fos_tendon = uts / tendon_force if tendon_force > 0 else 999
        recommendation = f"Ø{dia}mm prestressing bar (Gr.1030, UTS={uts}kN)"
    
    notes.append(f"Tendon: {recommendation}")
    total_length = free_length + bond_length
    notes.append(f"Total: {free_length:.1f}m free + {bond_length:.1f}m bond = {total_length:.1f}m")
    
    status = "OK"
    if fos_bond < anchor.fos_bond: status = "FAIL — bond"
    if fos_tendon < anchor.fos_tendon: status = "FAIL — tendon"
    
    return AnchorDesignResult(
        label=anchor.label or f"Prestressed anchor at {anchor.level:.1f}m",
        level=anchor.level,
        reaction_horizontal=reaction_per_m, reaction_per_anchor=reaction_per_anchor,
        tendon_force=tendon_force, free_length=free_length,
        bond_length_required=bond_length, total_length=total_length,
        tendon_type=f"prestressed_{anchor.tendon_type}",
        tendon_area_required=tendon_area,
        tendon_recommendation=recommendation,
        fos_bond_actual=fos_bond, fos_tendon_actual=fos_tendon,
        status=status, notes=notes,
    )


# ─────────────────────────────────────────────
# Rebar (Passive) Anchor Design
# ─────────────────────────────────────────────

def _design_rebar(anchor, reaction_per_m, reaction_per_anchor,
                  tendon_force, alpha_rad, notes):
    """
    Passive rebar anchor (rock bolt / dowel).
    
    No free length — full length is grouted (bonded).
    No prestress — force mobilized by wall displacement.
    
    Capacity checks:
      (a) Rebar yield: n × As × fy / FOS
      (b) Grout-rock bond: π × D_hole × τ_bond × L_bond / FOS
      (c) Grout-rebar bond: n × π × d_bar × τ_grout × L_bond / FOS
          τ_grout ≈ 1.0-2.5 MPa for cement grout (IS 456 cl 26.2)
    """
    d_bar = anchor.rebar_dia    # mm
    n_bars = anchor.rebar_count
    fy = anchor.rebar_fy        # MPa
    
    # ── (a) Rebar capacity — auto-size if needed ──
    area_per_bar = math.pi / 4 * d_bar**2
    total_area = n_bars * area_per_bar
    rebar_yield_kN = total_area * fy / 1000  # kN
    rebar_working = rebar_yield_kN / anchor.fos_tendon
    
    if rebar_working < tendon_force:
        # Auto-select: try standard diameters
        required_area = tendon_force * anchor.fos_tendon * 1000 / fy
        for try_dia in [16, 20, 25, 28, 32, 36, 40]:
            try_area = math.pi / 4 * try_dia**2
            n_needed = math.ceil(required_area / try_area)
            if n_needed <= 4:  # practical limit per hole
                d_bar = try_dia
                n_bars = n_needed
                total_area = n_bars * math.pi / 4 * d_bar**2
                rebar_yield_kN = total_area * fy / 1000
                rebar_working = rebar_yield_kN / anchor.fos_tendon
                break
    
    fos_tendon = rebar_yield_kN / tendon_force if tendon_force > 0 else 999
    notes.append(f"Rebar: {n_bars}×Ø{d_bar:.0f}mm {anchor.rebar_grade} "
                 f"(As={total_area:.0f} mm², yield={rebar_yield_kN:.1f} kN, "
                 f"FOS={fos_tendon:.2f})")
    
    # ── (b) Bond length (grout-rock) ──
    # Passive anchor: no free length, all bonded
    bond_length = (tendon_force * anchor.fos_bond) / (
        math.pi * anchor.drill_diameter * anchor.bond_stress)
    bond_length = max(bond_length, 1.5)  # practical min
    bond_length = math.ceil(bond_length * 2) / 2  # round to 0.5m
    
    bond_capacity = math.pi * anchor.drill_diameter * anchor.bond_stress * bond_length
    fos_bond = bond_capacity / tendon_force if tendon_force > 0 else 999
    
    notes.append(f"Grout-rock bond: {anchor.bond_stress:.0f} kPa, "
                 f"Ø{anchor.drill_diameter*1000:.0f}mm hole")
    notes.append(f"Bond length: {bond_length:.1f}m "
                 f"(capacity: {bond_capacity:.1f} kN, FOS: {fos_bond:.2f})")
    
    # ── (c) Grout-rebar bond check ──
    tau_grout = 1500  # kPa (1.5 MPa, conservative for cement grout)
    grout_rebar_cap = n_bars * math.pi * (d_bar / 1000) * tau_grout * bond_length
    fos_grout_rebar = grout_rebar_cap / tendon_force if tendon_force > 0 else 999
    notes.append(f"Grout-rebar bond: τ={tau_grout/1000:.1f} MPa, "
                 f"capacity={grout_rebar_cap:.1f} kN (FOS: {fos_grout_rebar:.2f})")
    
    # Total length = bond + 0.3m projection (nut + plate)
    total_length = bond_length + 0.3
    recommendation = (f"{n_bars}×Ø{d_bar:.0f}mm {anchor.rebar_grade}, "
                      f"L={bond_length:.1f}m bonded + 0.3m projection")
    notes.append(f"Recommendation: {recommendation}")
    
    status = "OK"
    if fos_bond < anchor.fos_bond: status = "FAIL — grout-rock bond"
    if fos_tendon < anchor.fos_tendon: status = "FAIL — rebar capacity"
    if fos_grout_rebar < 1.5: status = "FAIL — grout-rebar bond"
    
    return AnchorDesignResult(
        label=anchor.label or f"Rebar anchor at {anchor.level:.1f}m",
        level=anchor.level,
        reaction_horizontal=reaction_per_m, reaction_per_anchor=reaction_per_anchor,
        tendon_force=tendon_force, free_length=0.0,
        bond_length_required=bond_length, total_length=total_length,
        tendon_type=f"rebar_{n_bars}x{d_bar:.0f}mm",
        tendon_area_required=total_area,
        tendon_recommendation=recommendation,
        fos_bond_actual=fos_bond, fos_tendon_actual=fos_tendon,
        status=status, notes=notes,
    )


# ─────────────────────────────────────────────
# SDA (Self Drilling Anchor) Design
# ─────────────────────────────────────────────

def _design_sda(anchor, reaction_per_m, reaction_per_anchor,
                tendon_force, alpha_rad, notes):
    """
    Self Drilling Anchor (SDA).
    
    Hollow bar = drill rod + anchor. Sacrificial drill bit.
    Grout pumped through hollow center after drilling.
    No casing needed. Can also be used without grout but
    grouting improves bond and corrosion protection.
    
    Brands: DYWI Drill (DSI), Ischebeck TITAN, MAI, IBO
    
    Capacity governed by:
      (a) Bar capacity: yield load / FOS (from manufacturer data)
      (b) Grout-ground bond: π × D_bit × τ_bond × L / FOS
    
    Catalog:
      R25: OD 25mm, UTS 250kN, Yield 200kN, bit Ø42mm
      R32: OD 32mm, UTS 360kN, Yield 280kN, bit Ø51mm
      R38: OD 38mm, UTS 500kN, Yield 400kN, bit Ø57mm
      R51: OD 51mm, UTS 800kN, Yield 630kN, bit Ø76mm
    """
    SDA_CATALOG = {
        "R25": (25.0, 250, 200, 42),
        "R32": (32.0, 360, 280, 51),
        "R38": (38.0, 500, 400, 57),
        "R51": (51.0, 800, 630, 76),
    }
    
    # ── (a) Bar capacity — auto-upgrade if needed ──
    sda_size = anchor.sda_size
    sda_yield = anchor.sda_yield
    sda_uts = anchor.sda_uts
    drill_dia = anchor.drill_diameter
    
    bar_working = sda_yield / anchor.fos_tendon
    
    if bar_working < tendon_force:
        for name, (od, uts, yld, dd) in SDA_CATALOG.items():
            if yld / anchor.fos_tendon >= tendon_force:
                sda_size = name
                sda_uts = uts
                sda_yield = yld
                drill_dia = dd / 1000
                bar_working = yld / anchor.fos_tendon
                notes.append(f"Auto-upgraded to {name}")
                break
    
    fos_tendon = sda_yield / tendon_force if tendon_force > 0 else 999
    notes.append(f"SDA: {sda_size} (yield={sda_yield:.0f}kN, "
                 f"UTS={sda_uts:.0f}kN, FOS={fos_tendon:.2f})")
    
    # ── (b) Bond length ──
    # Effective diameter = sacrificial bit diameter (larger than bar OD)
    bond_length = (tendon_force * anchor.fos_bond) / (
        math.pi * drill_dia * anchor.bond_stress)
    bond_length = max(bond_length, 2.0)  # practical min
    bond_length = math.ceil(bond_length * 2) / 2
    
    bond_capacity = math.pi * drill_dia * anchor.bond_stress * bond_length
    fos_bond = bond_capacity / tendon_force if tendon_force > 0 else 999
    
    notes.append(f"Drill bit Ø: {drill_dia*1000:.0f}mm (sacrificial)")
    notes.append(f"Bond: {bond_length:.1f}m "
                 f"(capacity: {bond_capacity:.1f} kN, FOS: {fos_bond:.2f})")
    notes.append(f"Grout: through hollow bar center (recommended even if optional)")
    
    total_length = bond_length + 0.3
    recommendation = f"{sda_size} SDA, L={bond_length:.1f}m + coupler/plate"
    notes.append(f"Recommendation: {recommendation}")
    
    status = "OK"
    if fos_bond < anchor.fos_bond: status = "FAIL — bond"
    if fos_tendon < anchor.fos_tendon: status = "FAIL — bar capacity"
    
    return AnchorDesignResult(
        label=anchor.label or f"SDA at {anchor.level:.1f}m",
        level=anchor.level,
        reaction_horizontal=reaction_per_m, reaction_per_anchor=reaction_per_anchor,
        tendon_force=tendon_force, free_length=0.0,
        bond_length_required=bond_length, total_length=total_length,
        tendon_type=f"SDA_{sda_size}",
        tendon_area_required=math.pi/4 * float(sda_size[1:])**2,
        tendon_recommendation=recommendation,
        fos_bond_actual=fos_bond, fos_tendon_actual=fos_tendon,
        status=status, notes=notes,
    )


# ─────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────

def analyze_anchored_wall(
    project: ProjectInput,
    anchors: List[Anchor],
    wall_toe_level: float,
    EI: float = 50000.0,
    n_elements: int = 100,
    point_loads: List[Tuple[float, float, float]] = None,
    use_spring_supports: bool = False,
    anchor_stiffness_per_anchor: float = None,  # kN/mm per anchor (REQUIRED when use_spring_supports=True)
    toe_stiffness_kN_mm: float = None,  # kN/mm at toe (REQUIRED when use_spring_supports=True)
) -> AnchoredWallResult:
    """
    Analyze an anchored wall system.
    
    Two support models:
      use_spring_supports=False → Rigid pins (conservative forces, no deflection info)
      use_spring_supports=True  → Spring supports (realistic deflections + forces)
    
    When using springs, the engineer MUST provide:
      - anchor_stiffness_per_anchor (kN/mm per anchor head)
      - toe_stiffness_kN_mm (kN/mm at wall toe)
    
    These values should come from pullout tests, monitoring data, or
    engineering judgement. The software does not estimate them.
    
    Args:
        project: ProjectInput with soil profile and excavation info
        anchors: List of Anchor objects
        wall_toe_level: Depth of wall toe (m below GL)
        EI: Flexural rigidity (kN·m²/m)
        n_elements: Number of beam elements
        point_loads: Optional [(depth, force_kN/m, moment_kNm/m), ...]
        use_spring_supports: If True, model anchors as springs
        anchor_stiffness_per_anchor: kN/mm per anchor head (engineer input)
        toe_stiffness_kN_mm: kN/mm at wall toe (engineer input)
        
    Returns:
        AnchoredWallResult
    """
    notes = []
    wall_height = wall_toe_level
    
    # ── Get pressure distribution on wall ──
    pressure_dist = get_wall_pressure_distribution(project, wall_toe_level)
    
    if not pressure_dist:
        raise ValueError("No pressure distribution computed. Check soil profile.")
    
    # Total active force
    total_force = 0
    for i in range(len(pressure_dist) - 1):
        d1, p1 = pressure_dist[i]
        d2, p2 = pressure_dist[i + 1]
        total_force += 0.5 * (p1 + p2) * (d2 - d1)
    
    total_point_force = 0
    if point_loads:
        for (pd, pf, pm) in point_loads:
            total_point_force += pf
        notes.append(f"Point loads: {len(point_loads)} applied "
                     f"(total horiz: {total_point_force:.1f} kN/m)")
    
    notes.append(f"Wall height: {wall_height:.1f}m (GL to toe)")
    notes.append(f"Total active force on wall: {total_force:.1f} kN/m")
    notes.append(f"Total applied force (earth + point): {total_force + total_point_force:.1f} kN/m")
    notes.append(f"Anchors: {len(anchors)} levels")
    notes.append(f"EI: {EI:.0f} kN·m²/m")
    
    # ── Anchor spring stiffnesses ──
    anchor_stiffness_list = None
    toe_stiff_val = None
    
    if use_spring_supports:
        if anchor_stiffness_per_anchor is None or toe_stiffness_kN_mm is None:
            raise ValueError(
                "When use_spring_supports=True, both anchor_stiffness_per_anchor "
                "(kN/mm) and toe_stiffness_kN_mm (kN/mm) must be provided by the "
                "engineer. Determine from pullout tests or monitoring data.")
        
        anchor_stiffness_list = []
        k_per_anchor = anchor_stiffness_per_anchor
        
        for anchor in anchors:
            alpha_rad = math.radians(anchor.inclination)
            k_per_m = k_per_anchor * math.cos(alpha_rad)**2 / anchor.horizontal_spacing
            k_per_m_SI = k_per_m * 1000  # kN/m per m run
            
            anchor_stiffness_list.append((anchor.level, k_per_m_SI))
            notes.append(f"  {anchor.label}: k={k_per_anchor:.1f} kN/mm/anchor "
                         f"→ {k_per_m_SI:.0f} kN/m per m run")
        
        toe_stiff_val = toe_stiffness_kN_mm * 1000  # kN/m
        notes.append(f"  Toe: k={toe_stiffness_kN_mm:.1f} kN/mm")
        notes.append(f"Support model: SPRING (engineer-specified stiffness)")
    else:
        notes.append(f"Support model: RIGID PIN (conservative forces)")
    
    # ── Solve beam FE ──
    anchor_levels = [a.level for a in anchors]
    
    depths, moments, shears, deflections, anchor_reactions, toe_reaction = \
        solve_anchored_wall_beam(
            wall_height=wall_height,
            pressure_at_depths=pressure_dist,
            anchor_levels=anchor_levels,
            toe_level=wall_toe_level,
            EI=EI,
            n_elements=n_elements,
            point_loads=point_loads,
            anchor_stiffnesses=anchor_stiffness_list,
            toe_stiffness=toe_stiff_val,
        )
    
    # ── Find max values ──
    max_M_idx = np.argmax(np.abs(moments))
    max_M = moments[max_M_idx]
    max_M_depth = depths[max_M_idx]
    
    max_V = np.max(np.abs(shears))
    max_defl = np.max(np.abs(deflections))
    
    notes.append(f"Max BM: {max_M:.2f} kN·m/m at {max_M_depth:.2f}m")
    notes.append(f"Max SF: {max_V:.2f} kN/m")
    notes.append(f"Max deflection: {max_defl:.2f} mm")
    
    # Equilibrium check
    total_applied = total_force + total_point_force
    sum_reactions = sum(anchor_reactions) + toe_reaction
    equil_error = abs(abs(sum_reactions) - total_applied) / total_applied * 100 if total_applied > 0 else 0
    notes.append(f"Equilibrium check: ΣR={abs(sum_reactions):.1f}, P_total={total_applied:.1f} "
                 f"(error: {equil_error:.2f}%)")
    
    # ── Design anchors ──
    anchor_designs = []
    for i, anchor in enumerate(anchors):
        reaction = anchor_reactions[i]  # kN/m (negative = into wall)
        # Reaction is the force the support exerts ON the beam = compression into wall
        # Anchor must resist this = pull into rock
        reaction_magnitude = abs(reaction)
        
        design = design_anchor(
            anchor=anchor,
            reaction_per_m=reaction_magnitude,
            excavation_depth=project.excavation_depth,
            soil_layers=project.soil_layers,
            wall_toe_level=wall_toe_level,
        )
        anchor_designs.append(design)
    
    return AnchoredWallResult(
        wall_height=wall_height,
        toe_level=wall_toe_level,
        depths=depths,
        bending_moments=moments,
        shear_forces=shears,
        deflections=deflections,
        max_moment=abs(max_M),
        max_moment_depth=max_M_depth,
        max_shear=max_V,
        max_deflection=max_defl,
        anchor_reactions=[abs(r) for r in anchor_reactions],
        toe_reaction=abs(toe_reaction),
        anchor_designs=anchor_designs,
        total_active_force=total_force,
        notes=notes,
    )


# ─────────────────────────────────────────────
# Summary Output
# ─────────────────────────────────────────────

def print_anchored_wall_summary(result: AnchoredWallResult):
    """Print formatted anchored wall analysis results."""
    print("=" * 70)
    print("ANCHORED WALL ANALYSIS RESULTS")
    print("=" * 70)
    
    for note in result.notes:
        print(f"  {note}")
    
    # ── Reactions ──
    print(f"\n  {'─' * 60}")
    print(f"  SUPPORT REACTIONS")
    print(f"  {'─' * 60}")
    print(f"  {'Support':25} {'Level (m)':>10} {'Reaction (kN/m)':>16}")
    print(f"  {'─' * 55}")
    
    for i, (design, reaction) in enumerate(zip(result.anchor_designs, result.anchor_reactions)):
        print(f"  {design.label:25} {design.level:>10.1f} {reaction:>16.1f}")
    
    print(f"  {'Toe':25} {result.toe_level:>10.1f} {result.toe_reaction:>16.1f}")
    print(f"  {'─' * 55}")
    total_r = sum(result.anchor_reactions) + result.toe_reaction
    print(f"  {'TOTAL':25} {'':>10} {total_r:>16.1f}")
    print(f"  {'Active force':25} {'':>10} {result.total_active_force:>16.1f}")
    
    # ── Wall design values ──
    print(f"\n  {'─' * 60}")
    print(f"  WALL INTERNAL FORCES")
    print(f"  {'─' * 60}")
    print(f"  Max bending moment: {result.max_moment:.2f} kN·m/m at {result.max_moment_depth:.2f}m")
    print(f"  Max shear force:    {result.max_shear:.2f} kN/m")
    print(f"  Max deflection:     {result.max_deflection:.2f} mm")
    
    # ── Anchor designs ──
    for design in result.anchor_designs:
        print(f"\n  {'─' * 60}")
        print(f"  ANCHOR: {design.label}")
        print(f"  {'─' * 60}")
        print(f"  Horizontal reaction: {design.reaction_horizontal:.1f} kN/m")
        print(f"  Force per anchor:    {design.reaction_per_anchor:.1f} kN (@ {design.tendon_force:.1f} kN in tendon)")
        print(f"  Free length:         {design.free_length:.1f} m")
        print(f"  Bond length:         {design.bond_length_required:.1f} m")
        print(f"  Total length:        {design.total_length:.1f} m")
        print(f"  Tendon:              {design.tendon_recommendation}")
        print(f"  FOS (bond):          {design.fos_bond_actual:.2f}")
        print(f"  FOS (tendon):        {design.fos_tendon_actual:.2f}")
        print(f"  Status:              {design.status}")
        
        for note in design.notes:
            print(f"    {note}")
