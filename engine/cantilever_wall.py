"""
Module 7A: Cantilever Sheet Pile Wall Analysis
================================================

Analysis of unsupported (cantilever) sheet pile walls embedded in soil.
No anchors or struts — wall relies entirely on passive resistance below
the excavation level.

Methods implemented:
1. Free Earth Support (FES)
   - Wall rotates about pivot point near toe
   - Moment equilibrium about toe → embedment depth
   - FOS applied to passive resistance (typically 1.5-2.0)
   - Simple, conservative, suitable for preliminary design

2. Fixed Earth Support — Blum's Equivalent Beam (FES-Blum)
   - Wall is effectively fixed at depth of zero net pressure
   - Accounts for wall stiffness and fixity
   - More realistic BM distribution
   - Uses subgrade reaction concept below pivot

Both methods share:
- Active pressure: Rankine (IS 9527), full depth behind wall
- Passive pressure: Rankine, below excavation in front of wall
- Net pressure diagram below excavation
- FOS on passive resistance
- Beam FE for BM/SF/deflection
- Integration with section library for adequacy check

References:
- IS 9527 (Part 1):1981 — Design of retaining structures (sheet pile walls)
- IS 14458 (Part 1):1998 — Design of sheet pile walls
- Bowles, Foundation Analysis and Design, 5th Ed. Ch 11
- Terzaghi, Peck & Mesri — Soil Mechanics in Engineering Practice
- USS Steel Sheet Piling Design Manual (1984)
- CIRIA C760 — Embedded retaining walls


"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

import sys, os
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


# ══════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════

class CantileverMethod(Enum):
    FREE_EARTH = "Free Earth Support"
    BLUM_FIXED = "Blum's Fixed Earth"


@dataclass
class CantileverResult:
    """Complete results of cantilever wall analysis."""
    method: CantileverMethod
    excavation_depth: float         # m
    embedment_depth: float          # m (calculated, without FOS)
    embedment_with_fos: float       # m (with FOS on passive)
    total_wall_length: float        # m (excavation + embedment_with_fos)
    toe_kick: float                 # m (additional embedment for safety)

    # ── Factors of safety ──
    fos_passive: float              # FOS on passive resistance (applied)
    fos_moment: float               # FOS = restoring moment / overturning moment
    fos_horizontal: float           # FOS = passive force / active force below exc.

    # ── Forces per m of wall ──
    total_active_force: float       # kN/m
    total_passive_force: float      # kN/m (unfactored)
    max_bm: float                   # kN·m/m
    max_bm_depth: float             # m below GL
    max_sf: float                   # kN/m

    # ── Pressure profiles ──
    depths: List[float]             # m below GL
    active_pressures: List[float]   # kPa (total active)
    passive_pressures: List[float]  # kPa (total passive, zero above exc)
    net_pressures: List[float]      # kPa (active - passive, +ve = toward exc)

    # ── Internal forces ──
    bending_moments: List[float]    # kN·m/m
    shear_forces: List[float]       # kN/m
    deflections: List[float]        # mm

    # ── Blum-specific ──
    pivot_depth: float = 0.0        # m below GL (depth of zero net pressure)
    toe_force: float = 0.0          # kN/m (concentrated force at toe, Blum)

    # ── Embedment iteration data ──
    embedment_trials: List[Tuple[float, float]] = field(default_factory=list)
    # [(D, net_moment)] for plotting convergence


# ══════════════════════════════════════════════════════════════
# PRESSURE COMPUTATION HELPERS
# ══════════════════════════════════════════════════════════════

def _compute_Ka(phi: float) -> float:
    """Rankine active coefficient."""
    return math.tan(math.radians(45 - phi / 2)) ** 2


def _compute_Kp(phi: float) -> float:
    """Rankine passive coefficient."""
    return math.tan(math.radians(45 + phi / 2)) ** 2


def _get_soil_at_depth(layers: List[SoilLayer], depth: float) -> SoilLayer:
    """Return the soil layer containing a given depth."""
    cum = 0.0
    for lay in layers:
        if depth <= cum + lay.thickness + 0.001:
            return lay
        cum += lay.thickness
    return layers[-1]


def _compute_pressures_at_depth(
    depth: float,
    layers: List[SoilLayer],
    gwt_behind: float,
    gwt_front: float,
    surcharge: float,
    exc_depth: float,
) -> Tuple[float, float, float]:
    """
    Compute active and passive pressure at a given depth.

    Returns: (active_total, passive_total, net_pressure)
    where net = active - passive (positive = drives wall toward excavation)
    """
    gamma_w = 9.81

    # ── Vertical effective stress (behind wall) ──
    sigma_v = surcharge
    cum = 0.0
    for lay in layers:
        d_top = cum
        d_bot = cum + lay.thickness
        if depth <= d_top:
            break
        d_in = min(depth, d_bot) - d_top
        if depth > gwt_behind and d_bot > gwt_behind:
            # Partially above/below GWT
            above = max(0, min(gwt_behind, min(depth, d_bot)) - d_top)
            below = d_in - above
            sigma_v += lay.gamma * above + (lay.gamma_sat - gamma_w) * below
        else:
            sigma_v += lay.gamma * d_in
        cum = d_bot

    # ── Active pressure (behind wall, full depth) ──
    layer = _get_soil_at_depth(layers, depth)
    Ka = _compute_Ka(layer.phi_eff)
    c = layer.c_eff

    sigma_ah_eff = Ka * sigma_v - 2 * c * math.sqrt(Ka)
    sigma_ah_eff = max(sigma_ah_eff, 0)  # tension cutoff

    # Water pressure behind wall
    u_behind = max(0, (depth - gwt_behind) * gamma_w) if depth > gwt_behind else 0

    active_total = sigma_ah_eff + u_behind

    # ── Passive pressure (in front of wall, only below excavation) ──
    passive_total = 0.0
    if depth > exc_depth:
        z_below = depth - exc_depth

        # Vertical stress in front (only self-weight below excavation)
        sigma_v_front = 0.0
        cum_f = 0.0
        for lay in layers:
            d_top = cum_f
            d_bot = cum_f + lay.thickness
            # Map depth below GL to layer
            if exc_depth + z_below <= d_top:
                break
            if d_bot <= exc_depth:
                cum_f = d_bot
                continue
            # Depth range within this layer below excavation
            z_start = max(exc_depth, d_top) - exc_depth
            z_end = min(exc_depth + z_below, d_bot) - exc_depth
            if z_end <= z_start:
                cum_f = d_bot
                continue
            dz = z_end - z_start
            actual_depth_mid = exc_depth + (z_start + z_end) / 2
            if actual_depth_mid > gwt_front:
                sigma_v_front += (lay.gamma_sat - gamma_w) * dz
            else:
                sigma_v_front += lay.gamma * dz
            cum_f = d_bot

        # Passive layer at this depth
        layer_front = _get_soil_at_depth(layers, depth)
        Kp = _compute_Kp(layer_front.phi_eff)
        c_f = layer_front.c_eff

        sigma_ph_eff = Kp * sigma_v_front + 2 * c_f * math.sqrt(Kp)

        # Water pressure in front
        u_front = max(0, (depth - gwt_front) * gamma_w) if depth > gwt_front else 0

        passive_total = sigma_ph_eff + u_front

    net = active_total - passive_total
    return active_total, passive_total, net


# ══════════════════════════════════════════════════════════════
# FREE EARTH SUPPORT METHOD
# ══════════════════════════════════════════════════════════════

def analyze_cantilever_free_earth(
    project: ProjectInput,
    fos_passive: float = 1.5,
    toe_kick_factor: float = 1.2,
    dz: float = 0.05,
    EI: float = 50000.0,
    max_embedment: float = 20.0,
) -> CantileverResult:
    """
    Cantilever wall analysis using Free Earth Support method.

    The wall rotates about a pivot near the toe. Passive resistance
    is divided by FOS. Embedment found by moment equilibrium about toe.

    Parameters:
        project: ProjectInput with soil, water, excavation
        fos_passive: Factor of safety on passive resistance (IS 9527: 1.5-2.0)
        toe_kick_factor: Additional embedment multiplier (typically 1.2)
        dz: Depth increment for calculations (m)
        EI: Flexural rigidity (kN·m²/m) for deflection calc
        max_embedment: Maximum embedment to search (m)

    References:
        IS 9527 Cl. 5.3: FOS on passive ≥ 1.5
        IS 14458 Cl. 6.2: Cantilever wall design
        Bowles Ch. 11.8: Free earth support
    """
    exc = project.excavation_depth
    layers = project.soil_layers
    gwt_behind = project.water_table.depth_behind_wall
    gwt_front = project.water_table.depth_in_excavation or exc
    surcharge = sum(s.magnitude for s in project.surcharges if s.surcharge_type.value == 'uniform')

    # ── Step 1: Find embedment by moment equilibrium ──
    # Iterate embedment D from 0 to max_embedment.
    # At each D, compute:
    #   M_active = sum of (active force × arm about toe) for full wall height
    #   M_passive = sum of (passive force × arm about toe) below excavation
    #   Check: M_passive / fos_passive >= M_active

    embedment_trials = []
    D_required = max_embedment  # fallback

    for D_trial_mm in range(0, int(max_embedment * 1000), int(dz * 1000)):
        D = D_trial_mm / 1000.0
        total_depth = exc + D
        toe = total_depth

        M_active = 0.0
        M_passive = 0.0
        F_active = 0.0
        F_passive = 0.0

        z = 0.0
        while z < total_depth - dz / 2:
            z_mid = z + dz / 2
            if z_mid > total_depth:
                break
            act, pas, _ = _compute_pressures_at_depth(
                z_mid, layers, gwt_behind, gwt_front, surcharge, exc)
            arm = toe - z_mid  # moment arm about toe

            F_active += act * dz
            M_active += act * dz * arm

            F_passive += pas * dz
            M_passive += pas * dz * arm

            z += dz

        net_moment = M_passive / fos_passive - M_active
        embedment_trials.append((D, net_moment))

        if net_moment >= 0 and D > 0.1:
            D_required = D
            break

    # Apply toe kick
    D_with_fos = D_required
    D_final = D_with_fos * toe_kick_factor
    total_length = exc + D_final

    # ── Step 2: FOS checks ──
    # Recompute with D_final
    total_depth = total_length
    M_active = 0.0
    M_passive_unfactored = 0.0
    F_active_total = 0.0
    F_passive_total = 0.0

    depths = []
    active_p = []
    passive_p = []
    net_p = []

    z = 0.0
    while z <= total_depth + dz / 2:
        act, pas, net = _compute_pressures_at_depth(
            z, layers, gwt_behind, gwt_front, surcharge, exc)
        depths.append(z)
        active_p.append(act)
        passive_p.append(pas)
        net_p.append(net)

        arm = total_depth - z
        F_active_total += act * dz
        M_active += act * dz * arm
        F_passive_total += pas * dz
        M_passive_unfactored += pas * dz * arm

        z += dz

    fos_moment = M_passive_unfactored / M_active if M_active > 0 else 999
    fos_horiz = F_passive_total / (F_active_total - F_passive_total) if F_active_total > F_passive_total else 999

    # ── Step 3: Beam FE for BM/SF/Deflection ──
    bm, sf, defl, max_bm, max_bm_depth, max_sf = _solve_cantilever_beam(
        depths, active_p, passive_p, EI, fos_passive)

    # ── Find pivot depth (where net pressure changes sign below excavation) ──
    pivot = exc
    for i, d in enumerate(depths):
        if d > exc and net_p[i] < 0:
            pivot = d
            break

    return CantileverResult(
        method=CantileverMethod.FREE_EARTH,
        excavation_depth=exc,
        embedment_depth=D_required,
        embedment_with_fos=D_final,
        total_wall_length=total_length,
        toe_kick=D_final - D_required,
        fos_passive=fos_passive,
        fos_moment=fos_moment,
        fos_horizontal=fos_horiz,
        total_active_force=F_active_total,
        total_passive_force=F_passive_total,
        max_bm=max_bm,
        max_bm_depth=max_bm_depth,
        max_sf=max_sf,
        depths=depths,
        active_pressures=active_p,
        passive_pressures=passive_p,
        net_pressures=net_p,
        bending_moments=bm,
        shear_forces=sf,
        deflections=defl,
        pivot_depth=pivot,
        embedment_trials=embedment_trials,
    )


# ══════════════════════════════════════════════════════════════
# BLUM'S FIXED EARTH SUPPORT METHOD
# ══════════════════════════════════════════════════════════════

def analyze_cantilever_blum(
    project: ProjectInput,
    fos_passive: float = 1.5,
    toe_kick_factor: float = 1.2,
    dz: float = 0.05,
    EI: float = 50000.0,
    max_embedment: float = 20.0,
) -> CantileverResult:
    """
    Cantilever wall analysis using Blum's Fixed Earth Support method.

    The wall is modelled as fixed at the toe (w=0, theta=0).
    The correct embedment D is found when the fixed-end reaction
    moment at the toe equals zero — meaning the wall naturally
    develops zero rotation at that depth.

    Iteration:
    1. For trial D, build cantilever beam fixed at bottom
    2. Compute fixed-end moment (reaction moment at toe)
    3. Bisect to find D where M_toe = 0

    This gives a deeper embedment and lower max BM than FES,
    because the fixity at toe redistributes moments.

    Parameters: same as free_earth method

    References:
        Blum (1931): Einspannungsverhältnisse bei Bohlwerken
        Bowles Ch. 11.9: Fixed earth support
        USS Steel Sheet Piling Design Manual (1984) Ch. 3
    """
    exc = project.excavation_depth
    layers = project.soil_layers
    gwt_behind = project.water_table.depth_behind_wall
    gwt_front = project.water_table.depth_in_excavation or exc
    surcharge = sum(s.magnitude for s in project.surcharges if s.surcharge_type.value == 'uniform')

    def _compute_toe_moment(D):
        """
        For a given embedment D, build the fixed-bottom cantilever
        and return the fixed-end moment at the toe.
        Positive M_toe means passive is insufficient (need more D).
        """
        total = exc + D
        ds, ap, pp = [], [], []
        z = 0.0
        while z <= total + dz / 2:
            act, pas, _ = _compute_pressures_at_depth(
                z, layers, gwt_behind, gwt_front, surcharge, exc)
            ds.append(z)
            ap.append(act)
            pp.append(pas)
            z += dz

        n = len(ds)
        if n < 3:
            return 1e10, ds, ap, pp

        ndof = 2 * n
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        for i in range(n - 1):
            L = ds[i + 1] - ds[i]
            if L < 1e-6:
                continue
            q1 = ap[i] - pp[i] / fos_passive
            q2 = ap[i + 1] - pp[i + 1] / fos_passive

            k_e = EI / L ** 3 * np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2],
            ])
            f_e = np.array([
                L * (7*q1 + 3*q2) / 20,
                L**2 * (3*q1 + 2*q2) / 60,
                L * (3*q1 + 7*q2) / 20,
                -L**2 * (2*q1 + 3*q2) / 60,
            ])
            dofs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for a in range(4):
                F[dofs[a]] += f_e[a]
                for b in range(4):
                    K[dofs[a], dofs[b]] += k_e[a, b]

        # Fixed at bottom: w=0 AND theta=0
        penalty = 1e15
        K[ndof-2, ndof-2] += penalty
        K[ndof-1, ndof-1] += penalty

        try:
            U = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            return 1e10, ds, ap, pp

        # Reaction moment at toe = penalty × theta_toe
        M_toe = penalty * U[ndof - 1]
        return M_toe, ds, ap, pp

    # ── Coarse search for sign change ──
    embedment_trials = []
    D_min, D_max = 0.5, min(max_embedment, exc * 3)

    prev_M = None
    D_lo, D_hi = D_min, D_max

    for D_mm in range(int(D_min * 1000), int(D_max * 1000), 200):
        D = D_mm / 1000.0
        M_toe, _, _, _ = _compute_toe_moment(D)
        embedment_trials.append((D, M_toe))

        if prev_M is not None and prev_M * M_toe < 0:
            D_lo = embedment_trials[-2][0]
            D_hi = D
            break
        prev_M = M_toe

    # ── Bisection refinement ──
    D_required = (D_lo + D_hi) / 2
    for _ in range(30):
        D_mid = (D_lo + D_hi) / 2
        M_mid, _, _, _ = _compute_toe_moment(D_mid)
        M_lo, _, _, _ = _compute_toe_moment(D_lo)

        if abs(M_mid) < 0.1:  # converged
            D_required = D_mid
            break
        if M_lo * M_mid < 0:
            D_hi = D_mid
        else:
            D_lo = D_mid
        D_required = D_mid

    D_final = D_required * toe_kick_factor
    total_length = exc + D_final

    # ── Final analysis with D_final ──
    total_depth = total_length
    depths, active_p, passive_p, net_p = [], [], [], []
    F_active_total, F_passive_total = 0.0, 0.0
    M_active, M_passive_unfactored = 0.0, 0.0

    z = 0.0
    while z <= total_depth + dz / 2:
        act, pas, net = _compute_pressures_at_depth(
            z, layers, gwt_behind, gwt_front, surcharge, exc)
        depths.append(z)
        active_p.append(act)
        passive_p.append(pas)
        net_p.append(net)

        arm = total_depth - z
        F_active_total += act * dz
        M_active += act * dz * arm
        F_passive_total += pas * dz
        M_passive_unfactored += pas * dz * arm
        z += dz

    fos_moment = M_passive_unfactored / M_active if M_active > 0 else 999
    fos_horiz = F_passive_total / (F_active_total - F_passive_total) if F_active_total > F_passive_total else 999

    # Solve final beam (fixed at bottom)
    bm, sf, defl, max_bm, max_bm_depth, max_sf = _solve_cantilever_beam(
        depths, active_p, passive_p, EI, fos_passive)

    # Pivot depth
    pivot = exc
    for i, d in enumerate(depths):
        if d > exc:
            factored_net = active_p[i] - passive_p[i] / fos_passive
            if factored_net < 0:
                pivot = d
                break

    toe_force = abs(sf[-1]) if sf else 0

    return CantileverResult(
        method=CantileverMethod.BLUM_FIXED,
        excavation_depth=exc,
        embedment_depth=D_required,
        embedment_with_fos=D_final,
        total_wall_length=total_length,
        toe_kick=D_final - D_required,
        fos_passive=fos_passive,
        fos_moment=fos_moment,
        fos_horizontal=fos_horiz,
        total_active_force=F_active_total,
        total_passive_force=F_passive_total,
        max_bm=max_bm,
        max_bm_depth=max_bm_depth,
        max_sf=max_sf,
        depths=depths,
        active_pressures=active_p,
        passive_pressures=passive_p,
        net_pressures=net_p,
        bending_moments=bm,
        shear_forces=sf,
        deflections=defl,
        pivot_depth=pivot,
        toe_force=toe_force,
        embedment_trials=embedment_trials,
    )


# ══════════════════════════════════════════════════════════════
# BEAM FE SOLVER — CANTILEVER SPECIFIC
# ══════════════════════════════════════════════════════════════

def _solve_cantilever_beam(
    depths: List[float],
    active: List[float],
    passive: List[float],
    EI: float,
    fos_passive: float,
) -> Tuple[List[float], List[float], List[float], float, float, float]:
    """
    Solve cantilever wall as a beam with:
    - Fixed at bottom (embedment toe)
    - Free at top
    - Net load = active - passive/FOS

    Uses Euler-Bernoulli beam FE (2 DOF per node: w, theta).

    Returns: (BM, SF, deflection, max_BM, max_BM_depth, max_SF)
    """
    n = len(depths)
    if n < 3:
        return [0]*n, [0]*n, [0]*n, 0, 0, 0

    ndof = 2 * n
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # Assemble beam elements
    for i in range(n - 1):
        L = depths[i + 1] - depths[i]
        if L < 1e-6:
            continue

        # Net load at nodes (active - factored passive)
        q1 = active[i] - passive[i] / fos_passive
        q2 = active[i + 1] - passive[i + 1] / fos_passive

        # Stiffness matrix (Euler-Bernoulli)
        k_e = EI / L ** 3 * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
        ])

        # Equivalent nodal loads (linearly varying)
        f_e = np.array([
            L * (7 * q1 + 3 * q2) / 20,
            L ** 2 * (3 * q1 + 2 * q2) / 60,
            L * (3 * q1 + 7 * q2) / 20,
            -L ** 2 * (2 * q1 + 3 * q2) / 60,
        ])

        dofs = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        for a in range(4):
            F[dofs[a]] += f_e[a]
            for b in range(4):
                K[dofs[a], dofs[b]] += k_e[a, b]

    # Boundary conditions: fixed at bottom (w=0, theta=0)
    fixed_dofs = [ndof - 2, ndof - 1]

    # Apply BCs by penalty method
    penalty = 1e15
    for d in fixed_dofs:
        K[d, d] += penalty

    # Solve
    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        return [0]*n, [0]*n, [0]*n, 0, 0, 0

    # Extract deflections (mm)
    deflections = [U[2 * i] * 1000 for i in range(n)]

    # Compute BM and SF from element forces
    bm = [0.0] * n
    sf = [0.0] * n

    for i in range(n - 1):
        L = depths[i + 1] - depths[i]
        if L < 1e-6:
            continue
        dofs = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        u_e = np.array([U[d] for d in dofs])

        k_e = EI / L ** 3 * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
        ])

        q1 = active[i] - passive[i] / fos_passive
        q2 = active[i + 1] - passive[i + 1] / fos_passive

        f_e = np.array([
            L * (7 * q1 + 3 * q2) / 20,
            L ** 2 * (3 * q1 + 2 * q2) / 60,
            L * (3 * q1 + 7 * q2) / 20,
            -L ** 2 * (2 * q1 + 3 * q2) / 60,
        ])

        elem_forces = k_e @ u_e - f_e
        # elem_forces = [V1, M1, V2, M2]
        sf[i] = -elem_forces[0]
        bm[i] = elem_forces[1]
        sf[i + 1] = elem_forces[2]
        bm[i + 1] = -elem_forces[3]

    max_bm = max(abs(m) for m in bm)
    max_bm_idx = max(range(n), key=lambda i: abs(bm[i]))
    max_bm_depth = depths[max_bm_idx]
    max_sf = max(abs(s) for s in sf)

    return bm, sf, deflections, max_bm, max_bm_depth, max_sf


# ══════════════════════════════════════════════════════════════
# COMPARISON: RUN BOTH METHODS
# ══════════════════════════════════════════════════════════════

def analyze_cantilever_both(
    project: ProjectInput,
    fos_passive: float = 1.5,
    toe_kick_factor: float = 1.2,
    dz: float = 0.05,
    EI: float = 50000.0,
) -> Tuple[CantileverResult, CantileverResult]:
    """
    Run both Free Earth and Blum's methods for comparison.
    Returns: (free_earth_result, blum_result)
    """
    res_fe = analyze_cantilever_free_earth(
        project, fos_passive, toe_kick_factor, dz, EI)
    res_blum = analyze_cantilever_blum(
        project, fos_passive, toe_kick_factor, dz, EI)
    return res_fe, res_blum


# ══════════════════════════════════════════════════════════════
# QUICK DESIGN TABLE
# ══════════════════════════════════════════════════════════════

def cantilever_design_table(
    project: ProjectInput,
    exc_depths: Optional[List[float]] = None,
    EI: float = 50000.0,
) -> List[Dict]:
    """
    Generate a design table for multiple excavation depths.
    Useful for preliminary sizing / feasibility.

    Returns list of dicts with key results per depth.
    """
    if exc_depths is None:
        exc_depths = [2, 3, 4, 5, 6, 7, 8]

    results = []
    for exc in exc_depths:
        proj = ProjectInput(
            name=project.name,
            excavation_depth=exc,
            soil_layers=project.soil_layers,
            water_table=project.water_table,
            surcharges=project.surcharges,
        )
        try:
            res = analyze_cantilever_free_earth(proj, EI=EI)
            results.append({
                'exc_depth': exc,
                'embedment': res.embedment_with_fos,
                'total_length': res.total_wall_length,
                'max_bm': res.max_bm,
                'max_sf': res.max_sf,
                'fos_moment': res.fos_moment,
                'status': 'OK' if res.fos_moment >= 1.5 else 'CHECK',
            })
        except Exception:
            results.append({
                'exc_depth': exc,
                'embedment': 0,
                'total_length': 0,
                'max_bm': 0,
                'max_sf': 0,
                'fos_moment': 0,
                'status': 'FAIL',
            })
    return results
