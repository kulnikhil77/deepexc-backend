"""
Module 9: Staged Excavation / Construction Sequence Analysis
=============================================================

Simulates the real construction sequence:
  Stage 0: Wall installed (driven to full depth, no excavation)
  Stage 1: Excavate to first anchor level + working margin
  Stage 2: Install Anchor 1, prestress (if applicable)
  Stage 3: Excavate to next anchor level + working margin
  Stage 4: Install Anchor 2, prestress
  ...
  Stage N: Excavate to final depth

At each stage the solver:
  - Computes active pressure (behind wall, full depth)
  - Computes passive pressure (in front, below current excavation)
  - Applies net pressure (active - passive/FOS) as distributed load
  - Models only the anchors installed so far as supports
  - Solves beam FE for BM, SF, deflection

The peak forces often occur at an INTERMEDIATE stage (e.g. when the
excavation is deep but the lowest anchor isn't yet installed). The
envelope across all stages gives the DESIGN forces.

Method: Independent equilibrium at each stage (FHWA-IF-99-015).
Each stage is a snapshot analysis, not incremental. This is the
standard approach in practice (Bowles, CIRIA C760, USS Manual).

References:
  - FHWA-IF-99-015 Ground Anchors and Anchored Systems, Ch. 5
  - CIRIA C760 Embedded Retaining Walls, Section 5
  - IS 14458 (Part 1):1998 Sheet Pile Walls
  - Bowles 5th Ed., Ch. 11.11 Construction Sequence


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
    from engine.models import ProjectInput, SoilLayer, WaterTable, SoilType, Surcharge, SurchargeType
    from engine.anchored_wall import Anchor
except ImportError:
    from models import ProjectInput, SoilLayer, WaterTable, SoilType, Surcharge, SurchargeType
    from anchored_wall import Anchor


# ══════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════

class StageAction(Enum):
    EXCAVATE = "Excavate"
    INSTALL_ANCHOR = "Install Anchor"
    DEWATER = "Dewater"
    APPLY_SURCHARGE = "Apply Surcharge"
    REMOVE_SURCHARGE = "Remove Surcharge"


@dataclass
class ConstructionStage:
    """Definition of one construction stage."""
    stage_number: int
    description: str
    excavation_depth: float          # m below GL (current excavation level)
    active_anchor_indices: List[int] # which anchors (0-based) are installed & active
    gwt_in_excavation: float = None  # m below GL (if dewatered, else = exc depth)
    surcharge: float = 0.0           # kPa at GL (may change per stage)
    fos_passive: float = 1.5         # FOS on passive for this stage


@dataclass
class StageResult:
    """Results for a single construction stage."""
    stage: ConstructionStage
    depths: List[float]              # m below GL
    bending_moments: List[float]     # kN·m/m
    shear_forces: List[float]        # kN/m
    deflections: List[float]         # mm
    active_pressures: List[float]    # kPa (total active at each depth)
    passive_pressures: List[float]   # kPa (total passive at each depth)
    net_pressures: List[float]       # kPa (net load on wall)

    # ── Peak values ──
    max_bm: float
    max_bm_depth: float
    max_sf: float
    max_sf_depth: float
    max_defl: float
    max_defl_depth: float

    # ── Reactions ──
    anchor_reactions: List[float]    # kN/m at each active anchor
    toe_reaction: float              # kN/m

    # ── Status ──
    status: str = "OK"               # "OK" or warning message


@dataclass
class StagedAnalysisResult:
    """Complete staged analysis with per-stage results and envelope."""
    stages: List[StageResult]
    n_stages: int

    # ── Envelope across all stages ──
    envelope_depths: List[float]
    envelope_bm_max: List[float]     # max |BM| at each depth
    envelope_bm_min: List[float]     # min BM at each depth (may be negative)
    envelope_sf_max: List[float]
    envelope_defl_max: List[float]

    # ── Design values (peaks of envelope) ──
    design_bm: float                 # max |BM| from any stage
    design_bm_depth: float
    design_bm_stage: int             # which stage produced peak BM
    design_sf: float
    design_sf_depth: float
    design_sf_stage: int
    design_defl: float
    design_defl_depth: float
    design_defl_stage: int

    # ── Summary table ──
    summary: List[Dict]              # per-stage summary for table display

    # ── Input echo ──
    wall_length: float
    EI: float
    n_anchors: int


# ══════════════════════════════════════════════════════════════
# PRESSURE COMPUTATION (reused from cantilever_wall)
# ══════════════════════════════════════════════════════════════

def _compute_Ka(phi: float) -> float:
    return math.tan(math.radians(45 - phi / 2)) ** 2

def _compute_Kp(phi: float) -> float:
    return math.tan(math.radians(45 + phi / 2)) ** 2

def _get_soil_at_depth(layers: List[SoilLayer], depth: float) -> SoilLayer:
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
    Active and passive pressure at depth.
    Returns: (active_total, passive_total, net)
    """
    gamma_w = 9.81

    # Vertical effective stress behind wall
    sigma_v = surcharge
    cum = 0.0
    for lay in layers:
        d_top = cum
        d_bot = cum + lay.thickness
        if depth <= d_top:
            break
        d_in = min(depth, d_bot) - d_top
        if depth > gwt_behind and d_bot > gwt_behind:
            above = max(0, min(gwt_behind, min(depth, d_bot)) - d_top)
            below = d_in - above
            sigma_v += lay.gamma * above + (lay.gamma_sat - gamma_w) * below
        else:
            sigma_v += lay.gamma * d_in
        cum = d_bot

    layer = _get_soil_at_depth(layers, depth)
    Ka = _compute_Ka(layer.phi_eff)
    c = layer.c_eff
    sigma_ah_eff = max(Ka * sigma_v - 2 * c * math.sqrt(Ka), 0)
    u_behind = max(0, (depth - gwt_behind) * gamma_w) if depth > gwt_behind else 0
    active_total = sigma_ah_eff + u_behind

    # Passive — only below excavation
    passive_total = 0.0
    if depth > exc_depth:
        z_below = depth - exc_depth
        sigma_v_front = 0.0
        cum_f = 0.0
        for lay in layers:
            d_top = cum_f
            d_bot = cum_f + lay.thickness
            if exc_depth + z_below <= d_top:
                break
            if d_bot <= exc_depth:
                cum_f = d_bot
                continue
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

        layer_front = _get_soil_at_depth(layers, depth)
        Kp = _compute_Kp(layer_front.phi_eff)
        c_f = layer_front.c_eff
        sigma_ph_eff = Kp * sigma_v_front + 2 * c_f * math.sqrt(Kp)
        u_front = max(0, (depth - gwt_front) * gamma_w) if depth > gwt_front else 0
        passive_total = sigma_ph_eff + u_front

    net = active_total - passive_total
    return active_total, passive_total, net


# ══════════════════════════════════════════════════════════════
# BEAM FE SOLVER (from anchored_wall, adapted for staged)
# ══════════════════════════════════════════════════════════════

def _beam_stiffness(L: float, EI: float) -> np.ndarray:
    k = EI / L ** 3
    return k * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ])


def _nodal_loads_linear(q1: float, q2: float, L: float) -> np.ndarray:
    return np.array([
        L * (7*q1 + 3*q2) / 20,
        L**2 * (3*q1 + 2*q2) / 60,
        L * (3*q1 + 7*q2) / 20,
        -L**2 * (2*q1 + 3*q2) / 60,
    ])


def _solve_stage_beam(
    wall_length: float,
    depths_pressure: List[float],
    net_load: List[float],
    active_anchor_levels: List[float],
    EI: float,
    n_elements: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], float]:
    """
    Solve beam FE for one construction stage.

    Boundary conditions:
    - Fixed at bottom (wall toe: w=0, θ=0)
      (Toe is embedded in soil / grouted bench)
    - Pin support at each active anchor level (w=0)
    - Free at top

    Loading:
    - net_load = active - passive/FOS (positive = drives wall)

    Returns: (depths, BM, SF, deflection_mm, anchor_reactions, toe_reaction)
    """
    L_elem = wall_length / n_elements
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes

    depths_array = np.linspace(0, wall_length, n_nodes)

    # Interpolate net load to nodes
    load_at_nodes = np.interp(depths_array, depths_pressure, net_load)

    # Assemble global stiffness and load
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    for i in range(n_elements):
        k_e = _beam_stiffness(L_elem, EI)
        q1 = load_at_nodes[i]
        q2 = load_at_nodes[i + 1]
        f_e = _nodal_loads_linear(q1, q2, L_elem)

        dofs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for a in range(4):
            F[dofs[a]] += f_e[a]
            for b in range(4):
                K[dofs[a], dofs[b]] += k_e[a, b]

    # ── Boundary conditions ──
    constrained_dofs = []

    toe_w_dof = 2 * (n_nodes - 1)
    toe_t_dof = 2 * (n_nodes - 1) + 1

    if len(active_anchor_levels) == 0:
        # Cantilever stage: fixed toe (w=0, θ=0)
        constrained_dofs.extend([toe_w_dof, toe_t_dof])
    else:
        # Anchored stage: toe still has embedment in soil.
        # Fixed toe (w=0, θ=0) — passive resistance provides
        # rotational restraint, modelled via net load + fixity.
        constrained_dofs.extend([toe_w_dof, toe_t_dof])

    # Anchors: pin (w=0)
    anchor_nodes = []
    for a_level in active_anchor_levels:
        node_idx = int(round(a_level / L_elem))
        node_idx = max(0, min(node_idx, n_nodes - 1))
        anchor_nodes.append(node_idx)
        constrained_dofs.append(2 * node_idx)

    # Solve with constrained DOFs
    free_dofs = [d for d in range(n_dof) if d not in constrained_dofs]

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    # Conditioning check
    try:
        if np.linalg.cond(K_ff) > 1e14:
            # Add small stiffness to stabilize
            K_ff += np.eye(len(free_dofs)) * 1e-6
        u_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        n = len(depths_array)
        return depths_array, np.zeros(n), np.zeros(n), np.zeros(n), [0]*len(active_anchor_levels), 0

    u_global = np.zeros(n_dof)
    for i, d in enumerate(free_dofs):
        u_global[d] = u_f[i]

    # ── Reactions ──
    R_global = K @ u_global - F

    anchor_reactions = []
    for node_idx in anchor_nodes:
        anchor_reactions.append(R_global[2 * node_idx])

    toe_reaction = R_global[toe_w_dof]

    # ── Internal forces ──
    moments = np.zeros(n_nodes)
    shears = np.zeros(n_nodes)

    for i in range(n_elements):
        dofs = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u_e = u_global[dofs]
        k_e = _beam_stiffness(L_elem, EI)
        q1 = load_at_nodes[i]
        q2 = load_at_nodes[i + 1]
        f_fixed = _nodal_loads_linear(q1, q2, L_elem)
        f_int = k_e @ u_e - f_fixed

        if i == 0:
            shears[0] = f_int[0]
            moments[0] = f_int[1]
        shears[i + 1] = -f_int[2]
        moments[i + 1] = f_int[3]

    deflections = np.array([u_global[2 * i] * 1000 for i in range(n_nodes)])

    return depths_array, moments, shears, deflections, anchor_reactions, toe_reaction


# ══════════════════════════════════════════════════════════════
# STAGE GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_stages(
    excavation_depth: float,
    anchors: List[Anchor],
    exc_step: float = 1.0,
    working_margin: float = 0.5,
    surcharge: float = 0.0,
    gwt_behind: float = 99.0,
    fos_passive: float = 1.5,
) -> List[ConstructionStage]:
    """
    Auto-generate construction stages from anchor layout.

    Logic:
    1. Excavate in steps of exc_step (default 1m)
    2. When excavation passes an anchor level, install that anchor
    3. Continue to final depth

    Anchor installation rule:
    - Excavate to (anchor_level + working_margin) before installing
    - This gives the drill rig space to work

    Parameters:
        excavation_depth: final excavation depth (m)
        anchors: list of Anchor objects (sorted by level)
        exc_step: excavation increment (m)
        working_margin: extra depth below anchor before install (m)
        surcharge: surcharge at GL (kPa)
        gwt_behind: GWT behind wall (m)
        fos_passive: FOS on passive resistance
    """
    stages = []
    anchor_levels = sorted([(i, a.level) for i, a in enumerate(anchors)], key=lambda x: x[1])
    active_anchors = []
    next_anchor_idx = 0  # index into anchor_levels

    # Stage 0: Wall installed, no excavation
    stages.append(ConstructionStage(
        stage_number=0,
        description="Wall installed, no excavation",
        excavation_depth=0.0,
        active_anchor_indices=[],
        surcharge=surcharge,
        fos_passive=fos_passive,
    ))

    current_exc = 0.0
    stage_num = 1

    while current_exc < excavation_depth - 0.01:
        # Next excavation depth
        next_exc = min(current_exc + exc_step, excavation_depth)

        # Check if any anchor needs to be installed before/at this excavation
        anchors_to_install = []
        while next_anchor_idx < len(anchor_levels):
            a_orig_idx, a_level = anchor_levels[next_anchor_idx]
            install_exc = a_level + working_margin
            if install_exc <= next_exc + 0.01:
                # Need to excavate to install_exc first (if not already there)
                if current_exc < install_exc - 0.01:
                    # Excavation stage to reach anchor install depth
                    stages.append(ConstructionStage(
                        stage_number=stage_num,
                        description=f"Excavate to {install_exc:.1f}m (for Anchor {a_orig_idx+1})",
                        excavation_depth=install_exc,
                        active_anchor_indices=list(active_anchors),
                        surcharge=surcharge,
                        fos_passive=fos_passive,
                    ))
                    current_exc = install_exc
                    stage_num += 1

                # Install anchor
                active_anchors.append(a_orig_idx)
                stages.append(ConstructionStage(
                    stage_number=stage_num,
                    description=f"Install Anchor {a_orig_idx+1} at {a_level:.1f}m",
                    excavation_depth=current_exc,
                    active_anchor_indices=list(active_anchors),
                    surcharge=surcharge,
                    fos_passive=fos_passive,
                ))
                stage_num += 1
                next_anchor_idx += 1
            else:
                break

        # Excavate to next_exc (if not already there from anchor install)
        if next_exc > current_exc + 0.01:
            stages.append(ConstructionStage(
                stage_number=stage_num,
                description=f"Excavate to {next_exc:.1f}m",
                excavation_depth=next_exc,
                active_anchor_indices=list(active_anchors),
                surcharge=surcharge,
                fos_passive=fos_passive,
            ))
            current_exc = next_exc
            stage_num += 1
        else:
            current_exc = next_exc

    # Install any remaining anchors (shouldn't normally happen)
    while next_anchor_idx < len(anchor_levels):
        a_orig_idx, a_level = anchor_levels[next_anchor_idx]
        active_anchors.append(a_orig_idx)
        stages.append(ConstructionStage(
            stage_number=stage_num,
            description=f"Install Anchor {a_orig_idx+1} at {a_level:.1f}m",
            excavation_depth=current_exc,
            active_anchor_indices=list(active_anchors),
            surcharge=surcharge,
            fos_passive=fos_passive,
        ))
        stage_num += 1
        next_anchor_idx += 1

    return stages


# ══════════════════════════════════════════════════════════════
# MAIN STAGED ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_staged_excavation(
    project: ProjectInput,
    anchors: List[Anchor],
    wall_toe_level: float,
    EI: float = 50000.0,
    stages: Optional[List[ConstructionStage]] = None,
    exc_step: float = 1.0,
    working_margin: float = 0.5,
    fos_passive: float = 1.5,
    n_elements: int = 100,
    dz_pressure: float = 0.05,
) -> StagedAnalysisResult:
    """
    Run staged excavation analysis.

    If stages is None, auto-generates stages from anchor layout.

    Parameters:
        project: ProjectInput
        anchors: list of Anchor objects
        wall_toe_level: wall toe depth (m)
        EI: flexural rigidity (kN·m²/m)
        stages: optional custom stage list
        exc_step: auto-generation excavation step (m)
        working_margin: depth below anchor for install (m)
        fos_passive: FOS on passive resistance
        n_elements: beam FE elements
        dz_pressure: pressure computation increment (m)

    Returns:
        StagedAnalysisResult with per-stage results and envelope
    """
    layers = project.soil_layers
    gwt_behind = project.water_table.depth_behind_wall
    gwt_front_base = project.water_table.depth_in_excavation or project.excavation_depth
    surcharge = sum(s.magnitude for s in project.surcharges
                    if s.surcharge_type.value == 'uniform')

    # Ensure soil extends below wall toe
    total_soil = sum(l.thickness for l in layers)
    if total_soil < wall_toe_level + 1.0:
        layers = list(layers)
        extra = wall_toe_level + 2.0 - total_soil
        last = layers[-1]
        layers[-1] = SoilLayer(
            name=last.name, thickness=last.thickness + extra,
            gamma=last.gamma, gamma_sat=last.gamma_sat,
            c_eff=last.c_eff, phi_eff=last.phi_eff,
            c_u=last.c_u, soil_type=last.soil_type,
        )

    # ── Generate stages if not provided ──
    if stages is None:
        stages = generate_stages(
            excavation_depth=project.excavation_depth,
            anchors=anchors,
            exc_step=exc_step,
            working_margin=working_margin,
            surcharge=surcharge,
            gwt_behind=gwt_behind,
            fos_passive=fos_passive,
        )

    # ── Run each stage ──
    stage_results = []

    for stage in stages:
        exc_d = stage.excavation_depth
        gwt_front = stage.gwt_in_excavation if stage.gwt_in_excavation is not None else max(exc_d, gwt_front_base)
        q = stage.surcharge if stage.surcharge else surcharge
        fos_p = stage.fos_passive

        # Skip stage 0 if no excavation (just record zeros)
        if exc_d < 0.01:
            n_pts = int(wall_toe_level / dz_pressure) + 1
            d_list = [i * dz_pressure for i in range(n_pts)]
            stage_results.append(StageResult(
                stage=stage, depths=d_list,
                bending_moments=[0]*n_pts, shear_forces=[0]*n_pts,
                deflections=[0]*n_pts,
                active_pressures=[0]*n_pts, passive_pressures=[0]*n_pts,
                net_pressures=[0]*n_pts,
                max_bm=0, max_bm_depth=0, max_sf=0, max_sf_depth=0,
                max_defl=0, max_defl_depth=0,
                anchor_reactions=[], toe_reaction=0,
                status="OK (no excavation)",
            ))
            continue

        # ── Build pressure arrays ──
        depths_p, active_p, passive_p, net_p, net_factored = [], [], [], [], []
        z = 0.0
        is_cantilever = len(stage.active_anchor_indices) == 0

        while z <= wall_toe_level + dz_pressure / 2:
            act, pas, net = _compute_pressures_at_depth(
                z, layers, gwt_behind, gwt_front, q, exc_d)
            depths_p.append(z)
            active_p.append(act)
            passive_p.append(pas)
            net_p.append(net)

            if is_cantilever:
                # Cantilever: active drives above excavation.
                # Below excavation: net = max(0, active - passive/FOS).
                # Passive is a RESISTANCE — it reduces active but cannot
                # push the wall backward (create reverse loading).
                if z <= exc_d:
                    net_factored.append(act)
                else:
                    net_f = act - pas / fos_p if pas > 0 else act
                    net_factored.append(max(0.0, net_f))
            else:
                # Anchored stage: same principle.
                # Above excavation: no passive, full active drives wall.
                # Below excavation: passive reduces active but net >= 0.
                if z <= exc_d:
                    net_factored.append(act)
                else:
                    net_f = act - pas / fos_p if pas > 0 else act
                    net_factored.append(max(0.0, net_f))

            z += dz_pressure

        # ── Active anchor levels for this stage ──
        active_levels = [anchors[i].level for i in stage.active_anchor_indices]

        # ── Solve beam FE ──
        depths_arr, bm, sf, defl, anc_rxn, toe_rxn = _solve_stage_beam(
            wall_length=wall_toe_level,
            depths_pressure=depths_p,
            net_load=net_factored,
            active_anchor_levels=active_levels,
            EI=EI,
            n_elements=n_elements,
        )

        # Convert numpy to lists
        depths_list = depths_arr.tolist()
        bm_list = bm.tolist()
        sf_list = sf.tolist()
        defl_list = defl.tolist()

        # Peak values
        max_bm = max(abs(m) for m in bm_list) if bm_list else 0
        max_bm_idx = max(range(len(bm_list)), key=lambda i: abs(bm_list[i])) if bm_list else 0
        max_sf = max(abs(s) for s in sf_list) if sf_list else 0
        max_sf_idx = max(range(len(sf_list)), key=lambda i: abs(sf_list[i])) if sf_list else 0
        max_defl = max(abs(d) for d in defl_list) if defl_list else 0
        max_defl_idx = max(range(len(defl_list)), key=lambda i: abs(defl_list[i])) if defl_list else 0

        status = "OK"
        if max_defl > 50:
            status = f"Warning: deflection {max_defl:.0f}mm > 50mm"
        if not active_levels and exc_d > 3.0:
            status = f"Warning: cantilever at {exc_d:.1f}m (no anchors)"

        stage_results.append(StageResult(
            stage=stage,
            depths=depths_list,
            bending_moments=bm_list,
            shear_forces=sf_list,
            deflections=defl_list,
            active_pressures=active_p,
            passive_pressures=passive_p,
            net_pressures=net_factored,
            max_bm=max_bm,
            max_bm_depth=depths_list[max_bm_idx],
            max_sf=max_sf,
            max_sf_depth=depths_list[max_sf_idx],
            max_defl=max_defl,
            max_defl_depth=depths_list[max_defl_idx],
            anchor_reactions=anc_rxn,
            toe_reaction=toe_rxn,
            status=status,
        ))

    # ── Compute envelope ──
    # Use the node count from the first non-trivial stage
    ref_depths = None
    for sr in stage_results:
        if len(sr.depths) > 10:
            ref_depths = sr.depths
            break
    if ref_depths is None:
        ref_depths = stage_results[-1].depths

    n_pts = len(ref_depths)
    env_bm_max = [0.0] * n_pts
    env_bm_min = [0.0] * n_pts
    env_sf_max = [0.0] * n_pts
    env_defl_max = [0.0] * n_pts

    for sr in stage_results:
        if len(sr.bending_moments) != n_pts:
            # Interpolate to common depth array
            bm_interp = np.interp(ref_depths, sr.depths, sr.bending_moments).tolist()
            sf_interp = np.interp(ref_depths, sr.depths, sr.shear_forces).tolist()
            defl_interp = np.interp(ref_depths, sr.depths, sr.deflections).tolist()
        else:
            bm_interp = sr.bending_moments
            sf_interp = sr.shear_forces
            defl_interp = sr.deflections

        for i in range(n_pts):
            if abs(bm_interp[i]) > abs(env_bm_max[i]):
                env_bm_max[i] = bm_interp[i]
            env_bm_min[i] = min(env_bm_min[i], bm_interp[i])
            if abs(sf_interp[i]) > abs(env_sf_max[i]):
                env_sf_max[i] = sf_interp[i]
            if abs(defl_interp[i]) > abs(env_defl_max[i]):
                env_defl_max[i] = defl_interp[i]

    # ── Design values (peaks of envelope) ──
    design_bm = max(abs(m) for m in env_bm_max)
    design_bm_idx = max(range(n_pts), key=lambda i: abs(env_bm_max[i]))
    design_sf = max(abs(s) for s in env_sf_max)
    design_sf_idx = max(range(n_pts), key=lambda i: abs(env_sf_max[i]))
    design_defl = max(abs(d) for d in env_defl_max)
    design_defl_idx = max(range(n_pts), key=lambda i: abs(env_defl_max[i]))

    # Find which stage produced each peak
    def _find_governing_stage(value, attr):
        for sr in stage_results:
            if abs(getattr(sr, attr) - abs(value)) < 0.5:
                return sr.stage.stage_number
        return stage_results[-1].stage.stage_number

    design_bm_stage = _find_governing_stage(design_bm, 'max_bm')
    design_sf_stage = _find_governing_stage(design_sf, 'max_sf')
    design_defl_stage = _find_governing_stage(design_defl, 'max_defl')

    # ── Summary table ──
    summary = []
    for sr in stage_results:
        summary.append({
            'stage': sr.stage.stage_number,
            'description': sr.stage.description,
            'exc_depth': sr.stage.excavation_depth,
            'n_anchors': len(sr.stage.active_anchor_indices),
            'max_bm': sr.max_bm,
            'max_sf': sr.max_sf,
            'max_defl': sr.max_defl,
            'status': sr.status,
        })

    return StagedAnalysisResult(
        stages=stage_results,
        n_stages=len(stage_results),
        envelope_depths=ref_depths,
        envelope_bm_max=env_bm_max,
        envelope_bm_min=env_bm_min,
        envelope_sf_max=env_sf_max,
        envelope_defl_max=env_defl_max,
        design_bm=design_bm,
        design_bm_depth=ref_depths[design_bm_idx],
        design_bm_stage=design_bm_stage,
        design_sf=design_sf,
        design_sf_depth=ref_depths[design_sf_idx],
        design_sf_stage=design_sf_stage,
        design_defl=design_defl,
        design_defl_depth=ref_depths[design_defl_idx],
        design_defl_stage=design_defl_stage,
        summary=summary,
        wall_length=wall_toe_level,
        EI=EI,
        n_anchors=len(anchors),
    )
