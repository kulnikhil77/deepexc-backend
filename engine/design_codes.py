"""
Design Code Abstraction Layer

This is the critical architectural piece that keeps the analysis engine 
code-agnostic. Each national/international code implements the same interface.
Adding a new code (Eurocode 7, AASHTO, etc.) means writing a new class — 
the solver never changes.

Currently implemented:
- IS codes (IS 456, IS 9527, IS 14458, IS 1893, IS 800)

Planned:
- Eurocode 7 (DA-1, DA-2, DA-3)
- AASHTO LRFD
- ACI 318 / AISC 360
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


# ─────────────────────────────────────────────
# Load & resistance factor containers
# ─────────────────────────────────────────────

@dataclass
class LoadFactors:
    """Load factors for a specific load combination."""
    name: str
    dead_load: float = 1.0
    live_load: float = 1.0
    earth_pressure_active: float = 1.0
    earth_pressure_passive: float = 1.0
    water_pressure: float = 1.0
    surcharge: float = 1.0
    seismic: float = 1.0
    description: str = ""


@dataclass
class ResistanceFactors:
    """Resistance/capacity reduction factors."""
    name: str
    # Geotechnical
    passive_resistance: float = 1.0      # FOS or partial factor on passive
    bearing_capacity: float = 1.0
    sliding: float = 1.0
    basal_heave: float = 1.0
    # Structural
    bending: float = 1.0
    shear: float = 1.0
    axial: float = 1.0
    # Soil strength
    cohesion: float = 1.0
    friction: float = 1.0
    description: str = ""


@dataclass
class MaterialProperties:
    """Code-specific material properties and limits."""
    # Concrete
    fck: float = 0.0          # kPa, characteristic compressive strength
    fy: float = 0.0           # kPa, rebar yield strength
    gamma_c: float = 1.0      # partial safety factor for concrete
    gamma_s: float = 1.0      # partial safety factor for steel rebar
    # Structural steel
    fy_steel: float = 0.0     # kPa, structural steel yield strength
    gamma_steel: float = 1.0  # partial safety factor for structural steel
    # Derived
    Es_steel: float = 2.0e8   # kPa, elastic modulus of steel


@dataclass 
class SeismicParams:
    """Seismic design parameters."""
    zone_factor: float = 0.0          # Z
    importance_factor: float = 1.0    # I
    response_reduction: float = 1.0   # R
    soil_type: str = "medium"         # for site coefficient
    Ah: float = 0.0                   # horizontal seismic coefficient
    Av: float = 0.0                   # vertical seismic coefficient
    method: str = "mononobe_okabe"    # earth pressure method under seismic


# ─────────────────────────────────────────────
# Abstract base class — THE interface
# ─────────────────────────────────────────────

class DesignCode(ABC):
    """
    Abstract interface for all design codes.
    
    Every national code must implement these methods.
    The analysis engine calls ONLY these methods — 
    it never knows which code is being used.
    """

    @property
    @abstractmethod
    def code_name(self) -> str:
        """Full code name, e.g. 'IS 456:2000'"""
        pass

    @property
    @abstractmethod
    def code_country(self) -> str:
        """Country/region, e.g. 'India', 'Europe'"""
        pass

    # ── Load combinations ──

    @abstractmethod
    def get_load_combinations(self) -> List[LoadFactors]:
        """Return all applicable load combinations for excavation design."""
        pass

    @abstractmethod
    def get_resistance_factors(self, check_type: str) -> ResistanceFactors:
        """
        Return resistance factors for a specific check.
        check_type: 'wall_bending', 'wall_shear', 'embedment', 
                     'basal_heave', 'overall_stability', 'strut_design'
        """
        pass

    # ── Earth pressure ──

    @abstractmethod
    def get_fos_passive(self) -> float:
        """Factor of safety on passive resistance for embedment check."""
        pass

    @abstractmethod
    def get_fos_active(self) -> float:
        """Factor of safety on active pressure (usually 1.0)."""
        pass

    # ── Seismic earth pressure ──

    @abstractmethod
    def compute_seismic_Ka(self, phi_deg: float, delta_deg: float,
                           seismic: SeismicParams, beta_deg: float = 0.0) -> float:
        """
        Compute active earth pressure coefficient under seismic loading.
        Most codes use Mononobe-Okabe method.
        """
        pass

    @abstractmethod
    def compute_seismic_Kp(self, phi_deg: float, delta_deg: float,
                           seismic: SeismicParams, beta_deg: float = 0.0) -> float:
        """Compute passive earth pressure coefficient under seismic loading."""
        pass

    # ── Structural design checks ──

    @abstractmethod
    def check_rc_section(self, Mu: float, Vu: float, b: float, d: float,
                         material: MaterialProperties) -> Dict:
        """
        Check/design RC section (diaphragm wall).
        Returns dict with: Ast_required, Asv_required, Mu_capacity, Vu_capacity, 
                          utilization_bending, utilization_shear, status
        """
        pass

    @abstractmethod
    def check_steel_section(self, Mu: float, Vu: float, Pu: float,
                            section_props: Dict, material: MaterialProperties) -> Dict:
        """
        Check steel section (sheet pile, soldier pile, strut).
        Returns dict with utilization ratios and status.
        """
        pass

    # ── Stability checks ──

    @abstractmethod
    def get_fos_basal_heave(self) -> float:
        """Required FOS against basal heave."""
        pass

    @abstractmethod
    def get_fos_hydraulic_uplift(self) -> float:
        """Required FOS against hydraulic uplift."""
        pass

    @abstractmethod
    def get_fos_overall_stability(self) -> float:
        """Required FOS for overall/global stability."""
        pass

    # ── Reporting ──

    @abstractmethod
    def get_report_header(self) -> str:
        """Code-specific header text for design report."""
        pass

    @abstractmethod
    def get_references(self) -> List[str]:
        """List of code clause references used."""
        pass


# ─────────────────────────────────────────────
# IS Code Implementation
# ─────────────────────────────────────────────

class IS_Code(DesignCode):
    """
    Indian Standards implementation for deep excavation design.
    
    Codes covered:
    - IS 9527 (Part 1): 1981 — Design of sheet pile walls
    - IS 14458 (Part 1): 1998 — Design of retaining walls
    - IS 456: 2000 — RC design
    - IS 800: 2007 — Steel design (LSM)
    - IS 1893 (Part 1): 2016 — Seismic design
    - IS 1893 (Part 3): 2014 — Seismic for bridges & retaining walls
    """

    def __init__(self, seismic_zone: int = 3, importance_factor: float = 1.0,
                 soil_type: str = "medium"):
        """
        Args:
            seismic_zone: IS 1893 zone (2, 3, 4, or 5)
            importance_factor: IS 1893 importance factor
            soil_type: 'hard', 'medium', or 'soft' per IS 1893 Table 1
        """
        self._zone = seismic_zone
        self._I = importance_factor
        self._soil_type = soil_type

        # Zone factors per IS 1893:2016 Table 3
        self._zone_factors = {2: 0.10, 3: 0.16, 4: 0.24, 5: 0.36}

    @property
    def code_name(self) -> str:
        return "IS 9527/IS 456/IS 800/IS 1893"

    @property
    def code_country(self) -> str:
        return "India"

    # ── Load combinations (IS approach: working stress + limit state) ──

    def get_load_combinations(self) -> List[LoadFactors]:
        """
        IS codes for excavations primarily use working stress (FOS) approach.
        IS 456 uses limit state, so we provide both.
        
        For geotechnical checks: working stress with FOS on resistance
        For structural checks: IS 456 limit state combinations
        """
        combinations = [
            # Geotechnical (service) — IS 9527 approach
            LoadFactors(
                name="Service-1: DL + EP + Water",
                dead_load=1.0, live_load=0.0,
                earth_pressure_active=1.0, earth_pressure_passive=1.0,
                water_pressure=1.0, surcharge=1.0, seismic=0.0,
                description="Normal operating condition per IS 9527"
            ),
            LoadFactors(
                name="Service-2: DL + EP + Water + Surcharge",
                dead_load=1.0, live_load=1.0,
                earth_pressure_active=1.0, earth_pressure_passive=1.0,
                water_pressure=1.0, surcharge=1.0, seismic=0.0,
                description="With surcharge per IS 9527"
            ),
            # Seismic combination
            LoadFactors(
                name="Seismic: DL + EP + Water + 0.5LL + EQ",
                dead_load=1.0, live_load=0.5,
                earth_pressure_active=1.0, earth_pressure_passive=1.0,
                water_pressure=1.0, surcharge=0.5, seismic=1.0,
                description="Seismic per IS 1893"
            ),
            # Structural limit state — IS 456 Table 18
            LoadFactors(
                name="ULS-1: 1.5(DL + EP + Water)",
                dead_load=1.5, live_load=1.5,
                earth_pressure_active=1.5, earth_pressure_passive=1.0,
                water_pressure=1.5, surcharge=1.5, seismic=0.0,
                description="IS 456 Cl. 36.4 — Limit state of collapse"
            ),
            LoadFactors(
                name="ULS-2: 1.2(DL + EP + Water + EQ)",
                dead_load=1.2, live_load=1.2,
                earth_pressure_active=1.2, earth_pressure_passive=1.0,
                water_pressure=1.2, surcharge=1.2, seismic=1.2,
                description="IS 456 — With seismic"
            ),
            LoadFactors(
                name="ULS-3: 1.5(DL + EP) + 1.5EQ",
                dead_load=1.5, live_load=0.0,
                earth_pressure_active=1.5, earth_pressure_passive=1.0,
                water_pressure=1.5, surcharge=0.0, seismic=1.5,
                description="IS 456 — Seismic dominant"
            ),
        ]
        return combinations

    def get_resistance_factors(self, check_type: str) -> ResistanceFactors:
        """Resistance factors per IS codes."""
        factors = {
            "wall_bending": ResistanceFactors(
                name="Wall Bending — IS 456",
                bending=1.0,  # already factored in limit state design
                gamma_note="Partial factors: γ_c=1.5, γ_s=1.15 built into IS 456 design"
            ) if False else ResistanceFactors(
                name="Wall Bending — IS 456",
                bending=1.0,
            ),
            "wall_shear": ResistanceFactors(
                name="Wall Shear — IS 456",
                shear=1.0,
            ),
            "embedment": ResistanceFactors(
                name="Embedment — IS 9527",
                passive_resistance=2.0,  # FOS = 2.0 on net passive (IS 9527)
                description="IS 9527 Cl. 9.2: FOS ≥ 2.0 on passive for cantilever, ≥ 1.5 for anchored"
            ),
            "embedment_anchored": ResistanceFactors(
                name="Embedment Anchored — IS 9527",
                passive_resistance=1.5,
                description="IS 9527: FOS ≥ 1.5 for anchored/strutted walls"
            ),
            "basal_heave": ResistanceFactors(
                name="Basal Heave",
                basal_heave=1.5,
                description="FOS ≥ 1.5 against basal heave"
            ),
            "overall_stability": ResistanceFactors(
                name="Overall Stability",
                sliding=1.5,
                description="FOS ≥ 1.5 for overall stability (Bishop's method)"
            ),
            "strut_design": ResistanceFactors(
                name="Strut Design — IS 800",
                axial=1.0,  # IS 800 LSM handles via partial factors
                bending=1.0,
                description="IS 800:2007 limit state design for struts"
            ),
            "hydraulic_uplift": ResistanceFactors(
                name="Hydraulic Uplift",
                passive_resistance=1.5,
                description="FOS ≥ 1.5 against hydraulic uplift"
            ),
        }
        return factors.get(check_type, ResistanceFactors(name=f"Default — {check_type}"))

    # ── Earth pressure FOS ──

    def get_fos_passive(self) -> float:
        """IS 9527: FOS = 2.0 on passive for cantilever walls."""
        return 2.0

    def get_fos_active(self) -> float:
        return 1.0

    # ── Seismic: Mononobe-Okabe per IS 1893 ──

    def _get_Ah(self, seismic: Optional[SeismicParams] = None) -> float:
        """Compute horizontal seismic coefficient Ah per IS 1893:2016."""
        if seismic and seismic.Ah > 0:
            return seismic.Ah

        Z = self._zone_factors.get(self._zone, 0.16)
        I = self._I
        R = 1.0  # For retaining walls, R=1 (no ductility)

        # Sa/g depends on soil type and period — for retaining walls, 
        # IS 1893 Part 3 uses simplified approach
        # For rigid walls: Sa/g = 2.5 (short period)
        Sa_g = 2.5

        Ah = (Z / 2) * (I / R) * Sa_g
        return Ah

    def compute_seismic_Ka(self, phi_deg: float, delta_deg: float,
                           seismic: SeismicParams = None,
                           beta_deg: float = 0.0) -> float:
        """
        Mononobe-Okabe active coefficient.
        
        Ka_dyn = [1/(cos²(ψ) × cos²(α) × cos(δ+α+ψ))] × 
                 [sin²(α+φ-ψ) / (1 + √(sin(φ+δ)×sin(φ-β-ψ)/(cos(δ+α+ψ)×cos(β-α))))²]
        
        where ψ = atan(Ah/(1±Av))
        
        For vertical wall (α=90°) and horizontal backfill (β=0°), this simplifies.
        
        Ref: IS 1893 (Part 3): 2014, Cl. 8
        """
        Ah = self._get_Ah(seismic)
        Av = seismic.Av if seismic else 0.0

        # Seismic inertia angle
        psi = math.atan(Ah / (1 - Av)) if (1 - Av) != 0 else 0

        phi = math.radians(phi_deg)
        delta = math.radians(delta_deg)
        alpha = math.pi / 2  # vertical wall
        beta = math.radians(beta_deg)

        # Check: φ - β - ψ must be > 0 for M-O to be valid
        if phi - beta - psi <= 0:
            import warnings
            warnings.warn(
                f"M-O not valid: φ-β-ψ = {math.degrees(phi-beta-psi):.1f}° ≤ 0. "
                f"Use trial wedge method or reduce seismic coefficient."
            )
            # Fallback: return static Ka amplified by (1 + Ah)
            try:
                from engine.coefficients import ka_rankine
            except ImportError:
                from coefficients import ka_rankine
            return ka_rankine(phi_deg) * (1 + Ah)

        num = math.sin(alpha + phi - psi) ** 2

        sqrt_num = math.sin(phi + delta) * math.sin(phi - beta - psi)
        sqrt_den = math.cos(delta + alpha + psi) * math.cos(beta - alpha)

        if sqrt_den <= 0:
            raise ValueError("M-O Ka: denominator term ≤ 0")

        sqrt_term = math.sqrt(sqrt_num / sqrt_den)
        bracket = (1 + sqrt_term) ** 2

        den = (math.cos(psi) * math.cos(alpha) ** 2 *
               math.cos(delta + alpha + psi) * bracket)

        Ka_dyn = num / den
        return Ka_dyn

    def compute_seismic_Kp(self, phi_deg: float, delta_deg: float,
                           seismic: SeismicParams = None,
                           beta_deg: float = 0.0) -> float:
        """
        Mononobe-Okabe passive coefficient (conservative: ignore seismic on passive side).
        
        IS 1893 Part 3 recommends using static Kp for the passive side
        as seismic effect reduces passive resistance. Being conservative,
        we use static Kp (already computed in the pressure module).
        
        Some references use M-O for passive too, but the reduction is 
        unconservative for the resisting side — better to use static.
        """
        try:
            from engine.coefficients import kp_rankine
        except ImportError:
            from coefficients import kp_rankine
        return kp_rankine(phi_deg)

    # ── RC design per IS 456:2000 ──

    def check_rc_section(self, Mu: float, Vu: float, b: float, d: float,
                         material: MaterialProperties = None) -> Dict:
        """
        Design/check RC section per IS 456:2000.
        
        Args:
            Mu: Factored bending moment (kN·m per m run)
            Vu: Factored shear force (kN per m run)
            b: Width of section (mm) — typically 1000 for per-m-run
            d: Effective depth (mm)
            material: MaterialProperties (defaults to M30/Fe500)
        
        Returns:
            Dict with Ast_required, Asv_required, status, etc.
        """
        if material is None:
            material = MaterialProperties(
                fck=30000,       # 30 MPa = 30000 kPa → but we work in N/mm² internally
                fy=500000,       # 500 MPa
                gamma_c=1.5,
                gamma_s=1.15,
            )

        # Convert to N/mm² for IS 456 calculations
        fck = material.fck / 1000 if material.fck > 1000 else material.fck  # handle kPa or MPa input
        fy = material.fy / 1000 if material.fy > 1000 else material.fy

        # IS 456 Cl. 38.1 — Limiting moment of resistance
        # xu_max/d values per IS 456 Table
        xu_max_ratio = {250: 0.53, 415: 0.48, 500: 0.46}.get(int(fy), 0.46)
        xu_max = xu_max_ratio * d

        Mu_lim = 0.36 * fck * b * xu_max * (d - 0.42 * xu_max) * 1e-6  # kN·m

        # Required Ast (singly reinforced)
        Mu_Nmm = abs(Mu) * 1e6  # Convert kN·m to N·mm
        
        # From IS 456: Mu = 0.87 × fy × Ast × (d - 0.42×xu)
        # Using quadratic: Mu = 0.87×fy×Ast×d×(1 - Ast×fy/(b×d×fck))
        # Ast = (b×d)/(2×fy) × (fck/0.87) × (1 - √(1 - 4.598×Mu/(fck×b×d²)))

        if b * d * d > 0:
            term = 4.598 * Mu_Nmm / (fck * b * d * d)
            if term > 1.0:
                # Section is inadequate — needs compression steel or larger section
                Ast_req = -1  # flag
                status_bending = "FAIL — section inadequate, increase depth"
            elif term < 0:
                Ast_req = 0
                status_bending = "OK — no tension reinforcement needed"
            else:
                Ast_req = (0.5 * fck * b * d / fy) * (1 - math.sqrt(1 - term))  # mm²
                status_bending = "OK"
        else:
            Ast_req = 0
            status_bending = "ERROR — invalid dimensions"

        # Minimum steel: IS 456 Cl. 26.5.1.1
        Ast_min = 0.12 * b * d / 100 if fy >= 415 else 0.15 * b * d / 100  # mm²

        # Shear check: IS 456 Cl. 40
        Vu_N = abs(Vu) * 1000  # kN to N
        tau_v = Vu_N / (b * d) if (b * d) > 0 else 0  # N/mm²

        # Permissible shear stress (Table 19, IS 456) — simplified for pt ≈ 0.5%
        tau_c = 0.48  # conservative for M30, pt ≈ 0.25-0.5%

        # Max shear stress IS 456 Table 20
        tau_c_max = 3.5 if fck >= 30 else 2.8

        if tau_v <= tau_c:
            Asv_req = 0.0
            status_shear = "OK — no shear reinforcement needed"
        elif tau_v <= tau_c_max:
            # Shear reinforcement: Vus = Vu - τc×b×d
            Vus = Vu_N - tau_c * b * d
            # Asv/sv = Vus / (0.87 × fy × d) — assuming vertical stirrups
            # For sv = 200mm (typical):
            sv = 200  # mm
            Asv_req = Vus * sv / (0.87 * fy * d)  # mm² per pair of legs
            status_shear = f"OK — stirrups needed, Asv = {Asv_req:.0f} mm² @ {sv}mm c/c"
        else:
            Asv_req = -1
            status_shear = "FAIL — increase section depth"

        return {
            "Mu": Mu,
            "Vu": Vu,
            "Mu_lim": Mu_lim,
            "Ast_required": max(Ast_req, Ast_min) if Ast_req >= 0 else Ast_req,
            "Ast_min": Ast_min,
            "Asv_required": max(Asv_req, 0),
            "tau_v": tau_v,
            "tau_c": tau_c,
            "utilization_bending": abs(Mu) / Mu_lim if Mu_lim > 0 else 999,
            "utilization_shear": tau_v / tau_c_max if tau_c_max > 0 else 999,
            "status_bending": status_bending,
            "status_shear": status_shear,
            "code_ref": "IS 456:2000, Cl. 38.1, Cl. 40"
        }

    def check_steel_section(self, Mu: float, Vu: float, Pu: float,
                            section_props: Dict, material: MaterialProperties = None) -> Dict:
        """
        Check steel section per IS 800:2007 (Limit State Method).
        
        Args:
            Mu: Factored bending moment (kN·m)
            Vu: Factored shear force (kN)
            Pu: Factored axial force (kN) — compression for struts
            section_props: Dict with 'Zp' (plastic modulus mm³), 'A' (area mm²),
                          'tw' (web thickness mm), 'd' (depth mm), 'Iz' (mm⁴)
            material: MaterialProperties
        """
        if material is None:
            material = MaterialProperties(fy_steel=250000, gamma_steel=1.10)  # Fe250, γ=1.10

        fy = material.fy_steel / 1000 if material.fy_steel > 1000 else material.fy_steel  # MPa
        gamma_m0 = material.gamma_steel if material.gamma_steel > 1 else 1.10

        Zp = section_props.get('Zp', 0)     # mm³
        A = section_props.get('A', 0)        # mm²
        tw = section_props.get('tw', 0)      # mm
        d_sec = section_props.get('d', 0)    # mm

        # Bending capacity: Md = Zp × fy / γ_m0
        Md = Zp * fy / (gamma_m0 * 1e6) if Zp > 0 else 0  # kN·m

        # Shear capacity: Vd = (fy/√3) × Av / γ_m0
        Av = d_sec * tw  # shear area (conservative: web area)
        Vd = (fy / math.sqrt(3)) * Av / (gamma_m0 * 1e3) if Av > 0 else 0  # kN

        # Axial capacity (compression — strut): Pd = A × fy / γ_m0 (stocky, no buckling)
        # Full buckling check needs slenderness — placeholder
        Pd = A * fy / (gamma_m0 * 1e3) if A > 0 else 0  # kN (squash load)

        util_bending = abs(Mu) / Md if Md > 0 else 999
        util_shear = abs(Vu) / Vd if Vd > 0 else 999
        util_axial = abs(Pu) / Pd if Pd > 0 else 999

        # Combined check (simplified interaction)
        # IS 800 Cl. 9.3.1: (P/Pd) + (M/Md) ≤ 1.0
        interaction = util_axial + util_bending if Pu != 0 else util_bending

        return {
            "Md": Md, "Vd": Vd, "Pd": Pd,
            "utilization_bending": util_bending,
            "utilization_shear": util_shear,
            "utilization_axial": util_axial,
            "interaction_ratio": interaction,
            "status": "OK" if interaction <= 1.0 and util_shear <= 1.0 else "FAIL",
            "code_ref": "IS 800:2007, Cl. 8.2, Cl. 9.3"
        }

    # ── Stability FOS requirements ──

    def get_fos_basal_heave(self) -> float:
        return 1.5

    def get_fos_hydraulic_uplift(self) -> float:
        return 1.5

    def get_fos_overall_stability(self) -> float:
        return 1.5

    # ── Reporting ──

    def get_report_header(self) -> str:
        return (
            "DESIGN OF DEEP EXCAVATION SUPPORT SYSTEM\n"
            "As per IS 9527 (Part 1):1981, IS 14458 (Part 1):1998,\n"
            "IS 456:2000, IS 800:2007, IS 1893:2016\n"
            f"Seismic Zone: {self._zone} (Z = {self._zone_factors.get(self._zone, 0.16)})"
        )

    def get_references(self) -> List[str]:
        return [
            "IS 9527 (Part 1):1981 — Code of practice for design and construction of port and harbour structures: Sheet pile walls",
            "IS 14458 (Part 1):1998 — Retaining wall for hill area: Guidelines — Selection of type of wall",
            "IS 456:2000 — Plain and reinforced concrete: Code of practice",
            "IS 800:2007 — General construction in steel: Code of practice",
            "IS 1893 (Part 1):2016 — Criteria for earthquake resistant design of structures",
            "IS 1893 (Part 3):2014 — Bridges and retaining walls",
            "IS 2911 (Part 1):2010 — Design and construction of pile foundations",
        ]


# ─────────────────────────────────────────────
# Default material libraries (Indian sections)
# ─────────────────────────────────────────────

def get_default_concrete(grade: str = "M30") -> MaterialProperties:
    """Standard concrete grades per IS 456."""
    grades = {
        "M20": MaterialProperties(fck=20, fy=500, gamma_c=1.5, gamma_s=1.15),
        "M25": MaterialProperties(fck=25, fy=500, gamma_c=1.5, gamma_s=1.15),
        "M30": MaterialProperties(fck=30, fy=500, gamma_c=1.5, gamma_s=1.15),
        "M35": MaterialProperties(fck=35, fy=500, gamma_c=1.5, gamma_s=1.15),
        "M40": MaterialProperties(fck=40, fy=500, gamma_c=1.5, gamma_s=1.15),
    }
    return grades.get(grade, grades["M30"])


def get_default_steel(grade: str = "Fe250") -> MaterialProperties:
    """Standard structural steel grades per IS 2062 / IS 800."""
    grades = {
        "Fe250": MaterialProperties(fy_steel=250, gamma_steel=1.10),
        "Fe350": MaterialProperties(fy_steel=350, gamma_steel=1.10),
        "Fe410": MaterialProperties(fy_steel=410, gamma_steel=1.10),
        "Fe450": MaterialProperties(fy_steel=450, gamma_steel=1.10),
    }
    return grades.get(grade, grades["Fe250"])
