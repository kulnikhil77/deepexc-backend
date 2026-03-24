"""
Data models for deep excavation design system.
All units: kN, m, kPa, degrees
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import math


class SoilType(Enum):
    SAND = "sand"
    CLAY = "clay"
    SILT = "silt"
    MIXED = "mixed"  # c-phi soil


class WallType(Enum):
    SHEET_PILE = "sheet_pile"
    DIAPHRAGM = "diaphragm"
    SOLDIER_PILE = "soldier_pile"


class SurchargeType(Enum):
    UNIFORM = "uniform"
    LINE = "line"
    STRIP = "strip"


class PressureTheory(Enum):
    RANKINE = "rankine"
    COULOMB = "coulomb"


@dataclass
class SoilLayer:
    """Single soil layer with properties."""
    name: str
    thickness: float           # m
    gamma: float               # kN/m3 (bulk unit weight above WT)
    gamma_sat: float           # kN/m3 (saturated unit weight)
    c_eff: float               # kPa (effective cohesion, c')
    phi_eff: float             # degrees (effective friction angle, φ')
    c_u: float = 0.0           # kPa (undrained shear strength, for clay)
    K0: float = 0.5            # at-rest earth pressure coefficient
    Es: float = 0.0            # kPa (elastic modulus for spring stiffness)
    delta: float = 0.0         # degrees (wall friction angle)
    soil_type: SoilType = SoilType.MIXED

    def __post_init__(self):
        # Default delta as fraction of phi if not specified
        if self.delta == 0.0 and self.phi_eff > 0:
            self.delta = self.phi_eff * 2.0 / 3.0  # common assumption: δ = 2φ/3

        # Default K0 from Jaky's formula if not explicitly set
        if self.K0 == 0.5 and self.phi_eff > 0:
            self.K0 = 1.0 - math.sin(math.radians(self.phi_eff))

    @property
    def gamma_sub(self) -> float:
        """Submerged unit weight."""
        return self.gamma_sat - 9.81  # γ_w = 9.81 kN/m3


@dataclass
class Surcharge:
    """Surcharge loading on ground surface behind wall."""
    surcharge_type: SurchargeType
    magnitude: float           # kPa for uniform/strip, kN/m for line load
    offset: float = 0.0        # m, distance from wall face
    width: float = 0.0         # m, for strip load only

    def __post_init__(self):
        if self.surcharge_type == SurchargeType.STRIP and self.width <= 0:
            raise ValueError("Strip surcharge requires width > 0")


@dataclass
class WaterTable:
    """Water table definition."""
    depth_behind_wall: float     # m below GL on retained side
    depth_in_excavation: float = None  # m below GL on excavation side (None = same as excavation level)
    gamma_w: float = 9.81       # kN/m3

    def __post_init__(self):
        if self.depth_in_excavation is None:
            self.depth_in_excavation = self.depth_behind_wall


@dataclass
class ExcavationStage:
    """Single excavation stage."""
    stage_number: int
    excavation_depth: float      # m below GL
    strut_level: Optional[float] = None  # m below GL where strut is installed (None = no strut this stage)
    description: str = ""


@dataclass
class ProjectInput:
    """Complete project input definition."""
    name: str
    soil_layers: List[SoilLayer]
    water_table: WaterTable
    excavation_depth: float       # m, final excavation depth
    wall_type: WallType = WallType.SHEET_PILE
    embedment_depth: float = 0.0  # m, trial embedment below excavation
    surcharges: List[Surcharge] = field(default_factory=list)
    stages: List[ExcavationStage] = field(default_factory=list)
    pressure_theory: PressureTheory = PressureTheory.RANKINE
    dz: float = 0.1              # m, depth increment for calculations

    @property
    def total_wall_height(self) -> float:
        """Total wall height = excavation depth + embedment."""
        return self.excavation_depth + self.embedment_depth

    @property
    def total_soil_depth(self) -> float:
        """Sum of all soil layer thicknesses."""
        return sum(layer.thickness for layer in self.soil_layers)

    def validate(self):
        """Basic input validation."""
        errors = []

        if self.excavation_depth <= 0:
            errors.append("Excavation depth must be positive")

        if self.total_soil_depth < self.total_wall_height:
            errors.append(
                f"Soil profile depth ({self.total_soil_depth:.1f}m) is less than "
                f"total wall height ({self.total_wall_height:.1f}m). "
                f"Extend the deepest soil layer."
            )

        for i, layer in enumerate(self.soil_layers):
            if layer.thickness <= 0:
                errors.append(f"Layer {i+1} ({layer.name}): thickness must be positive")
            if layer.gamma <= 0:
                errors.append(f"Layer {i+1} ({layer.name}): unit weight must be positive")
            if layer.phi_eff < 0 or layer.phi_eff > 50:
                errors.append(f"Layer {i+1} ({layer.name}): φ' = {layer.phi_eff}° seems unreasonable")

        if errors:
            raise ValueError("Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True
