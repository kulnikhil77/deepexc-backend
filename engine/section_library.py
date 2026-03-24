"""
Module 6: Sheet Pile Section Library
=====================================

Complete database of sheet pile sections with:
- Z-profiles (LARSSEN series, AZ series)
- U-profiles (PU series, AU series)
- Hat-type / cold-formed (light duty)
- Flat web sections (combined walls)
- Steel grades: S240GP, S270GP, S355GP, Fe410, Fe540 (IS/EN)
- Stress checks per IS 800:2007 (LSM) and working stress
- Utilization ratios: bending, shear, combined, interlock
- Auto-selection: lightest adequate section for given forces

References:
- IS 800:2007 (General construction in steel - LSM)
- IS 14458 (Part 1):1998 (Design and construction of sheet pile walls)
- EN 10248:1996 (Hot rolled sheet piling)
- ArcelorMittal Sheet Pile Catalogue
- JFE Steel Sheet Pile Catalogue


"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import math


# ══════════════════════════════════════════════════════════════
# ENUMERATIONS
# ══════════════════════════════════════════════════════════════

class ProfileType(Enum):
    """Sheet pile cross-section shape."""
    Z = "Z-profile"          # LARSSEN, AZ — symmetric, single interlock per pair
    U = "U-profile"          # PU, AU — symmetric U, interlocks at neutral axis
    HAT = "Hat-type"         # Cold-formed trapezoidal
    FLAT = "Flat-web"        # For combined walls (HZ + infill)
    TUBULAR = "Tubular"      # Pipe piles (combined walls)
    SL = "S/L-type"          # Pennar S/L light profiles
    TRENCH = "Trench"        # Pennar PRG trench piles


class SteelGrade(Enum):
    """Steel grades commonly used for sheet piles."""
    # EN grades (European, most common globally)
    S240GP = "S240GP"    # fy = 240 MPa (mild, general purpose)
    S270GP = "S270GP"    # fy = 270 MPa
    S355GP = "S355GP"    # fy = 355 MPa (high strength, most common)
    S390GP = "S390GP"    # fy = 390 MPa
    S430GP = "S430GP"    # fy = 430 MPa (extra high strength)
    # IS grades (Indian)
    Fe410 = "Fe410"      # fy = 250 MPa (IS 2062 / IS 800 Table 2)
    Fe540 = "Fe540"      # fy = 410 MPa (IS 2062)
    # JIS grades (Japanese, common in India)
    SY295 = "SY295"      # fy = 295 MPa
    SY390 = "SY390"      # fy = 390 MPa
    # Pennar grades (Indian cold-formed)
    Fe360 = "Fe360"      # fy = 235 MPa (Pennar brochure)
    Fe510 = "Fe510"      # fy = 345 MPa (Pennar brochure)
    BSK46 = "BSK46"      # fy = 380 MPa (Pennar brochure)


# Yield stress lookup (MPa)
STEEL_FY: Dict[SteelGrade, float] = {
    SteelGrade.S240GP: 240,
    SteelGrade.S270GP: 270,
    SteelGrade.S355GP: 355,
    SteelGrade.S390GP: 390,
    SteelGrade.S430GP: 430,
    SteelGrade.Fe410: 250,
    SteelGrade.Fe540: 410,
    SteelGrade.SY295: 295,
    SteelGrade.SY390: 390,
    SteelGrade.Fe360: 235,
    SteelGrade.Fe510: 345,
    SteelGrade.BSK46: 380,
}

# Ultimate stress lookup (MPa)
STEEL_FU: Dict[SteelGrade, float] = {
    SteelGrade.S240GP: 370,
    SteelGrade.S270GP: 410,
    SteelGrade.S355GP: 510,
    SteelGrade.S390GP: 540,
    SteelGrade.S430GP: 570,
    SteelGrade.Fe410: 410,
    SteelGrade.Fe540: 540,
    SteelGrade.SY295: 490,
    SteelGrade.SY390: 540,
    SteelGrade.Fe360: 410,
    SteelGrade.Fe510: 540,
    SteelGrade.BSK46: 560,
}

# Elastic modulus (MPa)
E_STEEL = 200_000  # MPa (2.0 x 10^5)


# ══════════════════════════════════════════════════════════════
# SECTION DATA MODEL
# ══════════════════════════════════════════════════════════════

@dataclass
class SheetPileSection:
    """Complete properties of a sheet pile section (per m of wall)."""
    name: str                     # e.g. "LARSSEN 4", "AZ 18"
    profile_type: ProfileType
    manufacturer: str             # ArcelorMittal, JFE, SAIL etc.

    # ── Geometric properties (per m of wall run) ──
    width: float                  # mm, system width (single pile or pair)
    height: float                 # mm, overall depth of section
    thickness_web: float          # mm, web (or minimum) thickness
    thickness_flange: float       # mm, flange thickness
    weight: float                 # kg/m² of wall (per m of wall, not per pile)

    # ── Section properties (per m of wall) ──
    area: float                   # cm²/m
    moment_of_inertia: float      # cm⁴/m  (I)
    elastic_modulus: float        # cm³/m  (Ze = I/y)
    plastic_modulus: float        # cm³/m  (Zp)

    # ── Interlock properties ──
    interlock_strength: float = 0.0   # kN/m (interlock shear capacity)
    coating: str = "bare"             # bare, zinc, epoxy etc.

    # ── Derived ──
    section_class: int = 0        # IS 800 section classification (1=plastic, 2=compact, 3=semi-compact, 4=slender)

    def __post_init__(self):
        """Compute derived values."""
        if self.section_class == 0:
            # Classify per IS 800 Table 2 (simplified for sheet piles)
            # Most hot-rolled sheet piles are Class 2 (compact) or better
            if self.plastic_modulus > 0 and self.elastic_modulus > 0:
                shape_factor = self.plastic_modulus / self.elastic_modulus
                if shape_factor >= 1.15:
                    self.section_class = 1  # Plastic
                elif shape_factor >= 1.0:
                    self.section_class = 2  # Compact
                else:
                    self.section_class = 3  # Semi-compact

    @property
    def EI_per_m(self) -> float:
        """Flexural rigidity per m of wall (kN·m²/m)."""
        # I in cm⁴/m → m⁴/m: ÷ 10^8
        # E in MPa → kN/m²: × 10^3
        return (E_STEEL * 1e3) * (self.moment_of_inertia / 1e8)

    @property
    def depth_m(self) -> float:
        """Section depth in meters."""
        return self.height / 1000


# ══════════════════════════════════════════════════════════════
# SECTION DATABASE
# ══════════════════════════════════════════════════════════════

def _build_database() -> List[SheetPileSection]:
    """
    Build the complete section database.

    Sources:
    - ArcelorMittal Piling Handbook (2016)
    - JFE Steel Sheet Pile Catalogue
    - IS 14458 (Part 1) Table 1
    - Manufacturer technical data sheets

    All properties are PER METRE OF WALL.
    Elastic modulus Ze = I / (h/2), Plastic modulus Zp ≈ 1.12 × Ze for Z-profiles.
    """
    db = []

    # ══════════════════════════════════════
    # Z-PROFILES (LARSSEN series)
    # ══════════════════════════════════════
    # Most common in India. Driven as pairs (S-shaped).
    # Source: ArcelorMittal / thyssenkrupp catalogue

    larssen = [
        # (name, width_mm, height_mm, tw, tf, weight_kg/m2, area_cm2/m, I_cm4/m, Ze_cm3/m, Zp_cm3/m, interlock_kN/m)
        ("LARSSEN 600",   600, 130,  6.4,  8.5,  57,  72.5,  3900,   600,   672,  3000),
        ("LARSSEN 601",   600, 150,  7.0,  9.0,  68,  86.5,  5600,   747,   836,  3000),
        ("LARSSEN 602",   600, 170,  7.5,  9.5,  79, 101.0,  7800,   918,  1028,  3000),
        ("LARSSEN 2",     600, 200,  8.0, 10.0,  91, 116.0, 13400,  1340,  1501,  3500),
        ("LARSSEN 22",    600, 226,  8.5, 10.5, 100, 128.0, 17600,  1557,  1744,  3500),
        ("LARSSEN 23",    600, 260,  9.0, 11.5, 113, 144.0, 24000,  1846,  2068,  3500),
        ("LARSSEN 4",     600, 310, 10.0, 13.0, 139, 177.0, 35200,  2270,  2542,  4000),
        ("LARSSEN 43",    600, 340, 11.0, 14.0, 154, 196.0, 42500,  2500,  2800,  4000),
        ("LARSSEN 6",     600, 400, 12.5, 15.0, 182, 232.0, 71400,  3570,  3998,  4500),
        ("LARSSEN 6S",    600, 420, 13.0, 15.5, 196, 250.0, 82000,  3905,  4374,  4500),
    ]
    for name, w, h, tw, tf, wt, A, I, Ze, Zp, IL in larssen:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.Z, manufacturer="ArcelorMittal",
            width=w, height=h, thickness_web=tw, thickness_flange=tf,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # Z-PROFILES (AZ series — ArcelorMittal)
    # ══════════════════════════════════════
    # Modern high-performance Z-piles

    az = [
        # (name, width_mm, height_mm, tw, tf, weight, area, I, Ze, Zp, IL)
        ("AZ 12",     670, 302,  8.5,  8.5,  67,  84.9,  13200,   874,   979,  3500),
        ("AZ 13",     670, 304,  9.5,  9.5,  76,  96.7,  15300,  1005,  1126,  3500),
        ("AZ 14",     670, 307, 10.0, 10.0,  81, 102.8,  17000,  1107,  1240,  3500),
        ("AZ 17",     630, 378,  8.5,  8.5,  76,  96.7,  22100,  1170,  1310,  4000),
        ("AZ 18",     630, 380,  9.5,  9.5,  85, 108.5,  25000,  1316,  1474,  4000),
        ("AZ 18-700", 700, 370,  8.5,  8.5,  73,  93.0,  22500,  1216,  1362,  4000),
        ("AZ 19-700", 700, 372,  9.5,  9.5,  81, 103.5,  25400,  1366,  1530,  4000),
        ("AZ 20",     630, 382, 10.5, 10.5,  95, 120.0,  28000,  1466,  1642,  4000),
        ("AZ 24",     630, 411, 11.2, 11.2, 108, 137.0,  37200,  1810,  2027,  4500),
        ("AZ 26",     630, 427, 12.0, 12.0, 118, 150.0,  43600,  2042,  2287,  4500),
        ("AZ 28",     630, 441, 13.0, 13.0, 130, 165.0,  50100,  2272,  2545,  4500),
        ("AZ 36",     630, 460, 14.0, 15.0, 155, 197.0,  63000,  2739,  3068,  5000),
        ("AZ 40",     630, 474, 15.0, 16.0, 170, 216.0,  73500,  3101,  3473,  5000),
        ("AZ 46",     580, 481, 16.0, 17.0, 186, 237.0,  80900,  3363,  3767,  5500),
        ("AZ 48",     580, 497, 17.0, 18.5, 204, 260.0,  93000,  3743,  4192,  5500),
        ("AZ 50",     580, 500, 18.0, 19.0, 215, 274.0,  98500,  3940,  4413,  5500),
    ]
    for name, w, h, tw, tf, wt, A, I, Ze, Zp, IL in az:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.Z, manufacturer="ArcelorMittal",
            width=w, height=h, thickness_web=tw, thickness_flange=tf,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # U-PROFILES (PU series)
    # ══════════════════════════════════════
    # Symmetric U-shape, interlocks at neutral axis.
    # Lower efficiency than Z (interlock slip), but good for combined walls.
    # PU values are for "free interlocks" — wall properties.

    pu = [
        # (name, width, height, tw, tf, weight, area, I, Ze, Zp, IL)
        ("PU 6",     600, 216,  6.4,  7.6,  62,  79.5,  6060,   561,   628,  2500),
        ("PU 8",     600, 240,  7.0,  8.2,  72,  92.0,  8600,   717,   803,  2500),
        ("PU 12",    600, 305,  8.0,  8.0,  85, 108.0, 16100,  1055,  1182,  3000),
        ("PU 12-10", 750, 303,  8.5,  8.5,  80, 102.0, 16400,  1082,  1212,  3000),
        ("PU 18",    600, 380, 10.0, 10.0, 107, 137.0, 28500,  1500,  1680,  3500),
        ("PU 18-1",  600, 382, 10.0, 10.2, 110, 140.0, 29400,  1539,  1724,  3500),
        ("PU 22",    600, 408, 10.5, 12.0, 128, 163.0, 38600,  1892,  2119,  4000),
        ("PU 22-1",  600, 410, 11.0, 12.0, 132, 168.0, 40000,  1951,  2185,  4000),
        ("PU 25",    600, 432, 11.0, 13.0, 142, 181.0, 48400,  2241,  2510,  4000),
        ("PU 28",    600, 454, 11.5, 14.2, 157, 200.0, 57500,  2533,  2837,  4500),
        ("PU 32",    600, 476, 12.0, 15.5, 173, 220.0, 68400,  2874,  3219,  4500),
    ]
    for name, w, h, tw, tf, wt, A, I, Ze, Zp, IL in pu:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.U, manufacturer="ArcelorMittal",
            width=w, height=h, thickness_web=tw, thickness_flange=tf,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # U-PROFILES (AU series — heavier)
    # ══════════════════════════════════════
    au = [
        ("AU 14",    750, 408,  9.0,  9.5,  80, 102.0, 21600,  1059,  1186,  3000),
        ("AU 16",    750, 410, 10.0, 10.0,  90, 115.0, 24200,  1180,  1322,  3500),
        ("AU 18",    750, 441, 10.0, 10.5, 102, 130.0, 31900,  1447,  1621,  3500),
        ("AU 20",    750, 443, 11.0, 11.0, 112, 143.0, 35200,  1589,  1780,  3500),
        ("AU 23",    750, 446, 12.0, 12.0, 125, 159.0, 40800,  1830,  2050,  4000),
        ("AU 25",    750, 450, 13.0, 12.5, 135, 172.0, 45200,  2009,  2250,  4000),
        ("AU 26",    750, 467, 13.5, 13.0, 147, 187.0, 52800,  2262,  2533,  4500),
    ]
    for name, w, h, tw, tf, wt, A, I, Ze, Zp, IL in au:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.U, manufacturer="ArcelorMittal",
            width=w, height=h, thickness_web=tw, thickness_flange=tf,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # HAT-TYPE / COLD-FORMED
    # ══════════════════════════════════════
    # Light-duty cold-formed trapezoidal. Used for shallow excavations,
    # temporary works, cofferdams. Lower section modulus but cheap.

    hat = [
        # (name, width, height, t, weight, area, I, Ze, Zp, IL)
        ("SKS-II",     400,  90, 5.0, 35,  44.5,  1120,  249,  279, 1500),
        ("SKS-III",    400, 120, 5.5, 42,  53.5,  2100,  350,  392, 1500),
        ("SKS-IV",     400, 150, 6.0, 50,  64.0,  3600,  480,  538, 1800),
        ("SKS-V",      400, 180, 6.5, 60,  76.0,  5700,  633,  709, 1800),
        ("SKS-VI",     400, 200, 7.0, 68,  87.0,  7500,  750,  840, 2000),
        ("FSP-IA",     400, 100, 5.0, 36,  46.0,  1400,  280,  314, 1500),
        ("FSP-IIA",    400, 125, 5.5, 44,  56.0,  2600,  416,  466, 1500),
        ("FSP-IIIA",   400, 150, 6.0, 52,  66.0,  4200,  560,  627, 1800),
        ("FSP-IVA",    400, 175, 6.5, 62,  79.0,  6200,  709,  794, 1800),
    ]
    for name, w, h, t, wt, A, I, Ze, Zp, IL in hat:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.HAT, manufacturer="JFE/Nippon",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # FLAT WEB (for combined / king-pile walls)
    # ══════════════════════════════════════
    flat = [
        ("PZC 13",   575, 345,  9.0, 10.0, 85, 108, 14000,  812,  910, 3000),
        ("PZC 18",   575, 407, 10.5, 11.5, 110, 140, 26000, 1278, 1431, 3500),
        ("PZC 26",   575, 460, 12.5, 14.0, 147, 187, 44500, 1935, 2167, 4000),
    ]
    for name, w, h, tw, tf, wt, A, I, Ze, Zp, IL in flat:
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.FLAT, manufacturer="ArcelorMittal",
            width=w, height=h, thickness_web=tw, thickness_flange=tf,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=IL,
        ))

    # ══════════════════════════════════════
    # PENNAR INDUSTRIES (India) — Cold-formed
    # ══════════════════════════════════════
    # Source: Pennar Industries Sheet Pile Brochure
    # Cold-formed interlocked sections. Indian manufacturer (Hyderabad).
    # Standard grades: Fe360 (fy=235), Fe510 (fy≥345), BSK46 (fy≥380)
    # Brochure gives Ze only; Zp estimated as 1.10 × Ze for cold-formed.
    # Interlock strength estimated from thickness.

    def _il(t):
        """Estimate interlock strength (kN/m) from thickness."""
        if t <= 5: return 1500
        elif t <= 7: return 2000
        elif t <= 9: return 2500
        elif t <= 10: return 2800
        else: return 3000

    # ── PRU series (U-type) — 21 sections ──
    pru = [
        # (name, width, height, t, area_cm2/m, weight_wall_kg/m2, I_cm4/m, Ze_cm3/m)
        ("PRU7",      750, 320,  5,  71.3,  56.0, 10725,  670),
        ("PRU8",      750, 320,  6,  86.7,  68.1, 13169,  823),
        ("PRU9",      750, 320,  7, 101.4,  79.6, 15251,  953),
        ("PRU10-450", 450, 360,  8, 148.6, 116.7, 18268, 1015),
        ("PRU11-450", 450, 360,  9, 165.9, 130.2, 20375, 1132),
        ("PRU12-450", 450, 360, 10, 182.9, 143.8, 22444, 1247),
        ("PRU11-575", 575, 360,  8, 133.8, 105.1, 19685, 1094),
        ("PRU12-575", 575, 360,  9, 149.5, 117.4, 21793, 1221),
        ("PRU13-575", 575, 360, 10, 165.0, 129.5, 24224, 1348),
        ("PRU11-600", 600, 360,  8, 131.4, 103.2, 19897, 1105),
        ("PRU12-600", 600, 360,  9, 149.5, 117.4, 21973, 1221),
        ("PRU13-600", 600, 360, 10, 182.4, 127.5, 24491, 1361),
        ("PRU16-650", 650, 480,  8, 138.5, 109.6, 39864, 1661),
        ("PRU18-650", 650, 480,  9, 156.1, 122.3, 44521, 1855),
        ("PRU20-650", 650, 540,  8, 153.7, 120.2, 56002, 2074),
        ("PRU23-650", 650, 540,  9, 169.4, 133.0, 61084, 2318),
        ("PRU26-650", 650, 540, 10, 187.4, 146.9, 69093, 2559),
        ("PRU30-700", 700, 558, 11, 217.1, 170.5, 83139, 2980),
        ("PRU32-700", 700, 560, 12, 236.2, 185.4, 90880, 3246),
        ("PRU32-750", 750, 598, 11, 215.9, 169.5, 97362, 3256),
        ("PRU35-750", 750, 600, 12, 234.9, 184.4, 109416, 3547),
    ]
    for name, w, h, t, A, wt, I, Ze in pru:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.U, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    # ── PRZ series (Z-type) — 13 sections ──
    prz = [
        ("PRZ16-635", 635, 379,  7, 123.4,  96.9, 30502, 1610),
        ("PRZ18-635", 635, 380,  8, 140.6, 110.3, 34717, 1827),
        ("PRZ28-635", 635, 419, 11, 209.0, 164.1, 28785, 2805),
        ("PRZ30-635", 635, 420, 12, 227.3, 178.4, 63889, 3042),
        ("PRZ12-650", 650, 319,  7, 113.2,  88.9, 19603, 1229),
        ("PRZ14-650", 650, 320,  8, 128.9, 101.2, 22312, 1305),
        ("PRZ14-700", 700, 419,  7, 111.9,  87.8, 30824, 1471),
        ("PRZ16-700", 700, 420,  8, 127.5, 100.0, 35074, 1670),
        ("PRZ22-700", 700, 419,  9, 158.6, 124.5, 47058, 2246),
        ("PRZ25-700", 700, 420, 10, 175.6, 137.9, 52095, 2491),
        ("PRZ30-700", 700, 449, 11, 203.4, 159.7, 67025, 2986),
        ("PRZ32-700", 700, 450, 12, 221.3, 173.7, 72863, 3238),
        ("PRZ36-700", 700, 500, 12, 227.8, 178.9, 91788, 3672),
    ]
    for name, w, h, t, A, wt, I, Ze in prz:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.Z, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    # ── PRL series (S/L light type) — 6 sections ──
    prl = [
        ("PRL 1.5", 700, 100, 3.0,  39.0,  30.6,   724,  145),
        ("PRL 2",   700, 150, 3.0,  41.7,  32.7,  1674,  223),
        ("PRL 3",   700, 150, 4.5,  63.7,  50.0,  2469,  329),
        ("PRL 4",   700, 180, 5.0,  73.5,  57.7,  3979,  442),
        ("PRL 5",   700, 180, 6.5,  95.9,  75.3,  5094,  566),
        ("PRL 6",   700, 180, 7.0, 103.9,  81.6,  5458,  606),
    ]
    for name, w, h, t, A, wt, I, Ze in prl:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.SL, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    # ── PRS series (S-type) — 6 sections ──
    prs = [
        ("PRS 4",   600, 260, 3.5,  53.1,  41.7,  5528,  425),
        ("PRS 5",   600, 260, 4.0,  62.2,  48.8,  6703,  516),
        ("PRS 6",   600, 260, 5.0,  76.9,  57.7,  7899,  608),
        ("PRS 8",   750, 320, 5.5,  90.0,  70.7, 12987,  812),
        ("PRS 9",   750, 320, 6.5, 106.3,  83.4, 15225,  952),
        ("PRS 6*",  700, 180, 7.0, 103.9,  81.6,  5458,  606),  # PRL 6 also listed under PRS
    ]
    for name, w, h, t, A, wt, I, Ze in prs:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.HAT, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    # ── PRX Straight Web Piles — 3 sections ──
    prx = [
        ("PRX 600-10", 600, 60, 10.0, 144.8, 113.6, 396, 132),
        ("PRX 600-11", 600, 61, 11.0, 158.5, 124.4, 435, 143),
        ("PRX 600-12", 600, 62, 12.0, 172.1, 135.1, 474, 153),
    ]
    for name, w, h, t, A, wt, I, Ze in prx:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.FLAT, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    # ── PRG Trench Piles — 5 sections ──
    prg = [
        ("PRG I-1",  750, 95, 6.0,  72.1,  56.6,  975,  202),
        ("PRG I-2",  750, 95, 7.0,  84.2,  66.1, 1139,  234),
        ("PRG I-3",  750, 97, 8.0,  96.4,  75.7, 1304,  266),
        ("PRG II-1", 650, 80, 6.0,  79.5,  62.4,  758,  190),
        ("PRG II-2", 650, 82, 8.0, 105.7,  83.0, 1013,  247),
    ]
    for name, w, h, t, A, wt, I, Ze in prg:
        Zp = int(Ze * 1.10)
        db.append(SheetPileSection(
            name=name, profile_type=ProfileType.TRENCH, manufacturer="Pennar",
            width=w, height=h, thickness_web=t, thickness_flange=t,
            weight=wt, area=A, moment_of_inertia=I,
            elastic_modulus=Ze, plastic_modulus=Zp, interlock_strength=_il(t),
        ))

    return db


# Module-level database
SECTION_DATABASE: List[SheetPileSection] = _build_database()


# ══════════════════════════════════════════════════════════════
# SECTION LOOKUP FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_all_sections() -> List[SheetPileSection]:
    """Return complete section database."""
    return SECTION_DATABASE


def get_sections_by_type(profile_type: ProfileType) -> List[SheetPileSection]:
    """Filter sections by profile type."""
    return [s for s in SECTION_DATABASE if s.profile_type == profile_type]


def get_section_by_name(name: str) -> Optional[SheetPileSection]:
    """Look up a section by exact name."""
    for s in SECTION_DATABASE:
        if s.name.upper() == name.upper():
            return s
    return None


def search_sections(query: str) -> List[SheetPileSection]:
    """Fuzzy search by name substring."""
    q = query.upper()
    return [s for s in SECTION_DATABASE if q in s.name.upper()]


def get_sections_by_manufacturer(manufacturer: str) -> List[SheetPileSection]:
    """Filter sections by manufacturer name."""
    m = manufacturer.upper()
    return [s for s in SECTION_DATABASE if m in s.manufacturer.upper()]


def get_manufacturers() -> List[str]:
    """Return list of unique manufacturers in database."""
    return sorted(set(s.manufacturer for s in SECTION_DATABASE))


# ══════════════════════════════════════════════════════════════
# UTILIZATION CHECK — IS 800:2007 (LSM)
# ══════════════════════════════════════════════════════════════

@dataclass
class UtilizationResult:
    """Complete utilization check result for a sheet pile section."""
    section: SheetPileSection
    grade: SteelGrade

    # ── Applied forces (per m of wall) ──
    M_applied: float              # kN·m/m (max bending moment)
    V_applied: float              # kN/m   (max shear force)
    P_applied: float = 0.0       # kN/m   (axial force, compression +ve)

    # ── Bending check (IS 800 Cl. 8.2.1) ──
    Md: float = 0.0              # kN·m/m (design bending capacity)
    utilization_bending: float = 0.0
    status_bending: str = ""

    # ── Shear check (IS 800 Cl. 8.4) ──
    Vd: float = 0.0              # kN/m (design shear capacity)
    utilization_shear: float = 0.0
    status_shear: str = ""

    # ── Combined bending + axial (IS 800 Cl. 9.3) ──
    utilization_combined: float = 0.0
    status_combined: str = ""

    # ── Interlock shear ──
    V_interlock: float = 0.0     # kN/m (shear at interlocks)
    utilization_interlock: float = 0.0
    status_interlock: str = ""

    # ── Overall ──
    max_utilization: float = 0.0
    governing_check: str = ""
    overall_status: str = ""


def check_section(section: SheetPileSection, grade: SteelGrade,
                  M: float, V: float, P: float = 0.0,
                  gamma_m0: float = 1.10) -> UtilizationResult:
    """
    Perform complete utilization check per IS 800:2007.

    Parameters:
        section: SheetPileSection object
        grade: SteelGrade enum
        M: Applied bending moment (kN·m/m), absolute value
        V: Applied shear force (kN/m), absolute value
        P: Applied axial force (kN/m), compression positive
        gamma_m0: Partial safety factor for material (IS 800 Table 5)

    Returns:
        UtilizationResult with all checks.

    Clause references:
        Bending: IS 800 Cl. 8.2.1.2 (plastic section), Cl. 8.2.1.3 (compact)
        Shear:   IS 800 Cl. 8.4.1 (shear yielding)
        Combined: IS 800 Cl. 9.3.1 (bending + axial)
        Interlock: EN 10248 / BS 8004 approach
    """
    fy = STEEL_FY[grade]
    fu = STEEL_FU[grade]
    M = abs(M)
    V = abs(V)
    P = abs(P)

    result = UtilizationResult(
        section=section, grade=grade,
        M_applied=M, V_applied=V, P_applied=P
    )

    # ── 1. BENDING CHECK (IS 800 Cl. 8.2.1) ──
    # For Class 1 & 2: Md = Zp × fy / gamma_m0
    # For Class 3:     Md = Ze × fy / gamma_m0

    if section.section_class <= 2:
        # Plastic / compact → use plastic modulus
        Zp_m3 = section.plastic_modulus * 1e-6  # cm³ → m³
        Md = Zp_m3 * (fy * 1000) / gamma_m0     # kN·m/m
    else:
        # Semi-compact → use elastic modulus
        Ze_m3 = section.elastic_modulus * 1e-6
        Md = Ze_m3 * (fy * 1000) / gamma_m0

    result.Md = Md
    result.utilization_bending = M / Md if Md > 0 else 999
    result.status_bending = "OK" if result.utilization_bending <= 1.0 else "FAIL"

    # ── 2. SHEAR CHECK (IS 800 Cl. 8.4.1) ──
    # Vd = (fy / sqrt(3)) × Av / gamma_m0
    # For sheet piles, Av = web area = d × tw (per m of wall)
    # Number of webs per m: 2 / system_width (for Z-piles, 2 webs per pair)

    if section.profile_type in (ProfileType.Z, ProfileType.U):
        webs_per_m = 2.0 / (section.width / 1000)  # 2 webs per pile pair, per m of wall
    else:
        webs_per_m = 1.0 / (section.width / 1000)

    Av = section.height * section.thickness_web * webs_per_m  # mm² per m of wall
    Av_m2 = Av * 1e-6  # m²
    Vd = (fy / math.sqrt(3)) * Av_m2 * 1000 / gamma_m0  # kN/m

    result.Vd = Vd
    result.utilization_shear = V / Vd if Vd > 0 else 999
    result.status_shear = "OK" if result.utilization_shear <= 1.0 else "FAIL"

    # ── 2a. High shear check — reduce Md if V > 0.6 Vd ──
    Md_reduced = Md
    if V > 0.6 * Vd:
        # IS 800 Cl. 9.2.2: reduced moment capacity under high shear
        beta = ((2 * V / Vd) - 1) ** 2
        Zp_m3 = section.plastic_modulus * 1e-6
        Md_reduced = Zp_m3 * (fy * 1000) * (1 - beta) / gamma_m0
        # Re-check bending with reduced capacity
        if M / Md_reduced > result.utilization_bending:
            result.utilization_bending = M / Md_reduced
            result.Md = Md_reduced
            result.status_bending = "OK" if result.utilization_bending <= 1.0 else "FAIL"

    # ── 3. COMBINED BENDING + AXIAL (IS 800 Cl. 9.3.1) ──
    if P > 0:
        A_m2 = section.area * 1e-4  # cm² → m²
        Nd = A_m2 * fy * 1000 / gamma_m0  # kN/m (axial capacity)
        # Interaction: (M/Md) + (P/Nd) <= 1.0  (simplified, conservative)
        result.utilization_combined = (M / Md) + (P / Nd)
        result.status_combined = "OK" if result.utilization_combined <= 1.0 else "FAIL"
    else:
        result.utilization_combined = result.utilization_bending
        result.status_combined = result.status_bending

    # ── 4. INTERLOCK SHEAR CHECK ──
    # V_interlock = V × S / I  (shear flow at neutral axis)
    # For Z-piles: interlocks are at extremes, so interlock shear ≈ 0 (low)
    # For U-piles: interlocks at neutral axis — critical
    if section.profile_type == ProfileType.U:
        # First moment of area about NA for half-section
        # Approximate: V_interlock ≈ V × (section area above NA) × y_bar / I
        # Simplified: interlock shear ≈ V (conservative for U-piles)
        V_il = V * 0.8  # 80% of total shear at NA interlock
    else:
        V_il = V * 0.3  # Z-piles: interlocks at flanges, lower shear

    result.V_interlock = V_il
    if section.interlock_strength > 0:
        result.utilization_interlock = V_il / section.interlock_strength
    else:
        result.utilization_interlock = 0
    result.status_interlock = "OK" if result.utilization_interlock <= 1.0 else "FAIL"

    # ── 5. GOVERNING CHECK ──
    checks = {
        'Bending': result.utilization_bending,
        'Shear': result.utilization_shear,
        'Combined': result.utilization_combined,
        'Interlock': result.utilization_interlock,
    }
    result.max_utilization = max(checks.values())
    result.governing_check = max(checks, key=checks.get)
    result.overall_status = "OK" if result.max_utilization <= 1.0 else "FAIL"

    return result


# ══════════════════════════════════════════════════════════════
# WORKING STRESS METHOD (alternative)
# ══════════════════════════════════════════════════════════════

def check_section_wsd(section: SheetPileSection, grade: SteelGrade,
                      M: float, V: float, P: float = 0.0) -> UtilizationResult:
    """
    Working Stress Design check (IS 800:1984 / IS 14458).

    Permissible stresses:
        Bending: 0.66 × fy
        Shear:   0.40 × fy
        Axial:   0.60 × fy
    """
    fy = STEEL_FY[grade]
    M = abs(M); V = abs(V); P = abs(P)

    result = UtilizationResult(
        section=section, grade=grade,
        M_applied=M, V_applied=V, P_applied=P
    )

    # Bending
    sigma_b_perm = 0.66 * fy  # MPa
    Ze_m3 = section.elastic_modulus * 1e-6  # m³/m
    Md = Ze_m3 * sigma_b_perm * 1000  # kN·m/m
    result.Md = Md
    result.utilization_bending = M / Md if Md > 0 else 999
    result.status_bending = "OK" if result.utilization_bending <= 1.0 else "FAIL"

    # Shear
    tau_perm = 0.40 * fy
    if section.profile_type in (ProfileType.Z, ProfileType.U):
        webs_per_m = 2.0 / (section.width / 1000)
    else:
        webs_per_m = 1.0 / (section.width / 1000)
    Av = section.height * section.thickness_web * webs_per_m * 1e-6  # m²
    Vd = tau_perm * Av * 1000  # kN/m
    result.Vd = Vd
    result.utilization_shear = V / Vd if Vd > 0 else 999
    result.status_shear = "OK" if result.utilization_shear <= 1.0 else "FAIL"

    # Combined
    if P > 0:
        sigma_a_perm = 0.60 * fy
        A_m2 = section.area * 1e-4
        sigma_a = P / A_m2 / 1000  # MPa
        sigma_b = M / Ze_m3 / 1000  # MPa
        result.utilization_combined = (sigma_b / sigma_b_perm) + (sigma_a / sigma_a_perm)
    else:
        result.utilization_combined = result.utilization_bending
    result.status_combined = "OK" if result.utilization_combined <= 1.0 else "FAIL"

    # Interlock (same as LSM)
    V_il = V * (0.8 if section.profile_type == ProfileType.U else 0.3)
    result.V_interlock = V_il
    result.utilization_interlock = V_il / section.interlock_strength if section.interlock_strength > 0 else 0
    result.status_interlock = "OK" if result.utilization_interlock <= 1.0 else "FAIL"

    # Governing
    checks = {
        'Bending': result.utilization_bending,
        'Shear': result.utilization_shear,
        'Combined': result.utilization_combined,
        'Interlock': result.utilization_interlock,
    }
    result.max_utilization = max(checks.values())
    result.governing_check = max(checks, key=checks.get)
    result.overall_status = "OK" if result.max_utilization <= 1.0 else "FAIL"

    return result


# ══════════════════════════════════════════════════════════════
# AUTO-SELECTION
# ══════════════════════════════════════════════════════════════

@dataclass
class SelectionResult:
    """Result of auto-selecting the optimal section."""
    recommended: SheetPileSection
    grade: SteelGrade
    utilization: UtilizationResult
    all_checked: List[UtilizationResult]
    selection_criteria: str
    alternatives: List[Tuple[SheetPileSection, UtilizationResult]]


def auto_select(M: float, V: float, P: float = 0.0,
                grade: SteelGrade = SteelGrade.S355GP,
                profile_types: Optional[List[ProfileType]] = None,
                max_utilization: float = 0.90,
                method: str = "LSM") -> SelectionResult:
    """
    Auto-select the lightest adequate section.

    Parameters:
        M: Max bending moment (kN·m/m)
        V: Max shear force (kN/m)
        P: Axial force (kN/m)
        grade: Steel grade
        profile_types: Filter by profile type (None = all)
        max_utilization: Target max utilization ratio (default 0.90)
        method: "LSM" (IS 800:2007) or "WSD" (working stress)

    Returns:
        SelectionResult with recommended section and alternatives.
    """
    check_fn = check_section if method == "LSM" else check_section_wsd

    candidates = SECTION_DATABASE
    if profile_types:
        candidates = [s for s in candidates if s.profile_type in profile_types]

    # Check all sections
    all_results = []
    for sec in candidates:
        res = check_fn(sec, grade, M, V, P)
        all_results.append(res)

    # Filter passing sections
    passing = [(r.section, r) for r in all_results if r.max_utilization <= max_utilization]

    if not passing:
        # Try with max_utilization = 1.0
        passing = [(r.section, r) for r in all_results if r.max_utilization <= 1.0]

    if not passing:
        # Nothing passes — return lightest section with note
        all_results.sort(key=lambda r: r.max_utilization)
        best = all_results[0]
        return SelectionResult(
            recommended=best.section,
            grade=grade,
            utilization=best,
            all_checked=all_results,
            selection_criteria=f"NO SECTION ADEQUATE — lightest has util={best.max_utilization:.2f}",
            alternatives=[],
        )

    # Sort by weight (lightest first)
    passing.sort(key=lambda x: x[0].weight)

    recommended_sec, recommended_util = passing[0]

    # Get alternatives (next 3 lightest)
    alternatives = passing[1:4]

    return SelectionResult(
        recommended=recommended_sec,
        grade=grade,
        utilization=recommended_util,
        all_checked=all_results,
        selection_criteria=f"Lightest section with util <= {max_utilization:.0%}",
        alternatives=alternatives,
    )


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE GENERATOR
# ══════════════════════════════════════════════════════════════

def compare_sections(sections: List[str], grade: SteelGrade,
                     M: float, V: float, P: float = 0.0,
                     method: str = "LSM") -> List[UtilizationResult]:
    """
    Check multiple named sections and return comparison.

    Parameters:
        sections: List of section names (e.g. ["LARSSEN 2", "LARSSEN 4"])
        grade: Steel grade
        M, V, P: Applied forces
        method: "LSM" or "WSD"

    Returns:
        List of UtilizationResult for each section.
    """
    check_fn = check_section if method == "LSM" else check_section_wsd
    results = []
    for name in sections:
        sec = get_section_by_name(name)
        if sec:
            results.append(check_fn(sec, grade, M, V, P))
    return results


def get_grade_comparison(section_name: str,
                         M: float, V: float, P: float = 0.0,
                         method: str = "LSM") -> List[UtilizationResult]:
    """
    Check one section across all steel grades.
    Useful for deciding grade when section is fixed (e.g. available stock).
    """
    sec = get_section_by_name(section_name)
    if not sec:
        return []

    check_fn = check_section if method == "LSM" else check_section_wsd
    results = []
    for grade in SteelGrade:
        results.append(check_fn(sec, grade, M, V, P))
    return results


# ══════════════════════════════════════════════════════════════
# SUMMARY / STATS
# ══════════════════════════════════════════════════════════════

def database_summary() -> Dict:
    """Return database statistics."""
    db = SECTION_DATABASE
    by_type = {}
    for pt in ProfileType:
        secs = [s for s in db if s.profile_type == pt]
        if secs:
            by_type[pt.value] = {
                'count': len(secs),
                'weight_range': f"{min(s.weight for s in secs):.0f} - {max(s.weight for s in secs):.0f} kg/m²",
                'Zp_range': f"{min(s.plastic_modulus for s in secs):.0f} - {max(s.plastic_modulus for s in secs):.0f} cm³/m",
            }
    return {
        'total_sections': len(db),
        'by_type': by_type,
        'steel_grades': len(SteelGrade),
    }
