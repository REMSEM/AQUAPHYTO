"""
config.py
=========
Central configuration for the AQUAPHYTO project.
Edit only this file when paths or constants change.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent


# ── Data directories ──────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Station-specific raw data folders
RT1_RAW_2024    = DATA_RAW / "RT1_2024"
RT1_RAW_2025    = DATA_RAW / "RT1_2025"
CPOWER_RAW_2025 = DATA_RAW / "CPOWER_2025"

# Processed datacubes (.npz)
RT1_DATACUBE_2025    = DATA_PROCESSED / "datacube_RT1_2025.npz"
RT1_DATACUBE_2024    = DATA_PROCESSED / "datacube_RT1_2024.npz"
CPOWER_DATACUBE_2025 = DATA_PROCESSED / "datacube_CPOWER_2025.npz"

# Figures
FIGURES_DIR = ROOT / "figures"

# Final ML-ready datasets (.csv)
RT1_DATASET_2025     = DATA_PROCESSED / "REFERENCE_DATASET_4_WP2_RT1_2025.csv"
RT1_DATASET_2024     = DATA_PROCESSED / "REFERENCE_DATASET_4_WP2_RT1_2024.csv"

# Buiteveld (1994) water absorption look-up table
# Columns: lambda [nm], a [m⁻¹], A [m⁻¹ °C⁻¹]
# Temperature correction applied in algorithms.py: aw_T = a + A × (T − 20.1)
BUITEVELD_COEFFS = DATA_RAW / "buiteveld_coeffs.csv"
BUITEVELD_T      = 10.0    # water temperature [°C] — typical Belgian coastal water


# ── Station metadata ───────────────────────────────────────────────────────────
STATIONS = {
    "RT1":    {"lon": 2.9193,  "lat": 51.2464},
    "CPOWER": {"lon": 2.9566,  "lat": 51.2970},
    "MOW1":   {"lon": 2.8050,  "lat": 51.3633},
}

# ── Physical constants ─────────────────────────────────────────────────────────
AW_700 = 0.57     # water absorption at 700 nm [m⁻¹]  Kou et al. (1993)
AW_670 = 0.439    # water absorption at 670 nm [m⁻¹]
AW_708 = 0.840    # water absorption at 708 nm [m⁻¹]
APH_STAR_670 = 0.016   # specific chl-a absorption at 670 nm [m² mg⁻¹]

# ── Gons (2002) Chl-a algorithm — red-NIR ratio, works on coarse grids ─────────
# Reference: Gons et al. (2002), J. Plankton Res. 24(9), 947-951
# Wavelengths: 665, 708, 779 nm — all available on CHIME 5 nm and 10 nm grids
GONS_L1        = 665.0   # nm  — Chl-a absorption reference band
GONS_L2        = 708.0   # nm  — red-edge fluorescence band
GONS_L3        = 779.0   # nm  — NIR backscattering reference band
# Water absorption at Gons wavelengths [m⁻¹] — from Buiteveld (1994) at 10°C
# Used as fallback if Buiteveld table is not available; normally interpolated.
GONS_AW_L1     = 0.401   # aw at 665 nm  [m⁻¹]
GONS_AW_L2     = 0.840   # aw at 708 nm  [m⁻¹]  (same as AW_708)
# Specific Chl-a absorption at 665 nm [m² mg⁻¹]
GONS_APH_STAR  = 0.0161
# Backscattering slope coefficients (Gons 2002, eq. 4)
GONS_BB_A      = 1.61    # numerator factor
GONS_BB_B      = 0.082   # denominator offset
GONS_BB_C      = 0.6     # denominator slope

# ── CRAT minimum valid bands — sensor-specific ─────────────────────────────────
# CRAT requires resolving the narrow red-peak shape; threshold depends on Δλ.
CHL_MIN_BANDS_PANTHYR = 200   # PANTHYR: Δλ = 2.5 nm, ~240 bands
CHL_MIN_BANDS_CHIME   = 50    # CHIME:   Δλ = 5/10 nm, ~120/60 bands

# ── Algorithm parameters ───────────────────────────────────────────────────────
# MALH wavelengths (Lavigne et al. 2022, eq. 1)
MALH_L1   = 470.0
MALH_L2   = 482.5
MALH_L3   = 490.0
MALH_LNIR = 700.0

# Second derivative (Lubac et al. 2008) — legacy PANTHYR implementation (algorithms.compute_D2)
# D2_NORM_WL: normalisation wavelength [nm]
#   None      → array index 35 (= 442.5 nm on the PANTHYR grid only)
#   620.0     → 620 nm (Lubac et al. 2008 convention)
#   any float → nearest wavelength in the grid
D2_NORM_WL  = None    # default: index 35 (442.5 nm on PANTHYR)
D2_DELTA    = 2.5     # wavelength step [nm]; must equal the grid step (PANTHYR: 2.5 nm)
D2_N_SMOOTH = 2       # number of 5-point smoothing passes (defined in samples; PANTHYR only)

# ── QC thresholds ─────────────────────────────────────────────────────────────
QC_LDEX_MAX  = 0.05   # max Ld/Ed at 750 nm  [sr⁻¹]
QC_NAN_MAX   = 10     # max NaN per spectrum
QC_SZA_MAX   = 75.0   # max solar zenith angle [°]
QC_CHL_MIN   = 3.0    # min Chl-a to keep     [mg m⁻³]

# ── Cyanobacteria detection (phytospec/cyano.py) ───────────────────────────────
# Second derivative: explicit 3-point formula, step taken from the actual grid.
# Smoothing is defined in nm (target effective resolution), never in samples.
CY_DESIGN_STEP  = 2.5                      # nm — fine sensors are resampled to this regular grid
CY_SMOOTH_FWHM  = 10.0                     # nm — target resolution for REAL measurements;
                                           #      D²(620) > 0 loses specificity from ~12 nm (simulated CHIME tests)
                                           #      (None for noise-free simulated spectra)
CY_D2_WL        = 620.0                    # nm — phycocyanin trough: D²(620) > 0 → Cy+
CY_D2_WL_LEGACY = 552.5                    # nm — legacy green-peak index: D²(552) < 0 → Cy+
CY_TROUGH_WLS   = (597.5, 622.5, 647.5)    # nm — trough depth PC_T (experimental, not used for detection)
CY_DOM_FRAC     = 0.30                     # in-situ: cyano-dominated if ≥ 30 % of phytoplankton biomass
