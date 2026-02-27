"""
config.py
=========
Central configuration for the PHAEO_RT1 project.
Edit only this file when paths or constants change.
"""

from pathlib import Path

# ── Project root  ──
# ROOT = Path(__file__).parent
# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent


# ── Data directories ──────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Station-specific raw data folders
RT1_RAW_2025    = DATA_RAW / "RT1_2025"
CPOWER_RAW_2025 = DATA_RAW / "CPOWER_2025"   # add 

# Processed datacubes (.npz)
RT1_DATACUBE_2025    = DATA_PROCESSED / "datacube_RT1_2025.npz"
CPOWER_DATACUBE_2025 = DATA_PROCESSED / "datacube_CPOWER_2025.npz"

# Final ML-ready datasets (.csv)
RT1_DATASET_2025     = DATA_PROCESSED / "REFERENCE_DATASET_4_WP2_RT1_2025.csv"

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

# ── Algorithm parameters ───────────────────────────────────────────────────────
# MALH wavelengths (Lavigne et al. 2022, eq. 1)
MALH_L1   = 470.0
MALH_L2   = 482.5
MALH_L3   = 490.0
MALH_LNIR = 700.0

# Second derivative (Lubac et al. 2008)
#D2_NORM_WL  = 620.0   # normalisation wavelength [nm] D2_NORM_WL = None    
# Second derivative (Lubac et al. 2008)
# D2_NORM_WL: normalisation wavelength [nm]
#   None   → use array index 35 = 442.5 nm  
#   620.0  → use 620 nm  (Lubac 2008 paper convention, colleague Python code)
#   any float → nearest wavelength in the grid will be used
D2_NORM_WL  = None    # default: 442.5 nm 
D2_DELTA    = 2.5     # wavelength step [nm]
D2_N_SMOOTH = 2       # number of 5-pt smoothing passes depends of the sensor

# ── QC thresholds ─────────────────────────────────────────────────────────────
QC_LDEX_MAX  = 0.05   # max Ld/Ed at 750 nm  [sr⁻¹]
QC_NAN_MAX   = 10     # max NaN per spectrum
QC_SZA_MAX   = 75.0   # max solar zenith angle [°]
QC_CHL_MIN   = 3.0    # min Chl-a to keep     [mg m⁻³]