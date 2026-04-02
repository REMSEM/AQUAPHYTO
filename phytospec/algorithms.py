"""
phytospec/algorithms.py
=====================
Pure spectral algorithms — no I/O, no side effects.
Each function takes (rhow, wl) arrays and returns a scalar or array.

References
----------
MALH      : Lavigne et al. (2022), Remote Sensing of Environment, 282, 113270
             R implementation: AstorecaC() in functions_chl_phaeo.R
CHL (CRAT): Ruddick et al. (2001), Appl. Opt. 40  / Buiteveld et al. (1994)
             R implementation: CRAT() in functions_chl_phaeo.R
             Requires ≥ 50–200 bands depending on sensor (see min_bands param).
D2        : Lubac et al. (2008), JGR Oceans 113
             R implementation: D2() in functions_chl_phaeo.R
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from phytospec import config as cfg


# ── Buiteveld (1994) water absorption — loaded once at import time ─────────────
# CSV columns: lambda [nm], a [m⁻¹], A [m⁻¹ °C⁻¹]
# Temperature correction (identical to R CRAT):  a_T = a + A × (T − 20.1)
# Default T = 10°C (typical Belgian coastal water, North Sea)
# File path is defined in config.py (BUITEVELD_COEFFS).

def _load_buiteveld(path, T: float = 10.0):
    """
    Load Buiteveld coefficients CSV and return temperature-corrected
    (wl, a_T) arrays ready for np.interp().
    """
    df = pd.read_csv(path)
    wl_b = df["lambda"].values.astype(float)
    a_T  = (df["a"] + df["A"] * (T - 20.1)).values.astype(float)
    return wl_b, a_T

# Load at import time so every compute_CHL() call is fast.
_BW_WL, _BW_A = _load_buiteveld(cfg.BUITEVELD_COEFFS, T=cfg.BUITEVELD_T)


# ── wavelength utilities ───────────────────────────────────────────────────────

def interp_at(wl: np.ndarray, spectrum: np.ndarray, target: float) -> float:
    """
    Linear interpolation of `spectrum` at wavelength `target`.
    Returns NaN if target is outside the grid or neighbours are NaN.
    """
    if target < wl[0] or target > wl[-1]:
        return np.nan
    idx = int(np.searchsorted(wl, target))
    if idx == 0:
        return float(spectrum[0])
    if idx >= len(wl):
        return float(spectrum[-1])
    w1, w2 = wl[idx - 1], wl[idx]
    v1, v2 = spectrum[idx - 1], spectrum[idx]
    if np.isnan(v1) or np.isnan(v2):
        return np.nan
    return float(v1 + (v2 - v1) * (target - w1) / (w2 - w1))


# ── MALH index ─────────────────────────────────────────────────────────────────

def compute_MALH(rhow: np.ndarray, wl: np.ndarray,
                 l1:   float = cfg.MALH_L1,    # 470 nm
                 l2:   float = cfg.MALH_L3,    # 490 nm  (far wavelength)
                 lc3:  float = cfg.MALH_L2,    # 482.5 nm (center)
                 lNIR: float = cfg.MALH_LNIR   # 700 nm
                 ) -> float:
    """
    Modified Astoreca Line Height (MALH) .

    Formula (matching R implementation):
        w    = (lc3 - l1) / (l2 - l1)          →  0.625 for defaults
        MALH = [1/ρw(lc3) - (1/ρw(l1))^(1-w) · (1/ρw(l2))^w] × 0.57 × ρw(lNIR)


    Returns
    -------
    float : MALH [m⁻¹]
        Positive  → P. globosa present
        <= 0      → P. globosa absent
    """
    w = (lc3 - l1) / (l2 - l1)   # 0.625 for defaults

    r1  = interp_at(wl, rhow, l1)
    r2  = interp_at(wl, rhow, l2)
    rc3 = interp_at(wl, rhow, lc3)
    rN  = interp_at(wl, rhow, lNIR)

    if any(np.isnan([r1, r2, rc3, rN])):
        return np.nan
    if any(v <= 0 for v in [r1, r2, rc3, rN]):
        return np.nan

    baseline = (1.0 / r1) ** (1.0 - w) * (1.0 / r2) ** w
    anomaly  = 1.0 / rc3 - baseline

    return float(anomaly * 0.57 * rN)


# ── Chlorophyll-a (CRAT algorithm) ────────────────────────────────────────────

def compute_CHL(rhow: np.ndarray, wl: np.ndarray,
                aphy:      float = cfg.APH_STAR_670,
                min_bands: int   = cfg.CHL_MIN_BANDS_PANTHYR) -> float:
    """
    Chlorophyll-a using the CRAT red-peak algorithm.

    Algorithm (Ruddick et al. 2001 / Buiteveld et al. 1994):
      1. Require ≥ min_bands non-NaN wavelengths
         Default: 200 for PANTHYR (Δλ = 2.5 nm).
         Use cfg.CHL_MIN_BANDS_CHIME (= 50) for CHIME simulation data.
      2. Fix λ1 = 672 nm
      3. Find λmax = wavelength of maximum ρw in the strictly open window
         (672, 750] nm  — R: which(lambda1 < wl & wl <= 750)
      4. Verify spectral shape: ρw(750) < ρw(672) < ρw(λmax)
      5. Find λ2 = wavelength where ρw descends back through ρw(672)
         beyond λmax (linear interpolation of the crossing point)
      6. CHL = (aw_T(λ2) − aw_T(λ1)) / aphy*

    Parameters
    ----------
    rhow      : water reflectance spectrum [sr⁻¹]
    wl        : wavelength grid [nm], same length as rhow
    aphy      : specific Chl-a absorption at 670 nm [m² mg⁻¹] (default 0.016)
    min_bands : minimum number of non-NaN bands required (sensor-dependent).
                Use cfg.CHL_MIN_BANDS_CHIME for CHIME data.

    Returns
    -------
    float : Chl-a [mg m⁻³], or NaN if spectral shape is invalid.
    """
    # ── guard: require enough valid wavelengths ───────────────────
    if np.sum(~np.isnan(rhow)) < min_bands:
        return np.nan

    lam1 = 672.0

    # ── find λmax: strictly beyond 672 nm, up to 750 nm ──────────────────────
    # R: iii2 <- which(lambda1 < wl & wl <= 750)  → excludes 672 itself
    mask_peak = (wl > lam1) & (wl <= 750.0)
    if not np.any(mask_peak):
        return np.nan

    sub      = rhow[mask_peak]
    wl_sub   = wl[mask_peak]
    if np.all(np.isnan(sub)):
        return np.nan

    imax     = int(np.nanargmax(sub))
    rl_max   = sub[imax]
    lam_max  = wl_sub[imax]

    rl1  = interp_at(wl, rhow, lam1)
    r750 = interp_at(wl, rhow, 750.0)

    if np.isnan(rl1) or rl1 <= 0:
        return np.nan

    # ── shape check : R(750) < R(672) < R(λmax) ──────────────────
    if not (r750 < rl1 < rl_max):
        return np.nan

    # ── find λ2: ρw crosses back down through ρw(672) beyond λmax ────────────
    # R: iii2 <- which(lambdamax < wl & wl <= 750)
    mask_desc = (wl > lam_max) & (wl <= 750.0)
    if not np.any(mask_desc):
        return np.nan

    wl_desc   = wl[mask_desc]
    rhow_desc = rhow[mask_desc] - rl1          # shifted so crossing = 0

    lam2 = np.nan
    for j in range(len(rhow_desc) - 1):
        if rhow_desc[j] >= 0 and rhow_desc[j + 1] < 0:
            lam2 = np.interp(
                0.0,
                [rhow_desc[j + 1], rhow_desc[j]],
                [wl_desc[j + 1],   wl_desc[j]]
            )
            break

    if np.isnan(lam2):
        return np.nan

    # ── water absorption from Buiteveld table  ─────────
    aw1 = float(np.interp(lam1, _BW_WL, _BW_A))
    aw2 = float(np.interp(lam2, _BW_WL, _BW_A))

    chl = (aw2 - aw1) / aphy
    return float(chl) if chl > 0 else np.nan



# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_5pt(arr: np.ndarray, n_passes: int = cfg.D2_N_SMOOTH) -> np.ndarray:
    """
    5-point running-average smoothing applied `n_passes` times.
    Equivalent to R: filter(x, rep(1/5, 5), method='convolution') applied twice.
    NaN positions are preserved.
    """
    result = arr.astype(float).copy()
    nan_mask = np.isnan(result)
    for _ in range(n_passes):
        tmp = result.copy()
        tmp[nan_mask] = 0.0
        smoothed = uniform_filter1d(tmp, size=5, mode="nearest")
        smoothed[nan_mask] = np.nan
        result = smoothed
    return result


# ── Second derivative ─────────────────────────────────────────────────────────

def compute_D2(rhow: np.ndarray, wl: np.ndarray,
               norm_wl:  object = cfg.D2_NORM_WL,
               delta:    float  = cfg.D2_DELTA,
               n_smooth: int    = cfg.D2_N_SMOOTH) -> np.ndarray:
    """
    Second derivative of normalised water reflectance    

    Parameters
    ----------
    norm_wl  : normalisation wavelength [nm], or None for R-default (442.5 nm).
                 None  -> index 35 = 442.5 nm  (matches R exactly)
                 620.0 -> 620 nm  (Lubac 2008 paper / colleague Python convention)
                 Any float -> nearest wavelength in grid.
               Set globally via cfg.D2_NORM_WL in config.py.
    delta    : wavelength step [nm] (default 2.5)
    n_smooth : number of smoothing passes (default 2) and depends sensor

    Returns
    -------
    np.ndarray : same length as wl, NaN at boundary points.
    """
    if norm_wl is None:
        norm_idx = 35                                           # R: x/x[36] = 442.5 nm
    else:
        norm_idx = int(np.argmin(np.abs(wl - float(norm_wl)))) # nearest grid point

    if rhow[norm_idx] == 0 or np.isnan(rhow[norm_idx]):
        return np.full_like(wl, np.nan, dtype=float)

    rhow_N = rhow / rhow[norm_idx]
    rhow_S = smooth_5pt(rhow_N, n_passes=n_smooth)

    d2 = np.full_like(wl, np.nan, dtype=float)
    # R loop: for(i in 2:(n-2)) — 1-indexed → 0-indexed: range(1, n-2)
    for i in range(1, len(wl) - 2):
        d2[i] = (rhow_S[i + 1] - 2.0 * rhow_S[i] + rhow_S[i - 1]) / (delta ** 2)

    return d2


# ── Lubac classification ───────────────────────────────────────────────────────

def lubac_phaeo_index(wl: np.ndarray, d2r: np.ndarray) -> int:
    """
    Classify a spectrum as P. globosa dominant (1) or absent (0).
    R function: LubacFUN().

    P. globosa signature (Lubac et al. 2008):
        local maximum in d2r in 460-480 nm range >= 471 nm
        local minimum in d2r in 480-510 nm range >= 499 nm

    Returns
    -------
    int : 1 = P. globosa dominance, 0 = absence
    """
    
    try:
        wl = np.asarray(wl, dtype=float)
        d2r = np.asarray(d2r, dtype=float)

        # ── Maximum: search 460–490 nm ─────────────────────────────────────
        w_max = (wl >= 460.0) & (wl <= 490.0)
        if not w_max.any() or np.all(np.isnan(d2r[w_max])):
            return 0
        wl_max = float(wl[w_max][np.nanargmax(d2r[w_max])])

        # ── Second minimum: search 480–510 nm ─────────────────────────────
        w_min2 = (wl >= 480.0) & (wl <= 510.0)
        if not w_min2.any() or np.all(np.isnan(d2r[w_min2])):
            wl_min2 = 0.0   # safe fallback: fails the >= 499 check
        else:
            wl_min2 = float(wl[w_min2][np.nanargmin(d2r[w_min2])])

        # ── Classification ─────────────────────────────────────────────────

        # ── Both must shift to confirm P. globosa ─────────────────────────
        #   Diatoms   : max ~463,  min2 ~485  → both below threshold
        
        #   P. globosa: max ~475,  min2 ~510  → both above threshold
        return int(wl_max >= 470.0 and wl_min2 >= 495.0)


    except Exception:
        return 0