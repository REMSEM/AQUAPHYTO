"""
phytospec/cyano.py
==================
Cyanobacteria detection from water reflectance spectra.
Pure functions, no I/O. Works for any sensor grid (hyperspectral or CHIME-like).

Design rules (learned from the AQUATIME D2 analysis, Sep 2026)
----------------------------------------------------------------
1. ONE second-derivative formula everywhere: explicit 3-point central difference
   (Lubac et al. 2008 / R D2()), with the step taken from the actual wavelength grid.
   A double np.gradient is NOT equivalent: it differentiates over 2*Δλ and acts as
   extra smoothing.
2. Smoothing is defined in NANOMETRES (target effective resolution, FWHM), never in
   number of bands. A 5-point window is ~12 nm on PANTHYR (2.5 nm) but ~42 nm on
   CHIME (8.4 nm).
3. Coarse sensors (Δλ > design step) are NOT up-sampled: cubic up-sampling creates
   curvature between the native bands and makes the sign of D² at a fixed wavelength
   depend on where it falls between bands.
4. Smoothing is NaN-aware: NaNs are excluded from each window average (never set to 0).
5. The cyanobacteria-specific feature is the phycocyanin absorption trough near
   620 nm (local reflectance minimum → D²(620) > 0). D²(552) < 0 (green peak) is kept
   only as the legacy index: it responds to the position of the green peak (Chl-a,
   CDOM, water) and is not specific to phycocyanin.

Sign conventions
----------------
    D²(552.5) < 0  → Cy+   (legacy, green-peak curvature)
    D²(620)   > 0  → Cy+   (phycocyanin trough)

Experimental
------------
    PC_T (trough depth) is computed for reference only. It is NOT part of the detection:
    it carried the same information as D²(620) in the AQUATIME evaluation, and its
    threshold would have to be calibrated per sensor spectral response.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

try:                                    # defaults from config.py when available
    from phytospec import config as _cfg
except Exception:                       # pragma: no cover
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))          # 2.3548

# ── defaults (overridable in config.py) ─────────────────────────────────────────
DESIGN_STEP   = _c('CY_DESIGN_STEP', 2.5)      # nm, regular grid used for fine sensors
SMOOTH_FWHM   = _c('CY_SMOOTH_FWHM', 10.0)     # nm, target effective resolution (None = no smoothing)
D2_WL_PC      = _c('CY_D2_WL', 620.0)          # nm, phycocyanin trough
D2_WL_LEGACY  = _c('CY_D2_WL_LEGACY', 552.5)   # nm, legacy green-peak index
TROUGH_WLS    = _c('CY_TROUGH_WLS', (597.5, 622.5, 647.5))   # (baseline low, centre, baseline high)

FEATURES = {    # name: (description, Cy+ rule)
    'D2_552': ('D²(552.5) — legacy green-peak curvature', 'D² < 0'),
    'D2_620': ('D²(620) — phycocyanin trough curvature', 'D² > 0'),
    'PC_T':   ('Phycocyanin trough depth at ~622 nm (experimental, not used for detection)', 'T ≥ threshold'),
}


# ── grid utilities ─────────────────────────────────────────────────────────────

def check_grid(wl):
    """Return the median step of a strictly increasing wavelength grid; raise otherwise."""
    wl = np.asarray(wl, float)
    if wl.ndim != 1 or wl.size < 3:
        raise ValueError('wavelength grid must be 1-D with at least 3 values')
    d = np.diff(wl)
    if np.any(~np.isfinite(wl)) or np.any(d <= 0):
        raise ValueError('wavelengths must be finite and strictly increasing')
    return float(np.median(d))


def value_at(wl, Y, lam):
    """
    Linear interpolation of Y (…, n_wl) at wavelength `lam`, vectorised over rows.
    NaN if `lam` is outside the grid or if a bracketing value is NaN
    (same behaviour as algorithms.interp_at).
    """
    wl = np.asarray(wl, float); Y = np.asarray(Y, float)
    if lam < wl[0] or lam > wl[-1]:
        return np.full(Y.shape[:-1], np.nan) if Y.ndim > 1 else np.nan
    j = int(np.clip(np.searchsorted(wl, lam), 1, len(wl) - 1))
    w = (lam - wl[j - 1]) / (wl[j] - wl[j - 1])
    out = Y[..., j - 1] * (1.0 - w) + Y[..., j] * w
    return out if Y.ndim > 1 else float(out)


def nearest_band(wl, lam):
    """Wavelength of the grid point closest to `lam`."""
    wl = np.asarray(wl, float)
    return float(wl[int(np.argmin(np.abs(wl - lam)))])


def bracketing_bands(wl, lam):
    """The two grid wavelengths that bracket `lam` (for band-sensitivity diagnostics)."""
    wl = np.asarray(wl, float)
    j = int(np.clip(np.searchsorted(wl, lam), 1, len(wl) - 1))
    return float(wl[j - 1]), float(wl[j])


# ── smoothing and resampling ───────────────────────────────────────────────────

def smooth_gaussian_nm(R, wl, fwhm_target, native_fwhm=None, min_weight=0.5):
    """
    Gaussian smoothing so that the effective resolution becomes `fwhm_target` [nm].

    The extra kernel is added in quadrature to the native resolution
    (max(Δλ, native_fwhm)); if the data are already at or above the target, they are
    returned unchanged. NaN-aware: NaNs are excluded from the weighted average
    (normalised convolution); positions that were NaN stay NaN, and positions whose
    valid-weight fraction is below `min_weight` become NaN.

    Parameters
    ----------
    R           : (n_wl,) or (n_spectra, n_wl)
    fwhm_target : nm, or None for no smoothing
    """
    R = np.asarray(R, float)
    if fwhm_target is None:
        return R.copy()
    d = check_grid(wl)
    native = max(d, native_fwhm or 0.0)
    extra = np.sqrt(max(fwhm_target ** 2 - native ** 2, 0.0))
    sigma = extra / FWHM_PER_SIGMA / d          # in samples
    if sigma < 0.1:                             # negligible extra smoothing
        return R.copy()
    valid = np.isfinite(R)
    num = gaussian_filter1d(np.where(valid, R, 0.0), sigma, axis=-1, mode='nearest')
    den = gaussian_filter1d(valid.astype(float), sigma, axis=-1, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den >= min_weight, num / den, np.nan)
    out[~valid] = np.nan
    return out


def prepare_spectra(R, wl, smooth_fwhm=SMOOTH_FWHM, native_fwhm=None, design_step=DESIGN_STEP):
    """
    Bring spectra to the analysis scale.

    * Fine sensors (Δλ ≤ design_step): smooth to `smooth_fwhm`, then resample linearly
      to a regular `design_step` grid (multiples of design_step inside the data range).
    * Coarse sensors (Δλ > design_step): smooth to `smooth_fwhm` if it is wider than the
      native resolution, and KEEP the native grid (no up-sampling).

    Returns
    -------
    R_out, wl_out, info (dict: step, native_resolution, effective_resolution, regridded)
    """
    R = np.atleast_2d(np.asarray(R, float))
    d = check_grid(wl)
    native = max(d, native_fwhm or 0.0)
    S = smooth_gaussian_nm(R, wl, smooth_fwhm, native_fwhm=native_fwhm)
    eff = max(native, smooth_fwhm or 0.0)
    if d <= design_step:
        lo = np.ceil(wl[0] / design_step) * design_step
        wl_out = np.arange(lo, wl[-1] + 1e-9, design_step)
        R_out = interp1d(wl, S, axis=1, bounds_error=False, fill_value=np.nan)(wl_out)
        regridded = True
    else:
        wl_out, R_out, regridded = np.asarray(wl, float), S, False
    info = dict(step=check_grid(wl_out), native_resolution=native,
                effective_resolution=eff, regridded=regridded)
    return R_out, wl_out, info


# ── second derivative ──────────────────────────────────────────────────────────

def second_derivative(R, wl, norm_wl=None):
    """
    Explicit 3-point second derivative on the given grid (no smoothing inside).

        regular grid:   D²_i = (f[i+1] − 2 f[i] + f[i−1]) / Δλ²
        irregular grid: D²_i = 2 [ (f[i+1]−f[i])/h₂ − (f[i]−f[i−1])/h₁ ] / (h₁ + h₂)

    (the second form reduces exactly to the first on a regular grid). The step is taken
    from `wl`, so there is no hard-coded Δλ. Optionally the spectrum is first divided by
    R(norm_wl) (linear interpolation); a normalisation value ≤ 0 or NaN gives a NaN row.
    Normalisation never changes the sign of D².

    Returns an array with the shape of R; the first and last points are NaN.
    """
    R = np.asarray(R, float)
    one = R.ndim == 1
    R = np.atleast_2d(R)
    wl = np.asarray(wl, float); check_grid(wl)
    if norm_wl is not None:
        ref = value_at(wl, R, float(norm_wl))
        ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.nan)
        R = R / ref[:, None]
    h1 = np.diff(wl)[:-1]; h2 = np.diff(wl)[1:]
    D = np.full_like(R, np.nan)
    D[:, 1:-1] = 2.0 * ((R[:, 2:] - R[:, 1:-1]) / h2 - (R[:, 1:-1] - R[:, :-2]) / h1) / (h1 + h2)
    return D[0] if one else D


def d2_at(R, wl, lam, norm_wl=None):
    """D² evaluated at `lam` by linear interpolation between grid points (vectorised)."""
    return value_at(wl, second_derivative(R, wl, norm_wl=norm_wl), lam)


# ── phycocyanin trough depth ───────────────────────────────────────────────────

def pc_trough_depth(R, wl, wls=TROUGH_WLS, snap=None):
    """
    Relative depth of the phycocyanin absorption trough (EXPERIMENTAL — not used for detection).

        T = (R_base(λc) − R(λc)) / R_base(λc),   R_base = straight line R(λlo) → R(λhi)

    T increases with the cyanobacteria share. T > 0 does NOT by itself mean
    cyanobacteria (water absorption gives a background of ≈ 0.07 on CHIME-like data):
    the decision threshold must be calibrated per sensor spectral response.

    snap : use the nearest grid bands instead of interpolated values.
           None → automatic (snap when Δλ > 2.5 nm).
    """
    R = np.atleast_2d(np.asarray(R, float))
    d = check_grid(wl)
    lo, lc, hi = wls
    if snap is None:
        snap = d > DESIGN_STEP
    if snap:
        lo, lc, hi = (nearest_band(wl, x) for x in (lo, lc, hi))
    r_lo, r_c, r_hi = (value_at(wl, R, x) for x in (lo, lc, hi))
    w = (lc - lo) / (hi - lo)
    base = r_lo * (1.0 - w) + r_hi * w
    ok = np.isfinite(base) & np.isfinite(r_c) & (base > 0) & (r_c > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        T = np.where(ok, (base - r_c) / base, np.nan)
    return T if T.size > 1 else float(T[0])


# ── all features in one call ───────────────────────────────────────────────────

def cyano_features(R, wl, smooth_fwhm=SMOOTH_FWHM, native_fwhm=None, norm_wl=None,
                   d2_wl=D2_WL_PC, d2_wl_legacy=D2_WL_LEGACY, trough_wls=TROUGH_WLS):
    """
    Compute D2_552 (legacy), D2_620 (detection feature) and PC_T (experimental) for every spectrum.

    The trough depth is computed on the NATIVE bands (after the same smoothing), so
    that coarse sensors use exactly the band triplet nearest to trough_wls.

    Returns
    -------
    dict of 1-D arrays {'D2_552', 'D2_620', 'PC_T'} and the processing info dict.
    """
    Rp, wlp, info = prepare_spectra(R, wl, smooth_fwhm=smooth_fwhm, native_fwhm=native_fwhm)
    feats = {
        'D2_552': d2_at(Rp, wlp, d2_wl_legacy, norm_wl=norm_wl),
        'D2_620': d2_at(Rp, wlp, d2_wl, norm_wl=norm_wl),
        'PC_T':   np.atleast_1d(pc_trough_depth(
                      smooth_gaussian_nm(np.atleast_2d(R), wl, smooth_fwhm, native_fwhm=native_fwhm),
                      wl, wls=trough_wls)),
    }
    info.update(d2_wl=d2_wl, d2_wl_legacy=d2_wl_legacy, trough_wls=tuple(trough_wls),
                d2_bracket=bracketing_bands(wlp, d2_wl),
                d2_legacy_bracket=bracketing_bands(wlp, d2_wl_legacy))
    return feats, info


def cy_positive(name, values, threshold=None):
    """
    Boolean Cy+ decision for one feature (NaN → False).
    PC_T needs an explicit `threshold` (sensor-specific); returns None otherwise.
    """
    v = np.asarray(values, float)
    if name == 'D2_552':
        return np.isfinite(v) & (v < 0)
    if name == 'D2_620':
        return np.isfinite(v) & (v > 0)
    if name == 'PC_T':
        return None if threshold is None else (np.isfinite(v) & (v >= threshold))
    raise KeyError(name)
