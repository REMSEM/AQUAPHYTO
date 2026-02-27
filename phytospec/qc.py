"""
phytospec/qc.py
=============
Quality control filters for PANTHYR datacubes and computed datasets.
All functions return a boolean mask (True = keep) and a log dict.
"""

import numpy as np
import pandas as pd
from phytospec import config as cfg
from phytospec.algorithms import interp_at


def qc_datacube(datacube: dict,
                ldex_max: float = cfg.QC_LDEX_MAX,
                nan_max:  int   = cfg.QC_NAN_MAX) -> tuple[np.ndarray, dict]:
    """
    QC on the raw datacube (before computing indices).

    Tests
    -----
    1. Ld/Ed ratio at 750 nm >= ldex_max  ->  sun glint / cloud contamination
    2. Number of NaN wavelengths >= nan_max  ->  bad acquisition

    Returns
    -------
    keep : bool array  (True = spectrum passes QC)
    log  : dict with counts for each test
    """
    N    = len(datacube["dateP"])
    keep = np.ones(N, dtype=bool)
    log  = {}

    # Test 1 - glint
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = datacube["Ld750"] / datacube["Ed750"]
    bad_glint = np.where(np.isnan(ratio), False, ratio >= ldex_max)
    keep     &= ~bad_glint
    log["glint_removed"] = int(bad_glint.sum())

    # Test 2 - NaN count
    n_nan    = np.sum(np.isnan(datacube["RHOW"]), axis=1)
    bad_nan  = n_nan >= nan_max
    keep    &= ~bad_nan
    log["nan_removed"] = int(bad_nan.sum())

    log["total_input"] = N
    log["total_kept"]  = int(keep.sum())

    _print_qc_summary("Datacube QC", log)
    return keep, log


def qc_dataset(RHOW:  np.ndarray,
               CHL:   np.ndarray,
               SZA:   np.ndarray,
               wl:    np.ndarray,
               sza_max:       float = cfg.QC_SZA_MAX,
               chl_min:       float = cfg.QC_CHL_MIN,
               apply_chl_qc:  bool  = False) -> tuple[np.ndarray, dict]:
    """
    QC on computed spectral indices.

    Tests
    -----
    1. R(900) > 0.5 x Rmax  OR  SZA > sza_max  ->  bad geometry / high turbidity
    2. CHL <= chl_min  (only applied when apply_chl_qc=True)
       The NIR-red CHL algorithm is only valid for bloom conditions (>3 mg/m3).
       Set apply_chl_qc=False to keep low-biomass spectra in the ML dataset.

    Returns
    -------
    keep : bool array
    log  : dict
    """
    N    = len(CHL)
    keep = np.ones(N, dtype=bool)
    log  = {}

    # Test 1 - NIR / SZA
    Rmax = np.nanmax(RHOW, axis=1)
    R900 = np.array([interp_at(wl, RHOW[i, :], 900.0) for i in range(N)])
    with np.errstate(invalid="ignore"):
        bad_nir = (R900 > 0.5 * Rmax) | (SZA > sza_max)
    bad_nir = np.where(np.isnan(bad_nir), False, bad_nir)
    keep   &= ~bad_nir
    log["nir_sza_removed"] = int(bad_nir.sum())

    # Test 2 - low Chl-a (optional)
    if apply_chl_qc:
        bad_chl = np.where(np.isnan(CHL), True, CHL <= chl_min)
        keep   &= ~bad_chl
        log["low_chl_removed"] = int(bad_chl.sum())
    else:
        log["low_chl_removed"] = 0  # skipped

    log["apply_chl_qc"] = apply_chl_qc
    log["total_input"]  = N
    log["total_kept"]   = int(keep.sum())

    _print_qc_summary("Dataset QC", log)
    return keep, log


def _print_qc_summary(title: str, log: dict):
    print(f"\n[qc] {title}")
    for k, v in log.items():
        print(f"     {k:25s}: {v}")