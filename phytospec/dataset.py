"""
phytospec/dataset.py
==================
High-level orchestration: raw files → datacube → ML-ready dataset.
These are the functions you call from Jupyter notebooks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from phytospec.io         import read_raw_csv, find_raw_files, save_datacube
from phytospec.algorithms import compute_MALH, compute_CHL, compute_D2, lubac_phaeo_index
from phytospec.qc         import qc_datacube, qc_dataset
from phytospec import config as cfg


# ── Step 1 ─────────────────────────────────────────────────────────────────────

def build_datacube(datarep: Union[str, Path],
                   save_path: Union[str, Path] = None) -> dict:
    """
    Read all *QA_data.csv files in `datarep` and build a quality-controlled
    datacube.

    Parameters
    ----------
    datarep   : folder containing raw CSV files (searched recursively)
    save_path : optional .npz path to persist the datacube

    Returns
    -------
    dict with keys: RHOW (N x wl), wl, SZA, Ld750, Ed750, dateC, dateP, AZI
    """
    datarep = Path(datarep)
    files   = find_raw_files(datarep)
    if not files:
        raise FileNotFoundError(f"No *QA_data.csv files found in {datarep}")

    print(f"[dataset] Building datacube from {len(files)} files ...")

    # read first file to get reference wavelength grid
    ref    = read_raw_csv(files[0])
    ref_wl = ref["wl"]
    n_wl   = len(ref_wl)
    N      = len(files)

    RHOW  = np.full((N, n_wl), np.nan)
    SZA   = np.full(N, np.nan)
    Ld750 = np.full(N, np.nan)
    Ed750 = np.full(N, np.nan)
    dateC, dateP, AZI = [], [], []

    for i, f in enumerate(files):
        row = read_raw_csv(f, ref_wl=ref_wl)
        RHOW[i, :]  = row["rhow"]
        SZA[i]      = row["SZA"]
        Ld750[i]    = row["Ld750"]
        Ed750[i]    = row["Ed750"]
        dateC.append(row["dateC"])
        dateP.append(row["dateP"])
        AZI.append(row["AZI"])

    # set negatives to NaN before QC
    RHOW[RHOW < 0] = np.nan

    datacube = dict(RHOW=RHOW, wl=ref_wl, SZA=SZA,
                    Ld750=Ld750, Ed750=Ed750,
                    dateC=dateC, dateP=dateP, AZI=AZI)

    keep, _ = qc_datacube(datacube)
    datacube = _filter_datacube(datacube, keep)

    if save_path is not None:
        save_datacube(datacube, save_path)

    return datacube


# ── Step 2 ─────────────────────────────────────────────────────────────────────

def make_dataset(datacube: dict,
                 one_per_day:   bool  = True,
                 apply_chl_qc:  bool  = False,
                 noon_max_offset_hours: float = None,
                 save_path: Union[str, Path] = None) -> pd.DataFrame:
    """
    Compute spectral indices, apply QC, and build the final ML dataset.

    Parameters
    ----------
    datacube      : output of build_datacube() or load_datacube()
    one_per_day   : keep one spectrum per calendar day
                    (noon window 09-15 UTC preferred, else highest MALH of day)
    apply_chl_qc  : if True, remove spectra with CHL <= cfg.QC_CHL_MIN (3 mg/m3).
                    Default False — keeps all valid spectra so the ML dataset
                    covers the full season including low-biomass periods.
                    CHL is still computed and stored as a column.
    noon_max_offset_hours  : maximum distance (hours) from noon for the selected
                spectrum. Days with no spectrum within this window
                are dropped entirely.
                None (default) → keep closest, no exclusion.
                3.0 → matches colleague R pipeline exactly.
    save_path     : optional CSV path for the output dataset

    Returns
    -------
    pd.DataFrame with columns:
        date, CHL, MALH, P_LUB, rhow_355.0 ... rhow_945.0,
        D2rhow_355.0 ... D2rhow_945.0
    """
    RHOW  = datacube["RHOW"]
    wl    = datacube["wl"]
    dateP = datacube["dateP"]
    SZA   = datacube["SZA"]
    N     = len(dateP)

    print(f"[dataset] Computing spectral indices for {N} spectra ...")

    CHL  = np.full(N, np.nan)
    MALH = np.full(N, np.nan)
    PLUB = np.zeros(N, dtype=int)
    D2R  = np.full((N, len(wl)), np.nan)

    for i in range(N):
        rhow_i = RHOW[i, :]
        if np.sum(~np.isnan(rhow_i)) < 50:
            continue
        CHL[i]    = compute_CHL(rhow_i, wl)
        MALH[i]   = compute_MALH(rhow_i, wl)
        D2R[i, :] = compute_D2(rhow_i, wl)
        PLUB[i]   = lubac_phaeo_index(wl, D2R[i, :])

    # QC: always remove bad geometry (SZA / NIR), optionally CHL
    keep, _ = qc_dataset(RHOW, CHL, SZA, wl, apply_chl_qc=apply_chl_qc)

    n_removed = N - keep.sum()
    print(f"[dataset] QC removed {n_removed} spectra, {keep.sum()} retained.")

    if keep.sum() == 0:
        raise RuntimeError(
            "All spectra were removed by QC. "
            "Check your data or set apply_chl_qc=False."
        )

    CHL_k   = CHL[keep]
    MALH_k  = MALH[keep]
    PLUB_k  = PLUB[keep]
    D2R_k   = D2R[keep, :]
    RHOW_k  = RHOW[keep, :]
    dateP_k = [dateP[i] for i in range(N) if keep[i]]

    rhow_cols  = [f"rhow_{w:.1f}"   for w in wl]
    d2row_cols = [f"D2rhow_{w:.1f}" for w in wl]

    df = pd.DataFrame({
        "datetime": dateP_k,
        "CHL":      CHL_k,
        "MALH":     MALH_k,
        "P_LUB":    PLUB_k,
    })
    df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(RHOW_k,  columns=rhow_cols).reset_index(drop=True),
        pd.DataFrame(D2R_k,   columns=d2row_cols).reset_index(drop=True),
    ], axis=1)

    if one_per_day:
        df = _select_one_per_day(df, max_offset_hours=noon_max_offset_hours)

    df.insert(0, "date", df["datetime"].dt.date)
    df = df.drop(columns=["datetime"]).reset_index(drop=True)

    n_valid_chl = (~df["CHL"].isna()).sum()
    print(f"[dataset] Final dataset: {len(df)} days x {len(df.columns)} columns")
    print(f"          CHL valid (>0): {n_valid_chl} / {len(df)} days")
    print(f"          P. globosa days (P_LUB=1): {(df['P_LUB'] == 1).sum()}")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, float_format="%.8g")
        print(f"[dataset] Saved -> {save_path}")

    return df


# ── helpers ────────────────────────────────────────────────────────────────────

def _filter_datacube(dc: dict, keep: np.ndarray) -> dict:
    """Apply a boolean keep-mask to all arrays in the datacube."""
    dc["RHOW"]  = dc["RHOW"][keep, :]
    dc["SZA"]   = dc["SZA"][keep]
    dc["Ld750"] = dc["Ld750"][keep]
    dc["Ed750"] = dc["Ed750"][keep]
    dc["dateC"] = [v for v, k in zip(dc["dateC"], keep) if k]
    dc["dateP"] = [v for v, k in zip(dc["dateP"], keep) if k]
    dc["AZI"]   = [v for v, k in zip(dc["AZI"],   keep) if k]
    return dc


def _select_one_per_day(df: pd.DataFrame,
                         noon_target: str = "12:00",
                         max_offset_hours: float = None) -> pd.DataFrame:
    """
    Keep one spectrum per calendar day — closest to solar noon.

    Parameters
    ----------
    noon_target       : target time "HH:MM" in UTC (default "12:00")
    max_offset_hours  : if set, days where the closest spectrum is farther
                        than this many hours from noon are dropped entirely.
                        e.g. 3.0 → only keep days with data between 09:00–15:00.
                        None (default) → always keep the closest, no exclusion.
    """
    df = df.copy()
    dt_col = pd.to_datetime(df["datetime"])
    df["_date"] = dt_col.dt.date
    df["_noon_offset"] = dt_col.apply(
        lambda t: abs(t.hour * 60 + t.minute -
                      int(noon_target.split(":")[0]) * 60 -
                      int(noon_target.split(":")[1]))
    )

    rows = []
    for _, grp in df.groupby("_date"):
        best_idx  = grp["_noon_offset"].idxmin()
        best_mins = grp.loc[best_idx, "_noon_offset"]

        # Drop the day if it is too far from noon
        if max_offset_hours is not None:
            if best_mins > max_offset_hours * 60:
                continue          # skip this date entirely

        rows.append(grp.loc[best_idx])

    if not rows:
        return df.drop(columns=["_date", "_noon_offset"]).iloc[0:0]

    result = pd.DataFrame(rows)
    result = result.drop(columns=["_date", "_noon_offset"])
    return result.reset_index(drop=True)