"""
phytospec/io.py
=============
Reading raw PANTHYR CSV files and saving/loading processed datacubes.
"""

import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


# ── filename parser ────────────────────────────────────────────────────────────
_FNAME_PATTERN = re.compile(r"L2A_REF_(\d{8}T\d{6})_(\d+)_")


def _parse_filename(fname: str) -> tuple[str, str, pd.Timestamp]:
    """
    Extract datetime string, azimuth and pd.Timestamp from a PANTHYR filename.
    Example: PANTHYR_W_O1BE_L2A_REF_20250201T082153_270_...
    Returns ('20250201T082153', '270', Timestamp) or ('', '', NaT) on failure.
    """
    m = _FNAME_PATTERN.search(fname)
    if not m:
        return "", "", pd.NaT
    ttt = m.group(1)
    ppp = m.group(2)
    ts  = pd.to_datetime(ttt, format="%Y%m%dT%H%M%S", utc=True)
    return ttt, ppp, ts


def read_raw_csv(filepath: Union[str, Path],
                 ref_wl: np.ndarray = None) -> dict:
    """
    Read a single PANTHYR QA_data.csv file.

    Parameters
    ----------
    filepath : path to one *QA_data.csv file
    ref_wl   : optional reference wavelength grid; if provided the spectrum
               is interpolated onto it (useful when grids differ slightly).

    Returns
    -------
    dict with keys: dateC, dateP, AZI, SZA, Ld750, Ed750, rhow, wl
    """
    filepath = Path(filepath)
    dat = pd.read_csv(filepath)

    wl_file = dat["wavelength"].values.astype(float)

    # prefer rhow_nosc (sky-glint corrected) over rhow
    col_rhow = "rhow_nosc" if "rhow_nosc" in dat.columns else "rhow"
    rhow_vals = dat[col_rhow].values.astype(float)

    dateC, AZI, dateP = _parse_filename(filepath.name)

    idx_750 = int(np.argmin(np.abs(wl_file - 750.0)))
    Ld750 = dat["lu"].iloc[idx_750]  if "lu" in dat.columns else np.nan
    Ed750 = dat["ed"].iloc[idx_750]  if "ed" in dat.columns else np.nan
    SZA   = dat["solar_zenith_angle"].iloc[0] if "solar_zenith_angle" in dat.columns else np.nan

    # interpolate onto reference grid if supplied
    if ref_wl is not None and not np.array_equal(wl_file, ref_wl):
        rhow_out = np.interp(ref_wl, wl_file, rhow_vals, left=np.nan, right=np.nan)
        wl_out   = ref_wl
    else:
        rhow_out = rhow_vals
        wl_out   = wl_file

    return {
        "dateC": dateC,
        "dateP": dateP,
        "AZI":   AZI,
        "SZA":   float(SZA),
        "Ld750": float(Ld750),
        "Ed750": float(Ed750),
        "rhow":  rhow_out,
        "wl":    wl_out,
    }


def find_raw_files(datarep: Union[str, Path]) -> list[Path]:
    """Return sorted list of all *QA_data.csv files found recursively."""
    files = sorted(Path(datarep).rglob("*QA_data.csv"))
    return files


# ── datacube persistence ───────────────────────────────────────────────────────

def save_datacube(datacube: dict, path: Union[str, Path]):
    """
    Save datacube to a compressed .npz file.
    dateP is stored as ISO-format strings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        RHOW  = datacube["RHOW"],
        wl    = datacube["wl"],
        SZA   = datacube["SZA"],
        Ld750 = datacube["Ld750"],
        Ed750 = datacube["Ed750"],
        dateC = np.array(datacube["dateC"], dtype=str),
        AZI   = np.array(datacube["AZI"],   dtype=str),
        dateP = np.array([str(d) for d in datacube["dateP"]], dtype=str),
    )
    print(f"[io] Datacube saved → {path}  ({datacube['RHOW'].shape[0]} spectra)")


def load_datacube(path: Union[str, Path]) -> dict:
    """
    Load a datacube previously saved by save_datacube().
    Returns dict with same structure as build_datacube().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datacube not found: {path}")
    npz = np.load(path, allow_pickle=False)
    dc  = {k: npz[k] for k in npz.files}
    dc["dateP"] = [pd.Timestamp(str(s)) for s in dc["dateP"]]
    dc["dateC"] = [str(s) for s in dc["dateC"]]
    dc["AZI"]   = [str(s) for s in dc["AZI"]]
    print(f"[io] Datacube loaded ← {path}  ({dc['RHOW'].shape[0]} spectra)")
    return dc