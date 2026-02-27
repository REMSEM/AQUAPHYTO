"""
phytospec  —  spectral processing package for PANTHYR radiometer data
====================================================================
"""

from phytospec import config
from phytospec.io         import read_raw_csv, save_datacube, load_datacube
from phytospec.algorithms import compute_MALH, compute_CHL, compute_D2, lubac_phaeo_index
from phytospec.qc         import qc_datacube, qc_dataset
from phytospec.dataset    import build_datacube, make_dataset

__all__ = [
    "config",
    "read_raw_csv", "save_datacube", "load_datacube",
    "compute_MALH", "compute_CHL", "compute_D2", "lubac_phaeo_index",
    "qc_datacube", "qc_dataset",
    "build_datacube", "make_dataset",
]