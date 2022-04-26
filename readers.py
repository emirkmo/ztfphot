from astropy.table import Table
from lightcurve import LC
import numpy as np


def get_ztf_header(filename):
    """
    Parse and return header from ZTFPS file
    """
    with open(filename, "r") as infile:
        lines = np.asarray(infile.readlines())

    lines_without_comment = lines[[not line.startswith("#") for line in lines]]
    header = lines_without_comment[0]
    header = header.replace(",", "").strip()
    return header


def to_numeric(column, nan_reprs=["null"], convert_to_nan=True, output_dtype=float):
    if convert_to_nan:
        for nan_repr in nan_reprs:
            column[column == nan_repr] = np.nan
    return column.astype(output_dtype)


def convert_to_numeric(at: LC) -> LC:
    # Convert some important string columns to numeric
    force_cols = [
        "forcediffimflux",
        "forcediffimfluxunc",
        "forcediffimsnr",
        "forcediffimchisq",
        "forcediffimfluxap",
        "forcediffimfluxuncap",
        "forcediffimsnrap",
        "aperturecorr",
    ]
    for col in force_cols:
        at[col] = to_numeric(at[col])
    return at


def read_ztf_lc(filepath: str) -> LC:
    # Get the data
    header = get_ztf_header(filepath)
    at = Table.read(
        filepath,
        header_start=0,
        data_start=1,
        format="ascii",
        comment="#",
        names=header.split(" "),
    )

    at = convert_to_numeric(at)

    return at
