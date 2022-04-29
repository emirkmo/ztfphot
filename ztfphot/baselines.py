import numpy as np
from typing import Any
from astropy.table import Table


def simple_median_baseline(flux: Table.Column, jd: Table.Column, jd_min: float, jd_max: float) -> float | Any:
    """Use simple median between to JDs for baseline flux"""
    return np.median(flux[(jd >= jd_min) & (jd <= jd_max)])


def rms_baseline(
    flux: Table.Column, fluxerr: Table.Column, jd: Table.Column, jd_min: float, jd_max: float
) -> float | Any:
    """Use RMS between to JDs for baseline flux"""
    from scipy.stats import trim_mean

    SNR = flux / fluxerr
    SNR_baseline = SNR[(jd >= jd_min) & (jd <= jd_max)]
    return np.sqrt(trim_mean(np.square(SNR_baseline), 0.16))  # Return robust RMS using trimmed mean.
