from typing import Optional
from strictly_typed_pandas.dataset import DataSet
import pandas as pd
import numpy as np
from astropy.table import Table

Numeric = int | float | np.number

class InputSchema:
    zpdiff: Numeric
    forcediffimflux: Numeric
    forcediffimfluxunc: Numeric
    forcediffimchisq: Numeric
    refjdend: Numeric
    jd: Numeric
    nearestrefsharp: Numeric
    nearestrefchi: Numeric
    nearestrefmag: Numeric
    filter: str
    procstatus: int
    sciinpseeing: Numeric
    scisigpix: Numeric

class VerifiedRefSchema(InputSchema):
    flux_ver: Numeric
    fluxerr_ber: Numeric
    snr_ver: Optional[Numeric]

class CorrectedRefSchema(InputSchema):
    flux_corr: Numeric
    fluxerr_corr: Numeric
    snr_corr: Optional[Numeric]

class VerCorrSchema(CorrectedRefSchema, VerifiedRefSchema):
    pass


class MagSchema:
    mag: Numeric
    magerr: Numeric
    mag_corr: Numeric
    magerr_corr: Numeric
    mag_ver: Numeric
    magerr_ver: Numeric

def to_pandas(lc: Table) -> DataSet[InputSchema]:
    return lc.to_pandas()
