from typing import Optional, Protocol, TypeVar, Generic

from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
import numpy as np
from ztfphot.plotting import add_plot
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy.typing as npt
import pandas as pd
from strictly_typed_pandas.dataset import DataSet
from .schemas import (Numeric, Schemas, VerCorrSchema, to_pandas, InputSchema,
 VerifiedRefSchema, CorrectedRefSchema, MagSchema)


LCType = TypeVar("LCType", bound="LC")

class LC(ABC, Table):

    @property
    def r(self: LCType) -> LCType:
        return self

    @property
    def g(self: LCType) -> LCType:
        return self

    @property
    def i(self: LCType) -> LCType:
        return self

    @abstractmethod
    def photometric(self: LCType, scisigpix_cutoff: float = 999) -> LCType:
        pass

    @abstractmethod
    def remove_bad_seeing(self: LCType, seeing_limit: float = 999) -> LCType:
        pass

    @abstractmethod
    def remove_bad_pixels(self: LCType, proc_statuses: list[int] = [0, 1]) -> LCType:
        pass

    @abstractmethod
    def clean_lc(self: LCType) -> LCType:
        pass

    @abstractmethod
    def get_mag_lc(self) -> None:
        pass

    @abstractmethod
    def rescale_uncertainty(self) -> None:
        pass

@dataclass
class ForcedPhotTable:
    """
    Class to hold a table of forced photometry.
    """
    zp: npt.NDArray[np.float_]


@add_plot
class ZtfPhot:

    def __init__(self, input_table: DataSet[InputSchema]):
        self.df = input_table
        self.schema = InputSchema

    def update_schema(self) -> None:
        if "flux_ver" in self.colnames and "flux_corr" in self.colnames:
            self.schema = VerCorrSchema
        elif "flux_ver" in self.colnames:
            self.schema = VerifiedRefSchema
        elif "flux_corr" in self.colnames:
            self.schema = CorrectedRefSchema
        else:
            self.schema = InputSchema

    @property
    def colnames(self) -> list[str]:
        return self.df.columns

    def _select_flux_col(self) -> str:
        """Select the flux column to use"""
        if "flux_ver" in self.colnames:
            return "flux_ver"
        if "flux_corr" in self.colnames:
            return "flux_corr"
        elif "forcediffimflux" in self.colnames:
            return "forcediffimflux"
        else:
            raise ValueError("No flux column found")

    def _select_err_col(self) -> str:
        """Select the flux column to use"""
        if "fluxerr_ver" in self.colnames:
            return "fluxerr_ver"
        if "fluxerr_corr" in self.colnames:
            return "fluxerr_corr"
        elif "forcediffimfluxunc" in self.colnames:
            return "forcediffimfluxunc"
        else:
            raise ValueError("No error column found")
        


    @property
    def r(self) -> DataSet:
        return DataSet(self.df[self.df["filter"] == "ZTF_r"])

    @property
    def g(self) -> DataSet:
        return DataSet(self.df[self.df["filter"] == "ZTF_g"])

    @property
    def i(self) -> DataSet:
        return DataSet(self.df[self.df["filter"] == "ZTF_i"])

    def estimate_peak_jd(self, bin_size: int = 5) -> float:
        """Estimate peak flux from lightcurve"""
        fluxcol = self._select_flux_col()
        interpjd = np.convolve(self.df["jd"], np.ones(bin_size) / bin_size, 'valid')
        interpf = np.convolve(self.df[fluxcol], np.ones(bin_size) / bin_size, 'valid')
        return interpjd[np.nanargmax(interpf)]

    def photometric(self, scisigpix_cutoff: float = 6) -> DataSet:
        return DataSet(self.df[self.df["scisigpix"] <= scisigpix_cutoff])

    def remove_bad_seeing(self, seeing_limit: float = 7) -> DataSet:
        return DataSet(self.df[self.df["sciinpseeing"] <= seeing_limit])

    def remove_bad_pixels(self, proc_statuses: list[int] = [56, 63, 64]) -> DataSet:
        """Remove bad pixels from the lightcurve"""
        return DataSet(self.df[~np.isin(self.df["procstatus"], proc_statuses)])

    def clean_lc(self, seeing_limit: Optional[float] = None,
                 proc_statuses: Optional[list[int]] = None,
                 scisigpix_cutoff: Optional[float | int] = None) -> DataSet:
        """Clean the lightcurve"""
        self.df = self.remove_bad_pixels(proc_statuses) if proc_statuses else self.remove_bad_pixels()
        self.df = self.remove_bad_seeing(seeing_limit) if seeing_limit else self.remove_bad_seeing()
        self.df = self.photometric(scisigpix_cutoff) if scisigpix_cutoff else self.photometric()
        return DataSet(self.df)

    def remove_non_psf_nearest_ref(self, sharp_hi=0.8, sharp_lo=-0.5, nearest_chi_limit: Optional[float] = None) -> DataSet:
        """
        Remove epochs without a suitable PSF nearest reference.
        nearestrefsharp is a measure of the PSF of the nearest reference. This cleaning
        should only be done if this value is around 0.
        Closer to 1 means galaxy/extended-psf, closer to -1 means spike/sharp edge.
        """

        self.df = self.df[(self.df.nearestrefsharp <= sharp_hi) & (self.df.nearestrefsharp >= sharp_lo)]
        if nearest_chi_limit is not None:
            self.df = self.df[self.df.nearestrefchi <= nearest_chi_limit]
        return DataSet(self.df)

    def correct_flux_using_nearest_ref(self) -> DataSet[VerCorrSchema] | DataSet[CorrectedRefSchema]:
        """
        This is used to correct for reference images contaminated by transient flux.
        Correct flux using mag of nearest reference, but this should only be done if nearestrefsharp
        is around 0. Closer to 1 means galaxy, closer to -1 means spike/sharp edge.
        """
        fluxcol = "flux_ver" if "flux_ver" in self.colnames else "forcediffimflux"
        errcol = "fluxerr_ver" if "fluxerr_ver" in self.colnames else "forcediffimfluxunc"

        nearestrefflux = 10 ** (0.4 * (self.df.zpdiff - self.df.nearestrefmag))
        # self['flux_corr'] = self[fluxcol] + nearestrefflux
        corrected_flux = self.df[fluxcol] + nearestrefflux

        nearestreffluxunc = (self.df.nearestrefmagunc * nearestrefflux) / 1.0857
 
        corrected_err = np.sqrt(self.df[errcol] ** 2 - nearestreffluxunc ** 2)

        schema = DataSet[VerCorrSchema] if "ver" in fluxcol else DataSet[CorrectedRefSchema]
        return self.df.assign(flux_corr=corrected_flux, fluxerr_corr=corrected_err).pipe(schema)


    def rescale_uncertainty(self):
        """Rescale uncertainty using chisq"""
        self["forcediffimfluxunc"] = self["forcediffimfluxunc"] * np.sqrt(self["forcediffimchisq"])

    def get_mag_lc(self, snr: int = 3, snt: int = 5):
        """Convert flux to AB magnitude"""
        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        errcol = "fluxerr_corr" if "fluxerr_corr" in self.colnames else "forcediffimfluxunc"

        snr_tot = self[fluxcol]/self[errcol]
        self["mag"] = self["zpdiff"] - 2.5 * np.log10(self[fluxcol])
        self["mag_err"] = 1.0857 / snr_tot

        self["islimit"] = snr_tot < snr
        self["lim"] = self["zpdiff"] - 2.5 * np.log10(snt * self[errcol])

    def simple_median_baseline(self, jd_min, jd_max):
        """Use simple median between to JDs for baseline flux"""
        return np.nanmedian(self["forcediffimflux"][(self["jd"] >= jd_min) & (self["jd"] <= jd_max)])

    def RMS_baseline(self, jd_min, jd_max):
        """Use RMS between to JDs for baseline flux"""
        from scipy.stats import trim_mean

        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        SNR = self[fluxcol] / self["forcediffimfluxunc"]
        SNR_baseline = SNR[(self["jd"] >= jd_min) & (self["jd"] <= jd_max)]

        return np.sqrt(trim_mean(np.square(SNR_baseline), 0.16))  # Return robust RMS using trimmed mean.



        """Rescale uncertainty using chisq"""
        self["forcediffimfluxunc"] = self["forcediffimfluxunc"] * np.sqrt(self["forcediffimchisq"])

    def get_mag_lc(self, snr: int = 3, snt: int = 5):
        """Convert flux to AB magnitude"""
        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        errcol = "fluxerr_corr" if "fluxerr_corr" in self.colnames else "forcediffimfluxunc"

        snr_tot = self[fluxcol]/self[errcol]
        self["mag"] = self["zpdiff"] - 2.5 * np.log10(self[fluxcol])
        self["mag_err"] = 1.0857 / snr_tot

        self["islimit"] = snr_tot < snr
        self["lim"] = self["zpdiff"] - 2.5 * np.log10(snt * self[errcol])

    def simple_median_baseline(self, jd_min, jd_max):
        """Use simple median between to JDs for baseline flux"""
        return np.nanmedian(self["forcediffimflux"][(self["jd"] >= jd_min) & (self["jd"] <= jd_max)])

    def RMS_baseline(self, jd_min, jd_max):
        """Use RMS between to JDs for baseline flux"""
        from scipy.stats import trim_mean

        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        SNR = self[fluxcol] / self["forcediffimfluxunc"]
        SNR_baseline = SNR[(self["jd"] >= jd_min) & (self["jd"] <= jd_max)]

        return np.sqrt(trim_mean(np.square(SNR_baseline), 0.16))  # Return robust RMS using trimmed mean.



def verify_reference(at: LC, jd_first: float) -> LC:
    """
    Verify that references do not contain epochs from after jd_first.
    jd_first: first relevant epoch for object
    at: ZTFFPS table
    return:
        ZTFFPS table with only references using data previous to the object being there.
    """

    only_pre = at[at["refjdend"] <= jd_first]

    if len(at) - len(only_pre) == 0:
        return at

    print(
        "Returning only pre: you will need to perform a baseline correction to use the full data!"
        "See: https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_forced_photometry.pdf"
    )
    return only_pre


def calculate_mag_lc():
    raise NotImplementedError()






