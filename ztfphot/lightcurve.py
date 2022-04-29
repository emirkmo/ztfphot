from typing import Optional

from astropy.table import Table
import numpy as np
from ztfphot.plotting import add_plot
from abc import ABC, abstractmethod


class LC(ABC, Table):

    @property
    def r(self):
        return self

    @property
    def g(self):
        return self

    @property
    def i(self):
        return self

    @abstractmethod
    def photometric(self):
        pass

    @abstractmethod
    def remove_bad_seeing(self):
        pass

    @abstractmethod
    def remove_bad_pixels(self):
        pass

    @abstractmethod
    def clean_lc(self):
        pass

    @abstractmethod
    def get_mag_lc(self):
        pass

    @abstractmethod
    def rescale_uncertainty(self):
        pass


@add_plot
class ZTF_LC(LC):

    def __getattr__(self, key):
        if key in self.colnames:
            return self[key]
        else:
            raise AttributeError(f"`Table` object has no attribute {key}")

    @property
    def r(self) -> LC:
        return self[self["filter"] == "ZTF_r"]

    @property
    def g(self) -> LC:
        return self[self["filter"] == "ZTF_g"]

    @property
    def i(self) -> LC:
        return self[self["filter"] == "ZTF_i"]

    def photometric(self, scisigpix_cutoff: float = 6) -> LC:
        return self[self["scisigpix"] <= scisigpix_cutoff]

    def remove_bad_seeing(self, seeing_limit: float = 7) -> LC:
        return self[self["sciinpseeing"] <= seeing_limit]

    def remove_bad_pixels(self, proc_statuses: list[int] = [56, 63, 64]) -> LC:
        """Remove bad pixels from the lightcurve"""
        return self[~np.isin(self["procstatus"], proc_statuses)]

    def clean_lc(self, seeing_limit: Optional[float] = None,
                 proc_statuses: Optional[list[int]] = None,
                 scisigpix_cutoff: Optional[float | int] = None) -> LC:
        """Clean the lightcurve"""
        self = self.remove_bad_pixels(proc_statuses) if proc_statuses else self.remove_bad_pixels()
        self = self.remove_bad_seeing(seeing_limit) if seeing_limit else self.remove_bad_seeing()
        self = self.photometric(scisigpix_cutoff) if scisigpix_cutoff else self.photometric()
        return self

    def rescale_uncertainty(self):
        """Rescale uncertainty using chisq"""
        self["forcediffimfluxunc"] = self["forcediffimfluxunc"] * np.sqrt(self["forcediffimchisq"])

    def get_mag_lc(self):
        """Convert flux to AB magnitude"""
        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        errcol = "fluxerr_corr" if "fluxerr_corr" in self.colnames else "forcediffimfluxunc"

        self["mag"] = self["zpdiff"] - 2.5 * np.log10(self[fluxcol])
        self["mag_err"] = 1.0857 * self[errcol] / self[fluxcol]

        self["islimit"] = self[fluxcol] / self[errcol] < 3
        self["lim"] = self["zpdiff"] - 2.5 * np.log10(5 * self[errcol])

    def simple_median_baseline(self, jd_min, jd_max):
        """Use simple median between to JDs for baseline flux"""
        return np.median(self["forcediffimflux"][(self["jd"] >= jd_min) & (self["jd"] <= jd_max)])

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






