from astropy.table import Table
import numpy as np
from typing import Any, Optional
from plotting import add_plot
from abc import ABC, abstractproperty, abstractmethod


class LC(ABC, Table):
    @abstractproperty
    def r(self):
        return self

    @abstractproperty
    def g(self):
        return self

    @abstractproperty
    def i(self):
        return self

    @abstractmethod
    def clean_lc(self):
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

    def photometric(self, scisigpix_cutoff=6):
        return self[self["scisigpix"] >= scisigpix_cutoff]

    def remove_bad_seeing(self, seeing_limit=7):
        return self[self["sciinpseeing"] <= seeing_limit]

    def remove_badpixels(self):
        return self[self["procstatus"] != 56]

    def clean_lc(self, good_seeing: bool = True, good_pixels: bool = True, photometric: bool = False) -> LC:
        """Clean the lightcurve"""
        self = self.remove_badpixels() if good_pixels else self
        self = self.remove_bad_seeing() if good_seeing else self
        self = self.photometric() if photometric else self
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
