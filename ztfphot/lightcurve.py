from typing import Optional

from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
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
class ZtfLC(LC):

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

    def estimate_peak_jd(self, bin_size: int = 5) -> float:
        """Estimate peak flux from lightcurve"""
        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        interpjd = np.convolve(self["jd"], np.ones(bin_size) / bin_size, 'valid')
        interpf = np.convolve(self[fluxcol], np.ones(bin_size) / bin_size, 'valid')
        return interpjd[np.nanargmax(interpf)]

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

    def remove_non_psf_nearest_ref(self, sharp_hi=0.8, sharp_lo=-0.5, nearest_chi_limit: Optional[float] = None) -> LC:
        """
        Remove epochs without a suitable PSF nearest reference.
        nearestrefsharp is a measure of the PSF of the nearest reference. This cleaning
        should only be done if this value is around 0.
        Closer to 1 means galaxy/extended-psf, closer to -1 means spike/sharp edge.
        """

        self = self[(self.nearestrefsharp <= sharp_hi) & (self.nearestrefsharp >= sharp_lo)]
        if nearest_chi_limit is not None:
            self = self[self.nearestrefchi <= nearest_chi_limit]
        return self

    def correct_flux_using_nearest_ref(self):
        """
        This is used to correct for reference images contaminated by transient flux.
        Correct flux using mag of nearest reference, but this should only be done if nearestrefsharp
        is around 0. Closer to 1 means galaxy, closer to -1 means spike/sharp edge.
        """
        fluxcol = "flux_corr" if "flux_corr" in self.colnames else "forcediffimflux"
        errcol = "fluxerr_corr" if "fluxerr_corr" in self.colnames else "forcediffimfluxunc"

        nearestrefflux = 10 ** (0.4 * (self.zpdiff - self.nearestrefmag))
        # self['flux_corr'] = self[fluxcol] + nearestrefflux
        corrected_flux = self[fluxcol] + nearestrefflux

        nearestreffluxunc = (self.nearestrefmagunc * nearestrefflux) / 1.0857
        term_a = self[errcol] ** 2
        term_b = nearestreffluxunc ** 2
        #if np.all(term_a > term_b):
        #    # Because uncorrelated
        new_err = np.sqrt(self[errcol] ** 2 - nearestreffluxunc ** 2)
        #else:
        #    # More conservative estimate if assumptions not valid
        #    new_err = np.sqrt(self[errcol] ** 2 + nearestreffluxunc ** 2)
        #new_err =
        #self['fluxerr_corr'] = new_err
        return corrected_flux, new_err


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






