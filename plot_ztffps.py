from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from plotting import PlotType
from lightcurve import ZTF_LC, verify_reference
from readers import read_ztf_lc

INPUTDIR = "tests/input/"
OUTPUTDIR = "tests/output/"


@dataclass
class SN:
    jd_first_epoch: float
    jd_min: float
    jd_max: float
    snname: str
    filename: str


def make_ztf_lc(fpath: str, jd_first: Optional[float] = None, verify_references: bool = False) -> ZTF_LC:
    at = read_ztf_lc(fpath)
    if jd_first is not None and verify_references:
        at = verify_reference(at, jd_first)  # Remove references contaminated by the object

    lc = ZTF_LC(at)
    lc.clean_lc(good_pixels=True, good_seeing=True)
    return lc


def plot_and_save_lc(jd_first_epoch: float, jd_min: float, jd_max: float, filename: str, snname: str,
    verify_references:bool = False):

    lc = make_ztf_lc(filename, jd_first_epoch, verify_references=verify_references)
    bands = [lc.g, lc.r, lc.i]

    for band in bands:
        band["flux_corr"] = band["forcediffimflux"] - band.simple_median_baseline(jd_min, jd_max)  # type: ignore
        band.rescale_uncertainty()  # type: ignore
        band.RMS = band.RMS_baseline(jd_min, jd_max)  # type: ignore
        if band.RMS >= 1.01:  # type: ignore
            band["forcediffimfluxunc"] = band["forcediffimfluxunc"] * band.RMS  # type: ignore

        band["limit"] = band["flux_corr"] / band["forcediffimfluxunc"] < 3
        band.get_mag_lc()  # type: ignore

        # Plot the data
        detections = band[~band.islimit]  # type: ignore
        limits = band[band.islimit]  # type: ignore
        fig, ax = detections.plot("jd", "mag", yerr="mag_err", kind=PlotType.errorbar)
        limits.plot("jd", "lim", kind=PlotType.scatter, ax=ax, plot_kwargs={'marker':'v'})
        plt.xlim(2458976, 2459120)
        plt.gca().invert_yaxis()
        fig.savefig(f"{OUTPUTDIR}{snname}_{band['filter'][0]}.png")
        band.write(f"{OUTPUTDIR}{snname}_ztffps_{band['filter'][0][-1]}band.ecsv", format="ascii.ecsv", overwrite=True)
    return lc


if __name__ == "__main__":
    # matplotlib.use('Qt5Agg')
    plt.ion()
    phases = pd.read_json(INPUTDIR + "phase_epochs.json", typ="Series")
    SN2020lao = SN(
        jd_first_epoch=phases.discovery - 2,
        jd_min=2458840.0,
        jd_max=2458989.0,
        snname="SN2020lao",
        filename=INPUTDIR + "forcedphotometry_req00104849_lc.txt",
    )

    SN2018ebt = SN(
        jd_first_epoch=2458323,
        jd_min=2458190,
        jd_max=2458313,
        snname="SN2018ebt",
        filename=INPUTDIR + "forcedphotometry_req00108452_lc.txt",
    )

    sn2020lao_lc = plot_and_save_lc(
        SN2020lao.jd_first_epoch, SN2020lao.jd_min, SN2020lao.jd_max, SN2020lao.filename, SN2020lao.snname, True
    )
    plt.show(block=True)
