from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
from ztfphot.plotting import PlotType
from ztfphot.lightcurve import ZTF_LC, LC, verify_reference
from ztfphot.readers import read_ztf_lc
from ztfphot.sn import SN


INPUTDIR = "tests/input/"
OUTPUTDIR = "tests/output/"


def make_ztf_lc(sn: SN, verify_references: bool = False) -> ZTF_LC:
    fpath = sn.filename
    jd_first = sn.jd_first_epoch
    at = read_ztf_lc(fpath)
    if jd_first is not None and verify_references:
        at = verify_reference(at, jd_first)  # Remove references contaminated by the object

    lc = ZTF_LC(at)
    lc.clean_lc(scisigpix_cutoff=25)
    return lc


def plot_and_save_lc(sn: SN, lc: LC):

    bands = [lc.g, lc.r, lc.i]

    for band in bands:
        band["flux_corr"] = band["forcediffimflux"] - band.simple_median_baseline(sn.jd_min, sn.jd_max)  # type: ignore
        band.rescale_uncertainty()  # type: ignore
        band.RMS = band.RMS_baseline(sn.jd_min, sn.jd_max)  # type: ignore
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
        fig.savefig(f"{OUTPUTDIR}{sn.snname}_{band['filter'][0]}.png")
        band.write(f"{OUTPUTDIR}{sn.snname}_ztffps_{band['filter'][0][-1]}band.ecsv", format="ascii.ecsv", overwrite=True)
    return lc


def main():

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
    lc = make_ztf_lc(SN2020lao, verify_references=True)
    sn2020lao_lc = plot_and_save_lc(SN2020lao, lc)
    plt.show(block=True)
    return lc


if __name__ == "__main__":
    lc = main()
