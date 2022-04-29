from io import StringIO

import matplotlib.pyplot as plt
import plotly.tools
import streamlit as st
from ztfphot import read_ztf_lc, ZTF_LC, LC, verify_reference, SN, PlotType
from ztfphot.readers import get_ztf_header
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ZTFphot: ZTF Forced Photometry lightcurve tool", page_icon=":telescope:", layout="wide")


st.title("ZTFphot")
st.write("Open source tool for making and plotting publication quality lightcurves from ZTF forced photometry")


@st.experimental_memo
def cached_read(uploaded_file) -> Table:
    with open('temp.txt', "wb") as outfile:
        outfile.write(uploaded_file.getvalue())
    return read_ztf_lc('temp.txt')


@st.experimental_memo
def cached_verify(_lc: LC, jd_first) -> LC:
    return verify_reference(_lc, jd_first)


snname = st.text_input("SN name: <SN2020xyz>")
uploaded_file = st.file_uploader('Choose a ZTFFPS file', type=['txt', 'csv'])
if uploaded_file:
    at = cached_read(uploaded_file)
    allbands = ZTF_LC(at)

    band = st.radio("ZTF Filter:", ("g", "r", "i"))
    lc = allbands[allbands["filter"] == f"ZTF_{band}"]

    if len(lc) <= 10:
        st.error("Not enough data points!")

    options = ["verify references", "photometric", "good seeing",
               "filter procstatus", "rescale uncertainty", "magnitude", "flux"]
    steps = st.multiselect(label="Select steps to perform",
                           options=options,
                           default=options[:-1],
                           help="""
                           Verify references: select only references without contamination
                           Photometric: select only scisigpix <= cutoff (default 6)""",
                           )

    if "verify references" in steps:
        jd_first = st.number_input(label="First epoch JD", min_value=lc.jd[0], max_value=lc.jd[-1], value=lc.jd[-1])
        lc = cached_verify(ZTF_LC(at), jd_first)





    if "photometric" in steps:
        scisigpix_cutoff = st.slider(label="scisigpix_cutoff", min_value=1, max_value=50, value=6, step=1)
        lc = lc.photometric(scisigpix_cutoff)

    if "good seeing" in steps:
        seeing_cutoff = st.slider(label="seeing_cutoff", min_value=0.9, max_value=10., value=7., step=0.1)
        lc = lc.remove_bad_seeing()

    if "filter procstatus" in steps:
        procstatus = st.multiselect(label="procstatus",
                                    options=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 255],
                                    default=[56, 63, 64],
                                    help="Select the filter procstatus to exclude"
                                    )
        lc = lc.remove_bad_pixels([int(p) for p in procstatus])

    if "verify references" in steps:
        jd_first = st.number_input(label="last JD of reference to include.",
                                   min_value=int(lc.jd[0]), max_value=int(lc.jd[-1]), value=int(lc.jd[0]),
                                   help='Everything before this JD will be allowed to be used in the reference.')
        lc = cached_verify(lc, jd_first)


    #for lc in [allbands.r, allbands.g, allbands.i]:
    st.write("Use the plot and sliders below to select the baseline time range")
    jd_min = st.slider("Baseline JD min", float(lc.jd[0]-1), float(lc.jd[-1]), value=float(lc.jd[0]), step=.1)
    jd_max = st.slider("Baseline JD max", float(lc.jd[0]), float(lc.jd[-1]+1), value=float(lc.jd[-1]), step=.1)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax = lc.plot_diff_flux(fig=fig, ax=ax, xmin=jd_min, xmax=jd_max)
    ax.axvline(jd_min)
    ax.axvline(jd_max)
    ax.set_xlim(jd_min-20, jd_max+20)
    st.pyplot(fig)

    st.write("---")
    if "rescale uncertainty" in steps:
        lc["flux_corr"] = lc["forcediffimflux"] - lc.simple_median_baseline(jd_min, jd_max)
        lc.rescale_uncertainty()
        lc.RMS = lc.RMS_baseline(jd_min, jd_max)
        if lc.RMS >= 1.01:
            lc["forcediffimfluxunc"] = lc["forcediffimfluxunc"] * lc.RMS

    if "magnitude" in steps:
        snr = st.slider(label="SNR Limit threshold", min_value=1, max_value=10, value=3, step=1)
        lc["limit"] = lc["flux_corr"] / lc["forcediffimfluxunc"] < snr
        lc.get_mag_lc()

        detections = lc[~lc.islimit]  # type: ignore
        limits = lc[lc.islimit]  # type: ignore

        fig2, ax2 = detections.plot("jd", "mag", yerr="mag_err", kind=PlotType.errorbar)
        limits.plot("jd", "lim", kind=PlotType.scatter, ax=ax2, plot_kwargs={'marker': 'v'})

    st.pyplot(fig2)

    if "flux" in steps:
        st.error("Not implemented yet")


    #verify_references = st.button("verify_references")
    #if verify_references:

    #    lc = cached_verify(lc, None)


    #bytes_data = uploaded_file.getvalue()
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #lines = np.asarray(stringio.readlines())
    #lines_without_comment = lines[[not line.startswith("#") for line in lines]]
    #st.write(lines_without_comment)
    #header = lines_without_comment[0]
    #header = header.replace(",", "").strip()
    #header = get_ztf_header(uploaded_file)