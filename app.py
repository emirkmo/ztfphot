import streamlit as st
from ztfphot import read_ztf_lc, ZTF_LC, LC, verify_reference, SN, PlotType
from astropy.table import Table
import altair as alt
import numpy as np
import pandas as pd

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

@st.experimental_memo
def cache_download(data: pd.DataFrame):
    return data.to_csv(index=False).encode('utf-8')


snname = st.text_input("SN name: <SN2020xyz>. (Currently only used for output filename).")
uploaded_file = st.file_uploader('Choose a ZTFFPS file', type=['txt', 'csv'])
if uploaded_file:
    at = cached_read(uploaded_file)
    allbands = ZTF_LC(at)

    band = st.radio("ZTF Filter:", ("g", "r", "i"))
    lc = allbands[allbands["filter"] == f"ZTF_{band}"]

    if len(lc) <= 10:
        st.error("Not enough data points!")

    options = ["verify references", "photometric", "good seeing",
               "filter procstatus", "rescale uncertainty"]
    steps = st.multiselect(label="Select steps to perform",
                           options=options,
                           default=options[:-1],
                           help="""
                           Verify references: select only references without contamination
                           Photometric: select only scisigpix <= cutoff (default 6)""",
                           )

    if "photometric" in steps:
        scisigpix_cutoff = st.slider(label="scisigpix_cutoff", min_value=1, max_value=50, value=6, step=1)
        lc = lc.photometric(scisigpix_cutoff)

    if "good seeing" in steps:
        seeing_cutoff = st.slider(label="seeing_cutoff", min_value=0.9, max_value=10., value=7., step=0.1)
        lc = lc.remove_bad_seeing(seeing_cutoff)

    if "filter procstatus" in steps:
        procstatus = st.multiselect(label="procstatus",
                                    options=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 255],
                                    default=[56, 63, 64],
                                    help="Select the filter procstatus to exclude"
                                    )
        lc = lc.remove_bad_pixels([int(p) for p in procstatus])

    jd_peak_est = lc.estimate_peak_jd()
    jd_disc_est = jd_peak_est - 25

    if "verify references" in steps:

        jd_first = st.slider(label="last JD of reference to include for reference cleaning.",
                                   min_value=float(np.round(lc.jd[0], 1)), max_value=float(np.round(lc.jd[-1], 1)),
                                   value=float(np.round(jd_disc_est, 1)), step=0.1,
                                   help='Everything before this JD will be allowed to be used in the reference.')
        #lc = cached_verify(lc, jd_first)
        lc = verify_reference(lc, jd_first)

    st.write("Use the plot and sliders below to select the baseline time range")
    # jd_min = st.slider("Baseline JD min", float(np.round(lc.jd[0]-1, 1)),  float(np.round(lc.jd[-1], 1)),
    #                    value=float(np.round(lc.jd[0], 1)), step=.1)
    # jd_max = st.slider("Baseline JD max",  float(np.round(lc.jd[0], 1)), float(np.round(lc.jd[-1]+1, 1)),
    #                    value=float(np.round(jd_disc_est, 1)), step=.1)

    jd_min, jd_max = st.slider(
        "Baseline JD min and max",
        float(np.round(lc.jd[0]-1, 1)),  float(np.round(lc.jd[-1]+1, 1)),
        (float(np.round(lc.jd[0], 1)), float(np.round(jd_disc_est, 1))),
        step=0.1
    )

    df = lc.to_pandas()
   # df['procstring'] = df.procstatus.astype(str)
    chart = alt.Chart(df).mark_point(size=60).encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('forcediffimflux', scale=alt.Scale(zero=False)),
        tooltip=['jd', 'forcediffimflux', 'forcediffimfluxunc', 'scisigpix',
                 'sciinpseeing', 'forcediffimchisq', 'procstatus']
    ).interactive()

    line = alt.Chart(lc.to_pandas()).mark_rule(color='cyan').encode(x=alt.datum(jd_min))
    line2 = alt.Chart(lc.to_pandas()).mark_rule(color='cyan').encode(x=alt.datum(jd_max))
    st.altair_chart(chart+line+line2, use_container_width=True)

    st.write("---")
    if "rescale uncertainty" in steps:
        lc["flux_corr"] = lc["forcediffimflux"] - lc.simple_median_baseline(jd_min, jd_max)
        lc.rescale_uncertainty()
        lc.RMS = lc.RMS_baseline(jd_min, jd_max)
        if lc.RMS >= 1.01:
            lc["fluxerr_corr"] = lc["forcediffimfluxunc"] * lc.RMS

    snr = st.slider(label="SNR limit threshold", min_value=1, max_value=10, value=3, step=1)
    snt = st.slider(label="Limit sigma (default 5)", min_value=1, max_value=15, value=5, step=1)
    lc.get_mag_lc(snr=snr, snt=snt)

    band_df = lc.to_pandas()

    band_df['plot_mag'] = pd.concat((band_df.mag[~band_df.islimit], band_df.lim[band_df.islimit]), axis=0)
    band_df['plot_err'] = band_df.mag_err[(~band_df.islimit) & (band_df.mag_err <= 1.5)]

    # annoying explicit domain setting because altair sometimes breaks and resets to zero anyway.
    ymin = band_df.plot_mag.min() -0.7

    base = alt.Chart(band_df).mark_point(size=60, filled=True).encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('plot_mag', scale=alt.Scale(reverse=True, zero=False, domainMin=ymin)),
        alt.YError('plot_err', band=1),
        color=alt.Color('islimit'),
        shape=alt.Shape('islimit', scale=alt.Scale(range=['circle', 'triangle'])),
        tooltip=['jd', 'mag', 'mag_err', 'islimit']
    )

    err = alt.Chart(band_df).mark_errorbar(clip=True, color='lightblue').encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('plot_mag', scale=alt.Scale(reverse=True, zero=False, domainMin=ymin)),
        alt.YError('plot_err', band=1),
        tooltip=['jd', 'mag', 'mag_err', 'islimit']
    )

    chart2 = base + err

    st.altair_chart(chart2.interactive(bind_y=False), use_container_width=True)

    detections = band_df[~band_df.islimit]
    limits = band_df[band_df.islimit]
    csv_det = cache_download(detections)
    csv_lim = cache_download(limits)

    st.download_button(
        label="Download data as CSV",
        data=csv_det,
        file_name=f'{snname}_detections.csv',
        mime='text/csv',
    )
    st.download_button(
        label="Download data as CSV",
        data=csv_lim,
        file_name=f'{snname}_limits.csv',
        mime='text/csv',
    )
    #st.dataframe(band_df[['mag','mag_err', 'plot_mag', 'plot_err']].dropna())
