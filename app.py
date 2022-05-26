import streamlit as st
from ztfphot import read_ztf_lc, ZtfLC, LC, verify_reference, SN, PlotType
from astropy.table import Table
import altair as alt
import numpy as np
import pandas as pd

from ztfphot.lightcurve import correct_flux_using_nearest_ref, get_ref_corrected_lc

st.set_page_config(page_title="ZTFphot: ZTF Forced Photometry lightcurve tool", page_icon=":telescope:", layout="wide")


st.title("ZTFphot")
st.write("Open source tool for making and plotting publication quality lightcurves from ZTF forced photometry")


@st.experimental_memo
def cached_read(uploaded_file) -> Table:
    file_exists = st.session_state.get(uploaded_file.name)
    if not file_exists:
        with open(uploaded_file.name, "wb") as outfile:
            outfile.write(uploaded_file.getvalue())
        st.session_state[uploaded_file.name] = True
    return read_ztf_lc(uploaded_file.name)


@st.experimental_memo
def cached_verify(_lc: LC, jd_first) -> LC:
    return verify_reference(_lc, jd_first)


@st.experimental_memo
def cache_download(data: pd.DataFrame):
    return data.to_csv(index=False).encode('utf-8')


snname = st.text_input("SN name: <SN2020xyz> / <ztf20TooLngNm> (Currently only used for output filename).")
uploaded_file = st.file_uploader('Upload a ZTF forced photometry service file', type=['txt', 'csv'], key='ztffps_file')
if uploaded_file:
    at = cached_read(uploaded_file)
    allbands = ZtfLC(at)

    band = st.radio("ZTF Filter:", ("g", "r", "i"))
    lc = allbands[allbands["filter"] == f"ZTF_{band}"]

    if len(lc) <= 10:
        st.error("Not enough data points!")

    st.markdown("### Adjust parameters and cleaning steps, results will update live ([see docs](https://irsa.ipac.caltech.edu/data/ZTF/docs/forcedphot.pdf)):")
    col1, col2 = st.columns(2)

    with col1:
        options = ["photometric", "good seeing",
                   "filter procstatus", "rescale uncertainty"]
        steps = st.multiselect(label="Select steps to perform",
                               options=options,
                               default=options,
                               help="""
                               Verify references: select only references without contamination \n
                               Photometric: select only scisigpix <= cutoff (default 6) \n
                               Good seeing: select only seeing <= cutoff (default 7) \n
                               Filter procstatus: Exclude listed proc status flags (corresponding to errors) \n
                               Rescale uncertainty: rescale uncertainty using chisq (Yao et al. 2019) \n
                               Correct references: correct the flux of references contaminated by transient flux.
                               """,
                               )

    with col2:
        with st.expander("Parameters for selected cleaning steps:", expanded=True):

            if "filter procstatus" in steps:
                procstatus = st.multiselect(label="Procstatus: procstatus to exclude",
                                            options=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 255],
                                            default=[56, 63, 64],
                                            help="Select the filter procstatus to exclude"
                                            )
                lc = lc.remove_bad_pixels([int(p) for p in procstatus])

            if "photometric" in steps:
                defaults = {"g": 7, "r": 8, "i": 9}
                scisigpix_cutoff = st.slider(label="Photometric: scisigpix_cutoff in pixels", min_value=1, max_value=50, value=defaults[band], step=1)
                lc = lc.photometric(scisigpix_cutoff)

            if "good seeing" in steps:
                seeing_cutoff = st.slider(label="Good seeing: seeing_cutoff in arcsec", min_value=0.9, max_value=10., value=7., step=0.1)
                lc = lc.remove_bad_seeing(seeing_cutoff)

    jd_peak_est = lc.estimate_peak_jd()
    jd_disc_est = jd_peak_est - 25
    jd_verify = jd_disc_est
    jd_max = jd_verify
    jd_min = lc.jd[0]

    _values = st.session_state.get(f"baseline_jd_min_max_{band}")
    if _values:
        jd_min, jd_max = _values

    reference_option = st.selectbox("How would you like to deal with references contaminated by transient flux?",
                                    ("Verify: remove references made with images after my selected jd",
                                     "Correct: Correct the flux using the nearest object as reference"),
                                    index=0)
    st.markdown(
        "### Set baseline and approximate discovery epoch using difference image flux lightcurve and sliders below."
        "(Discovery epoch only needed if Verifying references instead of Correcting for transient flux.)")
    if str(reference_option).startswith("Verify"):
        st.markdown("""<span style="color:yellow">Estimated transient begin JD:</span>""", unsafe_allow_html=True)
        use_baseline = st.button("Set last reference JD to Baseline JD Max", help="click to set last reference JD to Baseline JD Max")
        if st.session_state.get(f"jd_first_{band}"):
            jd_verify = st.session_state.get(f"jd_first_{band}")
        if use_baseline:
            jd_verify = jd_max

        jd_first = st.slider(label="Last JD of reference to include for reference cleaning. Set to "
                                   "a safe epoch before transient:",
                                   min_value=float(np.round(lc.jd[0], 1)), max_value=float(np.round(lc.jd[-1], 1)),
                                   value=float(np.round(jd_verify, 1)), step=0.1, key=f"jd_first_{band}",
                                   help='Everything before this JD will be allowed to be used in the reference.')
        lc = verify_reference(lc, jd_first)
    elif str(reference_option).startswith("Correct"):
        st.markdown("Correct reference flux using nearest reference if following parameters are satisfied:")

        sharp_lo, sharp_hi = st.slider(
            "PSF sharpness low and high cutoffs for nearby reference",
            -1.0, 1.0, (-0.7, 0.8),
            step=0.1,
            help="Ideally, values should be near zero, too high -> extended, too low -> sharp-spike."
                 "Defaults are good, but may be too lenient."
        )
        lc = lc.remove_non_psf_nearest_ref(sharp_hi=sharp_hi, sharp_lo=sharp_lo)


    st.markdown("""
    <span style="color:cyan">Baseline JD min and max:</span>
    """, unsafe_allow_html=True)
    jd_min, jd_max = st.slider(
        "Use the plot and sliders below to select the baseline time range:",
        float(np.round(lc.jd[0]-1, 1)),  float(np.round(lc.jd[-1]+1, 1)),
        (float(np.round(jd_min, 1)), float(np.round(jd_max, 1))),
        step=0.1, key=f"baseline_jd_min_max_{band}",
        help="Baseline should be selected so that the interval does not contain any SN flux but >30 points!")

    df = lc.to_pandas()
    chart = alt.Chart(df).mark_point(size=60).encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('forcediffimflux', scale=alt.Scale(zero=False)),
        tooltip=['jd', 'forcediffimflux', 'forcediffimfluxunc', 'scisigpix',
                 'sciinpseeing', 'forcediffimchisq', 'procstatus']
    ).interactive()

    line_min = alt.Chart(lc.to_pandas()).mark_rule(color='cyan').encode(x=alt.datum(jd_min))
    line_max = alt.Chart(lc.to_pandas()).mark_rule(color='cyan').encode(x=alt.datum(jd_max))
    if str(reference_option).startswith("Verify"):
        line_verify = alt.Chart(lc.to_pandas()).mark_rule(color='yellow').encode(x=alt.datum(jd_first))
        st.altair_chart(chart+line_min+line_max+line_verify, use_container_width=True)
    else:
        st.altair_chart(chart + line_min + line_max, use_container_width=True)

    # correct the flux:
    lc["flux_corr"] = lc["forcediffimflux"] - lc.simple_median_baseline(jd_min, jd_max)
    st.write("---")
    if "rescale uncertainty" in steps:
        #rescaled = st.session_state.get(f"rescaled_{band}", None)
        #if rescaled is None:
        lc.rescale_uncertainty()
        #st.session_state[f"rescaled_{band}"] = True
        RMS = lc.RMS_baseline(jd_min, jd_max)
        if RMS >= 1.01:
            lc["fluxerr_corr"] = lc["forcediffimfluxunc"] * RMS

    # Get the corrected flux into flux_corr and fluxerr_corr
    if str(reference_option).startswith("Correct"):
        errcol = "fluxerr_corr" if "fluxerr_corr" in lc.colnames else "forcediffimfluxunc"
        fluxcol = "flux_corr" if "flux_corr" in lc.colnames else "forcediffimflux"
        corr_flux, corr_err = correct_flux_using_nearest_ref(lc, "forcediffimflux", errcol)

    col1, col2 = st.columns(2)
    with col1:
        snr = st.slider(label="SNR limit threshold", min_value=1, max_value=10, value=3, step=1)
    with col2:
        snt = st.slider(label="Limit sigma (default 5)", min_value=1, max_value=15, value=5, step=1)

    if str(reference_option).startswith("Verify"):
        lc.get_mag_lc(snr=snr, snt=snt)
    band_df = lc.to_pandas()
    if str(reference_option).startswith("Verify"):
        lc.get_mag_lc(snr=snr, snt=snt)
        band_df['plot_mag'] = pd.concat((band_df.mag[~band_df.islimit], band_df.lim[band_df.islimit]), axis=0)
        band_df['plot_err'] = band_df.mag_err[(~band_df.islimit) & (band_df.mag_err <= 1.5)]
    if str(reference_option).startswith("Correct"):
        band_df['corr_flux'] = corr_flux
        band_df['corr_err'] = corr_err
        plot_mag, plot_err, is_limit, mag, mag_err = get_ref_corrected_lc(band_df.corr_flux, band_df.corr_err, band_df.zpdiff,
                                                  SNT=snr, SNU=snt)
        band_df['mag'] = mag
        band_df['mag_err'] = mag_err
        band_df['plot_mag'] = plot_mag
        band_df['plot_err'] = plot_err
        band_df['islimit'] = is_limit

    # annoying explicit domain setting because altair sometimes breaks and resets to zero anyway.
    ymin = max(band_df.plot_mag.dropna().min() - 0.7, 0)
    base = alt.Chart(band_df).mark_point(size=60, filled=True).encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('plot_mag', scale=alt.Scale(reverse=True, zero=False, domainMin=ymin)),
        alt.YError('plot_err', band=1),
        color=alt.Color('islimit'),
        shape=alt.Shape('islimit', scale=alt.Scale(range=['circle', 'triangle'])),
        tooltip=['jd', 'mag', 'mag_err', 'islimit', 'scisigpix', 'sciinpseeing', 'procstatus']
    )

    err = alt.Chart(band_df).mark_errorbar(clip=True, color='lightblue').encode(
        alt.X('jd', scale=alt.Scale(zero=False)),
        alt.Y('plot_mag', scale=alt.Scale(reverse=True, zero=False, domainMin=ymin)),
        alt.YError('plot_err', band=1),
        tooltip=['jd', 'mag', 'mag_err', 'islimit', 'scisigpix', 'sciinpseeing', 'procstatus']
    )

    chart2 = base + err

    st.markdown("### Final lightcurve:")
    st.altair_chart(chart2.interactive(bind_y=False), use_container_width=True)

    detections = band_df[~band_df.islimit]
    limits = band_df[band_df.islimit]
    csv_det = cache_download(detections)
    csv_lim = cache_download(limits)

    st.write("---")
    st.markdown("# Download lightcurve:")
    snname = snname if len(snname) > 0 else "SN"
    st.download_button(
        label="Download detections as CSV",
        data=csv_det,
        file_name=f'{snname}_{band}_detections.csv',
        mime='text/csv',
    )

    st.download_button(
        label="Download limits as CSV",
        data=csv_lim,
        file_name=f'{snname}_{band}_limits.csv',
        mime='text/csv',
    )

