# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#     "lightgbm>=4.6.0",
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.4",
#     "pandas>=3.1.0",
#     "pyarrow>=14.0.0",
#     "scikit-learn>=1.9.1",
#     "statsmodels>=0.15.0",
#     "xgboost>=3.4.1",
# ]
#
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bus Punctuality in Rural Wales: Multi-Day Hourly Delay and Reliability
    """)
    return


@app.cell
def _(mo):
    mo.outline(label="Table of Contents")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Cymru Wledig Local Partnership and Innovation Plan (LPIP) · Research placement report**

    Prepared by Mustafa Aslan · November 2026

    This report examines five months of bus vehicle tracking data from rural and regional Wales, matched to the national timetable and to hourly weather observations. It describes how late buses actually run, tests how much of that delay can be predicted, and isolates the causal effect of route geometry on punctuality.

    **Data:** SIRI-VM vehicle positions, the GTFS timetable feed for Wales, and Cardiff hourly weather.

    **Code and cached results:** [github.com/mustafaslanCoto/marimo-outputs](https://github.com/mustafaslanCoto/marimo-outputs)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <div style="border: 1px solid #dbeafe; background: #f8fafc; border-radius: 12px; padding: 18px 22px; margin: 10px 0 22px 0; line-height: 1.7;">
    <strong>PhD researcher</strong> · Mustafa Aslan<br/>
    <strong>Placement supervisor</strong> · Prof. Scott Orford, Cymru Wledig LPIP Rural Wales<br/>
    <strong>Academic main supervisor</strong> · Prof. Bahman Rostami-Tabar, Cardiff University
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Key findings at a glance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **The network runs close to schedule in a typical hour.** Median delay is **1.1 minutes** with an interquartile range of **4.2 minutes**, measured across 236,671 hourly route records that represent **13.4 million vehicle stop observations**.
    - **Almost half of observations fall outside the on-time window.** Using the bands defined in section 2, **55% are on time**, **26% run early**, **10% are late** and **9% are very late**.
    - **Early running is the hidden reliability problem.** Services between 05:00 and 07:00 have negative median delay, and the earliest Sunday departures run more than 4 minutes ahead of timetable.
    - **Friday afternoon is the worst window in the week.** Median delay reaches **2.6 minutes at 15:00 on Friday**, more than four times the Saturday figure at the same hour.
    - **Delay can be predicted in level but not in detail.** A full feature set explains **36 to 39% of the variance** in hourly delay. LightGBM is the stronger learner on all seven prediction tasks, by about 2% lower error on delay itself.
    - **Route shape is not the cause of lateness.** After controlling for weather, time, operator and route, neither circuity nor stop spacing has a detectable effect. Only total route length is significant, at about **0.08 minutes less delay per extra kilometre**, a small effect that most likely reflects timetable padding on longer routes.
    - **Weather matters less than operators expect.** Clear, overcast and rainy hours all sit within 0.2 minutes of each other in median delay.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Policy context: punctuality on an infrequent network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Wales has a mixed bus network. Urban corridors have frequent services, while many rural routes run only a few times a day. The panel in this report covers both: every route in the Welsh timetable that could be matched to vehicle tracking, not a rural subset. That matters for interpretation, because the same delay means something different on a route every ten minutes and a route every two hours.

    Where service is infrequent, a missed or early bus is not absorbed by the next one. The cost falls on passengers with no alternative, and it is the reason this report treats early running as a failure rather than a minor deviation. Two features of the timetable shape the analysis:

    - **Early departures are worse than moderate lateness.** A bus that leaves ahead of schedule strands passengers who arrived on time, and the wait for the next service can be long.
    - **Variability matters as much as the average.** A service that is consistently a few minutes late can be planned around; the same average with a wide spread produces missed connections and lost confidence.

    To describe punctuality we group observations into four bands using thresholds commonly used in bus performance reporting:

    | Band | Definition | Reading |
    |---|---|---|
    | Early | more than 1 minute before timetable | the bus may already have gone |
    | On time | 1 minute early to 5 minutes 59 seconds late | the conventional on-time window |
    | Late | 6 to 14 minutes 59 seconds late | noticeable, usually recoverable |
    | Very late | 15 minutes or more late | severe delay |

    These are descriptive labels for this analysis, not a statutory compliance test. The panel is built from operational tracking data rather than a regulatory return, so the shares should be read as evidence about service quality, not as a formal punctuality audit.

    The analysis answers three questions in sequence. How late are buses, and when? How much of that delay can be predicted from the data? And what part of the remaining variation is genuinely caused by the shape of the route?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data: sources and preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The panel combines three sources that share a single key: the hour of observation. Vehicle positions show where a bus actually was, the timetable shows where it should have been, and weather describes the conditions it faced.

    | | |
    |---|---|
    | **Period covered** | 1 April to 21 August 2026, with 119 days of full coverage and a short gap in mid-August |
    | **Raw material** | 242,695 hourly route records built from 13.4 million stop-level observations |
    | **Analysis panel** | 236,671 records after removing the extreme 1% at each end of the delay distribution |
    | **Modelling data** | 165,669 training rows and 71,002 test rows, with 18 base variables expanded to 225 columns |

    The trimmed tails are dominated by recording artefacts and depot movements rather than passenger-carrying service, so keeping them would distort every average in the report without adding information.

    The rest of this section describes each source and the choices that turned raw vehicle pings into a reliable measure of lateness.
    """)
    return


@app.cell
def _():
    import os, io, re, sys, time, json, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    warnings.filterwarnings("ignore")

    # DfT Standards & Visual Theme
    DFT_BANDS = ["Early", "On time", "Late", "Very late"]
    DFT_COLORS = {
        "Early": "#2a78d6",      # Bright Blue
        "On time": "#1baf7a",    # Forest Green
        "Late": "#eda100",       # Amber Orange
        "Very late": "#e34948"   # Red
    }
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return DOW_ORDER, Path, io, json, np, pd, plt, sys


@app.cell
def io_helpers(Path, json, mo, pd, sys):
    # Shared loaders for cached results that ship next to the notebook.
    # Works locally and as a WASM app on GitHub Pages, using the site's standard
    # pattern: mo.notebook_location + pyodide open_url / pyfetch in WASM.
    data_url = mo.notebook_location() / "lpip_results"


    def load_csv(name, **kwargs):
        url = str(data_url / name)
        if sys.platform == "emscripten":
            from pyodide.http import open_url

            return pd.read_csv(open_url(url), **kwargs)
        return pd.read_csv(url, **kwargs)


    def load_json(name):
        url = str(data_url / name)
        if sys.platform == "emscripten":
            from pyodide.http import open_url

            return json.loads(open_url(url).read())
        with open(url, "r") as f:
            return json.load(f)


    def save_local(name, text_or_bytes) -> bool:
        """Write a file next to the notebook when running locally.

        GitHub Pages (WASM) locations are read-only, so this returns False there
        and callers treat the cache write as skipped.
        """
        if sys.platform == "emscripten":
            return False
        path = Path(str(data_url / name))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(text_or_bytes, bytes) else "w"
        with open(path, mode) as f:
            f.write(text_or_bytes)
        return True

    return data_url, load_csv, load_json, save_local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Vehicle positions and timetable matching

    **Vehicle positions.** The SIRI-VM feed records vehicle locations at high frequency, often several times a minute for an active vehicle. The daily archive is written as one-minute snapshot files, and each file contains many pings per vehicle. For this project the daily files were downloaded from the WISERD OneDrive archive and compressed in the terminal, reducing about 600 GB of raw CSV to 26 GB. Each day contains more than 900 CSV files. Andrew Price in the placement team retrieved the live records. Each file was processed in Python, grouped by minute, vehicle and route, and appended across days into one panel, using a uniform 30% sample of rows.

    **Timetable matching.** A vehicle ping does not carry a GTFS trip identifier. It does carry the operator, the route number, the scheduled departure time from the origin and the direction of travel. The pipeline pre-indexes the national GTFS timetable feed for Wales into three lookups:

    1. the first scheduled stop of every trip, expressed as minutes from midnight;
    2. a primary key of operator, route, origin minute and direction, with a direction-free fallback;
    3. per-trip arrays of stop coordinates and scheduled times.

    That index reconstructs the missing link between a moving vehicle and its scheduled trip in milliseconds, rather than the hours a row-by-row spatial join over 1.2 million stop records would take.

    **Measuring lateness.** For each ping the pipeline finds the nearest stop on the matched trip using great-circle distance, keeps observations within 200 metres, and compares the observed time with the scheduled time at that stop. The difference is unwrapped around midnight so that a bus crossing 00:00 is not recorded as 24 hours late:

    $$\text{delay} = (t_{\text{observed}} - t_{\text{scheduled}} + 720) \bmod 1440 - 720$$

    Measuring against the nearest stop rather than the origin matters. An origin-based measure accumulates the whole journey and would show an on-time bus as 30 to 45 minutes late halfway along its route. The 200 metre filter keeps only genuine arrival and dwell events, where a timetable comparison is meaningful.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Weather observations

    Hourly observations from Cardiff cover temperature, rainfall, humidity, wind speed and a short description of conditions. Weather enters the analysis as an exogenous driver: rain and wind reduce operating speeds, lengthen boarding and increase stopping distances, so the effect should be visible in the panel if it matters.

    Observations are keyed to `timestamp_hour`, which allows a one-to-one join onto the bus panel. One limitation belongs up front: the station is in Cardiff while many routes run across mid and north Wales. The measure therefore captures regional weather imperfectly, and any estimated effect should be read as a lower bound.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Building the hourly panel

    The pipeline makes a single pass over the daily archives:

    1. stream each one-minute snapshot from its daily ZIP file;
    2. sample 30% of rows uniformly from every snapshot;
    3. convert UTC telemetry timestamps to local civil time (`Europe/London`);
    4. match every ping to its scheduled trip through the index described above;
    5. find the nearest stop and keep pings within 200 metres;
    6. compute delay against the scheduled time at that stop and unwrap it around midnight;
    7. assign each observation to a delay band;
    8. aggregate to route-hours and merge the Cardiff weather observations;
    9. save the panel as a parquet file that ships with this notebook.

    Streaming rather than loading whole archives keeps memory under 500 MB. Converting to `Europe/London` matters because timetables are published in local civil time, and ignoring British Summer Time would shift every delay by exactly 60 minutes.

    The unit of analysis is one **route-hour**: a row per operator, route and clock hour, carrying the number of observations behind it, mean and median delay, the interquartile range, and the share of observations in each punctuality band. That granularity is what makes the rest of the report possible, but it has one limitation worth stating. Direction of travel is not distinguished, so a row on a route served in both directions combines the two.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 What the panel contains

    | Variable | Meaning |
    |---|---|
    | `day`, `hour`, `dow`, `month` | calendar date, clock hour, day of week and month of the route-hour |
    | `operator`, `operator_name`, `line` | National Operator Code, operator name and commercial route number |
    | `n_obs` | number of vehicle stop observations behind the route-hour |
    | `mean_delay`, `median_delay` | lateness in minutes against the scheduled time at the nearest stop |
    | `iqr_delay` | interquartile range of delay within the route-hour (p75 minus p25) |
    | `planned_stops`, `total_dist_km` | scheduled stops and cumulative route length |
    | `avg_stop_spacing_km` | route length divided by the number of scheduled gaps |
    | `circuity_ratio` | route length divided by the straight-line distance to the farthest point reached |
    | `sched_duration_min`, `sched_speed_kmh` | scheduled running time and average scheduled speed |
    | `temperature_c`, `precipitation_mm` | hourly temperature and rainfall from the Cardiff station |
    | `relative_humidity_pct`, `wind_speed_kmh`, `condition` | hourly humidity, wind and a short text description of conditions |

    Hours between 00:00 and 04:59 were excluded because they had too few observations for reliable analysis. The full panel is loaded once from the parquet file beside this notebook, and every figure and model below is derived from it.
    """)
    return


@app.cell
async def _(io, mo, pd, sys):
    ## read from parquet file
    # Load the panel from the data folder next to the notebook. This works both
    # locally and as a WASM app on GitHub Pages (mo.notebook_location resolves
    # to the notebook's directory / public URL).
    _parquet_url = mo.notebook_location() / "lpip_data" / "hourly_bus_delay_panel.parquet"
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        _resp = await pyfetch(str(_parquet_url))
        hourly_df = pd.read_parquet(io.BytesIO(await _resp.bytes()))
    else:
        hourly_df = pd.read_parquet(str(_parquet_url))
    return (hourly_df,)


@app.cell
def _(hourly_df):
    hourly_dff = hourly_df[~hourly_df["hour"].isin([0, 1, 2, 3, 4])]
    return (hourly_dff,)


@app.cell
def _(hourly_dff):
    q_low, q_high = hourly_dff["mean_delay"].quantile([0.01, 0.99])
    hourly_dfilter = hourly_dff[hourly_dff["mean_delay"].between(q_low, q_high)].copy()
    return (hourly_dfilter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Descriptive findings: how late are buses, and when?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Section 3 produced a panel of route-hours. This section uses it to answer the first descriptive question, one visualisation at a time: what does the distribution of delay look like, and how does it change across the operating day, the week, the season, the operator and the weather?

    Two conventions apply throughout. Percentiles and shares are computed from the underlying vehicle observations, not from unweighted route-hour averages, because a quiet Sunday morning row should not count as much as a busy Friday afternoon one. And every chart shows the full distribution band rather than only the mean, because the width of that band is itself a policy-relevant fact.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 The overall delay distribution
    """)
    return


@app.cell
def _(hourly_dff, mo):
    _hour_values = sorted(hourly_dff["hour"].dropna().unique())
    _dow_values = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    _op_values = sorted(hourly_dff["operator_name"].dropna().unique())
    _line_values = sorted(hourly_dff["line"].dropna().unique())

    hour_filter = mo.ui.range_slider(
        start=int(_hour_values[0]),
        stop=int(_hour_values[-1]),
        value=(int(_hour_values[0]), int(_hour_values[-1])),
        step=1,
        show_value=True,
        label="Operating hour",
    )
    dow_filter = mo.ui.multiselect(options=_dow_values, value=_dow_values, label="Day of week")
    operator_filter = mo.ui.multiselect(options=_op_values, value=_op_values, label="Operator")
    line_filter = mo.ui.multiselect(options=_line_values, value=_line_values, label="Route line")

    mo.vstack([
        mo.hstack([hour_filter, dow_filter], justify="start", wrap=True, gap=1.2),
        mo.hstack([operator_filter, line_filter], justify="start", wrap=True, gap=1.2),
    ])
    return dow_filter, hour_filter, line_filter, operator_filter


@app.cell
def _(
    dow_filter,
    hour_filter,
    hourly_dff,
    line_filter,
    mo,
    operator_filter,
    plt,
):
    _filtered = hourly_dff[
        hourly_dff["hour"].between(*hour_filter.value)
        & hourly_dff["dow"].isin(dow_filter.value)
        & hourly_dff["operator_name"].isin(operator_filter.value)
        & hourly_dff["line"].isin(line_filter.value)
    ]

    _s = _filtered["mean_delay"].dropna()
    if len(_s):
        _lo, _hi = _s.quantile([0.01, 0.99])
        _s = _s[(_s >= _lo) & (_s <= _hi)]

    dist_n = int(len(_s))
    dist_median = float(_s.median()) if dist_n else float("nan")
    dist_mean = float(_s.mean()) if dist_n else float("nan")
    dist_p25 = float(_s.quantile(0.25)) if dist_n else float("nan")
    dist_p75 = float(_s.quantile(0.75)) if dist_n else float("nan")
    dist_iqr = dist_p75 - dist_p25

    _fig, _ax = plt.subplots(figsize=(12, 4.2), dpi=120, facecolor="#fafafa")
    _ax.set_facecolor("#ffffff")

    if dist_n:
        _ax.hist(_s, bins=60, color="#38bdf8", edgecolor="#0284c7", alpha=0.75, density=True)
        _ax.axvline(dist_median, color="#0f172a", lw=2, label=f"Median ({dist_median:.1f}m)")
        _ax.axvline(dist_mean, color="#ef4444", linestyle="--", lw=1.8, label=f"Mean ({dist_mean:.1f}m)")
        _ax.axvline(dist_p25, color="#64748b", linestyle=":", lw=1.2)
        _ax.axvline(dist_p75, color="#64748b", linestyle=":", lw=1.2)
        _ax.text(
            0.99, 0.96,
            f"n = {dist_n:,}\nMedian = {dist_median:.1f} min\nMean = {dist_mean:.1f} min\nIQR = {dist_iqr:.1f} min",
            transform=_ax.transAxes, ha="right", va="top", fontsize=9.5, color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95),
        )
    else:
        _ax.text(0.5, 0.5, "No observations match this filter", ha="center", va="center",
                 transform=_ax.transAxes, fontsize=11)

    _ax.set_title("Distribution of hourly mean delay", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    _ax.set_xlabel("Mean delay (minutes)", fontsize=10.5, fontweight="bold", color="#334155")
    _ax.set_ylabel("Density", fontsize=10.5, fontweight="bold", color="#334155")
    _ax.grid(True, linestyle="--", alpha=0.25, color="#64748b", zorder=0)
    for _sp in ["top", "right"]:
        _ax.spines[_sp].set_visible(False)
    for _sp in ["left", "bottom"]:
        _ax.spines[_sp].set_color("#cbd5e1")
    if dist_n:
        _ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    _dist_chart = mo.as_html(_fig)
    plt.close(_fig)

    _dist_chart
    return dist_iqr, dist_mean, dist_median, dist_n, dist_p25, dist_p75


@app.cell(hide_code=True)
def _(dist_iqr, dist_mean, dist_median, dist_n, dist_p25, dist_p75, mo):
    mo.md(rf"""
    **What this shows.** Across the current selection of {dist_n:,} route-hours, median delay is {dist_median:.1f} minutes and mean delay {dist_mean:.1f} minutes, with an interquartile range of {dist_iqr:.1f} minutes. The central half of the selected observations lies between {dist_p25:.1f} and {dist_p75:.1f} minutes. The distribution sits just above the timetable and is pulled to the right by a long late tail.

    **How to read it.** The filters apply to this figure only; every other chart and model uses the full panel. Comparing selections shows whether a pattern is network-wide or confined to particular hours, operators or routes. The shape stays broadly similar, while the centre and the width move.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Hour of day
    """)
    return


@app.cell
def _(DOW_ORDER, mo):
    hour_dow_filter = mo.ui.multiselect(
        options=DOW_ORDER,
        value=DOW_ORDER,
        label="Day of week",
    )
    hour_dow_filter
    return (hour_dow_filter,)


@app.cell
def _(DOW_ORDER, hour_dow_filter, hourly_dfilter, mo, plt):
    _selected_dows = hour_dow_filter.value or DOW_ORDER
    _hf = hourly_dfilter[hourly_dfilter["dow"].isin(_selected_dows)]

    # 1. Compute median delay for every single calendar day by hour
    _daily_hourly = (
        _hf
        .groupby(["day", "hour"], observed=True)["median_delay"]
        .median()
        .reset_index()
    )

    # 2. Compute hourly percentiles across all days
    _stats = (
        _daily_hourly.groupby("hour", observed=True)["median_delay"]
        .agg(
            median="median",
            p10=lambda s: s.quantile(0.10),
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )

    # 3. Create styled figure
    _fig, _ax = plt.subplots(figsize=(12, 5.5), dpi=120, facecolor="#fafafa")
    _ax.set_facecolor("#ffffff")

    # Soft outer range (10th - 90th percentile: shows rare outliers softly)
    _ax.fill_between(
        _stats["hour"],
        _stats["p10"],
        _stats["p90"],
        color="#cbd5e1",
        alpha=0.35,
        label="Wide Variation (10th–90th percentile)",
        zorder=1,
    )

    # Core IQR band (25th - 75th percentile: typical day-to-day window)
    _ax.fill_between(
        _stats["hour"],
        _stats["p25"],
        _stats["p75"],
        color="#38bdf8",
        alpha=0.30,
        label="Typical Range (IQR: 25th–75th percentile)",
        zorder=2,
    )

    # Solid, bold median trendline
    _ax.plot(
        _stats["hour"],
        _stats["median"],
        color="#0f172a",
        linewidth=2.8,
        marker="o",
        markersize=5.5,
        markerfacecolor="#ffffff",
        markeredgewidth=2.2,
        markeredgecolor="#0f172a",
        label="Network Median Delay (All Days)",
        zorder=4,
    )

    # Reference Thresholds
    _ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.8, label="Timetable Schedule (0 min)")
    _ax.axhline(5.99, color="#ef4444", linestyle="--", linewidth=1.3, alpha=0.85, label="Late Standard (+6 min)")

    # Annotate the highest delay peak
    _max_row = _stats.loc[_stats["median"].idxmax()]
    _ax.annotate(
        f"Peak Median: {_max_row['median']:.1f} min",
        xy=(_max_row["hour"], _max_row["median"]),
        xytext=(_max_row["hour"] - 2.8, _max_row["median"] + 1.2),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
    )

    # Axis formatting
    _hours = list(range(int(_stats["hour"].min()), int(_stats["hour"].max()) + 1))
    _ax.set_xticks(_hours)
    _ax.set_xticklabels([f"{h:02d}:00" for h in _hours], rotation=40, ha="right", fontsize=9)
    _ax.set_xlim(min(_hours) - 0.4, max(_hours) + 0.4)
    _ax.set_ylim(bottom=min(-1.5, _stats["p10"].min() - 0.5))

    _ax.set_title("Network-Wide Diurnal Bus Delay Profile (Hourly Distribution)", fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    _ax.set_xlabel("Operating Hour of Day (Local Civil Time)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
    _ax.set_ylabel("Delay (Minutes)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")

    # Elegant subtle grid and spines
    _ax.grid(True, linestyle="--", alpha=0.25, color="#64748b", zorder=0)
    for _spine in ["top", "right"]:
        _ax.spines[_spine].set_visible(False)
    for _spine in ["left", "bottom"]:
        _ax.spines[_spine].set_color("#cbd5e1")

    # Clean legend
    _ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.2)

    plt.tight_layout()
    _clean_plot = mo.as_html(_fig)
    plt.close(_fig)

    _clean_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** Early services run ahead of timetable: the median between 05:00 and 07:00 runs between 0.2 and 0.7 minutes early, and the lower band shows that on a quarter of days the 05:00 median is close to 2 minutes early. Delay rises sharply after 08:00 and forms a broad plateau from 11:00 to 16:00, where the median sits around 1.6 to 1.7 minutes. It eases after 17:00 even though the number of observations is highest then. The figures describe the default view with all days selected; the filter above isolates a single day or a combination.

    **Why it matters.** The gap between the morning median and the timetable is the clearest evidence of early running in the data, and it is a service failure that a lateness target alone does not capture. The midday plateau coincides with shopping, school and health appointment travel, when a missed connection is most costly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Day of the week
    """)
    return


@app.cell
def _(hourly_dfilter, mo, np, plt):
    # 1. Compute median delay per unique day + day of week to get true daily variance
    _daily_medians = (
        hourly_dfilter.groupby(["day", "dow"], observed=True)["median_delay"]
        .median()
        .reset_index()
    )

    # 2. Group by Day of Week and compute percentiles
    _dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    _dow_stats = (
        _daily_medians.groupby("dow", observed=True)["median_delay"]
        .agg(
            median="median",
            p10=lambda s: s.quantile(0.10),
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            count="count",
        )
        .reindex(_dow_order)
        .reset_index()
    )

    # 3. Create styled figure
    _fig, _ax = plt.subplots(figsize=(11, 5.2), dpi=120, facecolor="#fafafa")
    _ax.set_facecolor("#ffffff")

    _x = np.arange(len(_dow_stats))

    # Soft outer range (10th–90th percentile: shows extreme days across weeks)
    _ax.fill_between(
        _x,
        _dow_stats["p10"],
        _dow_stats["p90"],
        color="#cbd5e1",
        alpha=0.35,
        label="Weekly Variation (10th–90th percentile)",
        zorder=1,
    )

    # Core IQR band (25th–75th percentile: typical operating window)
    _ax.fill_between(
        _x,
        _dow_stats["p25"],
        _dow_stats["p75"],
        color="#38bdf8",
        alpha=0.30,
        label="Typical Window (IQR: 25th–75th percentile)",
        zorder=2,
    )

    # Solid bold median trendline
    _ax.plot(
        _x,
        _dow_stats["median"],
        color="#0f172a",
        linewidth=2.8,
        marker="o",
        markersize=6,
        markerfacecolor="#ffffff",
        markeredgewidth=2.2,
        markeredgecolor="#0f172a",
        label="Median Delay by Day of Week",
        zorder=4,
    )

    # Subtle shaded separation for Weekend (Saturday & Sunday)
    _ax.axvspan(3.5, 6.5, color="#f1f5f9", alpha=0.6, zorder=0, label="Weekend Period")

    # Reference Thresholds
    # _ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.8, label="Timetable Schedule (0 min)")
    # _ax.axhline(5.99, color="#ef4444", linestyle="--", linewidth=1.3, alpha=0.85, label="Late Standard (+6 min)")

    # Annotate Friday or the highest delay day
    _peak_idx = _dow_stats["median"].idxmax()
    _peak_dow = _dow_stats.loc[_peak_idx]
    _ax.annotate(
        f"Peak Day: {_peak_dow['dow']}\nMedian: {_peak_dow['median']:.1f} min",
        xy=(_peak_idx, _peak_dow["median"]),
        xytext=(_peak_idx - 0.9, _peak_dow["median"] + 1.1),
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.2),
        fontsize=9.2,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
    )

    # Axes formatting
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_dow_stats["dow"], fontsize=10, fontweight="bold", color="#334155")
    _ax.set_xlim(-0.4, len(_dow_stats) - 0.6)
    # _ax.set_ylim(bottom=min(-1.5, _dow_stats["p10"].min() - 0.5))

    _ax.set_title("Weekly Bus Delay Profile: Day of Week Dynamics", fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    _ax.set_xlabel("Day of Week", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
    _ax.set_ylabel("Median Delay (Minutes)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")

    # Elegant subtle grid and spines
    _ax.grid(True, linestyle="--", alpha=0.25, color="#64748b", zorder=0)
    for _spine in ["top", "right"]:
        _ax.spines[_spine].set_visible(False)
    for _spine in ["left", "bottom"]:
        _ax.spines[_spine].set_color("#cbd5e1")

    # Clean legend
    _ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.0)

    plt.tight_layout()
    _dow_plot = mo.as_html(_fig)
    plt.close(_fig)

    _dow_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** Friday is the most delayed day in the daily series, with a median of 1.2 minutes; the central half of Fridays lies between 0.9 and 1.5 minutes. Saturday is the calmest at 0.7 minutes, and Monday to Thursday are tightly grouped between 0.8 and 0.9. The outer band shows that individual weeks vary more than the weekly averages suggest.

    **Why it matters.** The Friday penalty is a recurring operational pattern, not a one-off event, which makes it a candidate for a timetable fix rather than a monitoring problem. Planning Monday to Thursday as one block and Friday as another would match the data better than a single weekly schedule.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 Month by month
    """)
    return


@app.cell
def _(hourly_dfilter, mo, pd, plt):
    # 1. Drop months with too little coverage (a partial March tail) and build one series per month
    _month_counts = hourly_dfilter.groupby("month").size()
    _months = [m for m in sorted(_month_counts.index) if _month_counts[m] >= 1000]
    _month_labels = [pd.Timestamp(m).strftime("%b") for m in _months]
    _data_by_month = [
        hourly_dfilter[hourly_dfilter["month"] == _m]["median_delay"].dropna()
        for _m in _months
    ]

    # 2. Create styled figure
    _fig, _ax = plt.subplots(figsize=(10, 5.2), dpi=120, facecolor="#fafafa")
    _ax.set_facecolor("#ffffff")

    # 3. Create boxplot with mean displayed
    _bp = _ax.boxplot(
        _data_by_month,
        tick_labels=_month_labels,
        patch_artist=True,
        showmeans=True,
        showfliers=False,  # Hides individual outlier dots so the plot stays clean
        meanprops=dict(
            marker="D",
            markeredgecolor="#ffffff",
            markerfacecolor="#ef4444",
            markersize=6.5,
            zorder=5,
        ),
        medianprops=dict(
            color="#0f172a",
            linewidth=2.4,
        ),
        boxprops=dict(
            facecolor="#e0f2fe",
            edgecolor="#0284c7",
            linewidth=1.4,
        ),
        whiskerprops=dict(
            color="#64748b",
            linewidth=1.2,
            linestyle="--",
        ),
        capprops=dict(
            color="#64748b",
            linewidth=1.4,
        ),
        widths=0.45,
    )

    # Reference Thresholds
    _ax.axhline(0, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.8, label="Timetable Schedule (0 min)")
    _ax.axhline(5.99, color="#ef4444", linestyle="--", linewidth=1.3, alpha=0.85, label="Late Standard (+6 min)")

    # Annotate numeric values for Median and Mean next to each box
    for _i, _m in enumerate(_months):
        _vals = hourly_dfilter[hourly_dfilter["month"] == _m]["median_delay"].dropna()
        _med_val = _vals.median()
        _mean_val = _vals.mean()
        _ax.text(
            _i + 1 + 0.28,
            _med_val,
            f"Med: {_med_val:.1f}m",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
        )
        _ax.text(
            _i + 1 + 0.28,
            _mean_val,
            f"Mean: {_mean_val:.1f}m",
            va="center",
            fontsize=8.2,
            color="#b91c1c",
        )

    # Formatting
    _ax.set_title("Monthly Bus Delay Distribution (Box Plot with Mean & Median)", fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    _ax.set_xlabel("Calendar Month", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
    _ax.set_ylabel("Delay (Minutes)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")

    # Custom Legend
    from matplotlib.lines import Line2D
    _legend_elements = [
        Line2D([0], [0], color="#0f172a", lw=2.4, label="Median Delay"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#ef4444", markersize=7, label="Mean Delay"),
        Line2D([0], [0], color="#64748b", linestyle=":", lw=1.2, label="Schedule (0m)"),
        Line2D([0], [0], color="#ef4444", linestyle="--", lw=1.3, label="Late Standard (+6m)"),
    ]
    _ax.legend(handles=_legend_elements, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.0)

    # Elegant subtle grid and spines
    _ax.grid(axis="y", linestyle="--", alpha=0.25, color="#64748b", zorder=0)
    for _spine in ["top", "right"]:
        _ax.spines[_spine].set_visible(False)
    for _spine in ["left", "bottom"]:
        _ax.spines[_spine].set_color("#cbd5e1")

    _ax.set_xlim(0.5, len(_months) + 0.6)

    plt.tight_layout()
    _box_plot = mo.as_html(_fig)
    plt.close(_fig)

    _box_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** Median delay moves only between 0.78 and 0.83 minutes from April to August, and the interquartile range stays between 3.0 and 3.3 minutes. August is marginally more variable than spring, but the difference is small compared with the week-to-week variation within any single month.

    **Why it matters.** Punctuality is essentially flat across the season. That weakens the case that summer tourism is a major driver of rural lateness, and it supports treating the Friday and midday patterns in sections 4.3 and 4.5 as structural features of the timetable rather than seasonal artefacts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.5 When congestion bites: day by hour
    """)
    return


@app.cell
def _(DOW_ORDER, hourly_dfilter, mo, np, plt):
    # --- TEMPORAL CONGESTION MATRIX: DAY OF WEEK x OPERATING HOUR (DERIVED FROM hourly_df) ---
    # WHAT: Construct a 2D cross-tabulation of median delay by Day of Week (Mon-Sun) and Hour of Day (06-23).
    # WHY: Uncovers recurring operational bottlenecks (e.g. Friday PM peak congestion vs Sunday morning early departures).
    dow_hour = (
        hourly_dfilter.groupby(["dow", "hour"], observed=True)["median_delay"]
        .median()
        .unstack("hour")
        .reindex(DOW_ORDER)
    )

    # Render annotated thermal heatmap
    _fig, _ax = plt.subplots(figsize=(14, 4.8), dpi=100)
    _im = _ax.imshow(dow_hour.values, cmap="YlOrRd", aspect="auto")
    _cbar = _fig.colorbar(_im, ax=_ax, pad=0.02)
    _cbar.set_label("Median Delay (Minutes)", fontsize=10, fontweight="bold")

    _ax.set_xticks(range(dow_hour.shape[1]))
    _ax.set_xticklabels([f"{h:02d}:00" for h in dow_hour.columns], rotation=45, ha="right", fontsize=9)
    _ax.set_yticks(range(len(DOW_ORDER)))
    _ax.set_yticklabels(DOW_ORDER, fontsize=10, fontweight="bold")
    _ax.set_title("Day of Week × Operating Hour Median Delay Heatmap", fontsize=12, fontweight="bold")

    # Annotate numeric median delay inside each cell
    for _i in range(len(DOW_ORDER)):
        for _j in range(dow_hour.shape[1]):
            _val = dow_hour.values[_i, _j]
            if not np.isnan(_val):
                _ax.text(_j, _i, f"{_val:.1f}", ha="center", va="center", color="black" if _val < 2.5 else "white", fontsize=8)

    plt.tight_layout()
    _heat_chart = mo.as_html(_fig)
    plt.close(_fig)

    _heat_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** The heatmap localises the weekly peak at Friday 15:00, where median delay reaches 2.6 minutes against a weekday midday norm of about 1.6. The other extreme is Sunday at 05:00, where the median is 4.1 minutes early. Saturday mornings are consistently close to or ahead of schedule, and median delay falls back towards zero after 21:00 on every day.

    **Why it matters.** Two different problems appear in the same matrix. Friday afternoon is a lateness problem driven by road conditions and end-of-week demand. Sunday and early-morning services are an early-running problem driven by schedules that assume more traffic than exists. They call for opposite interventions: more recovery time in the first case, hold-back discipline in the second.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.6 Operator comparison
    """)
    return


@app.cell
def _(hourly_dfilter, mo, plt):
    # --- OPERATOR PERFORMANCE LEAGUE TABLE (DERIVED FROM hourly_df) ---
    # WHAT: Group hourly_df by operator_name to compute median delay, mean delay, IQR volatility, and statutory compliance.
    # WHY: Directly benchmarks major Welsh bus operators (First Cymru, Stagecoach, Arriva, Cardiff Bus, etc.)
    # to highlight who operates predictable schedules vs volatile operations.
    op_df = (
        hourly_dfilter.groupby("operator_name", observed=True)
        .agg(
            observations=("n_obs", "sum"),
            median_delay=("median_delay", "median"),
            mean_delay=("mean_delay", "mean"),
            iqr_delay=("iqr_delay", "median"),
            on_time_pct=("on_time_pct", "mean"),
            very_late_pct=("very_late_pct", "mean"),
        )
        .reset_index()
    )
    # Filter to operators with at least 200 observations for statistical reliability
    op_df = op_df[op_df["observations"] >= 200].sort_values("median_delay", ascending=True).round(2)

    # Render horizontal bar chart of operator median delays
    _fig, _ax = plt.subplots(figsize=(12, max(4, len(op_df) * 0.45)), dpi=100)
    _y = range(len(op_df))
    _ax.barh(_y, op_df["median_delay"], color="#4a3aa7", alpha=0.85, label="Median Delay (min)")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(op_df["operator_name"], fontsize=9.5)
    _ax.set_xlabel("Median Delay (Minutes)", fontsize=10, fontweight="bold")
    _ax.set_title("Operator Performance League Table in Wales", fontsize=12, fontweight="bold")
    _ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _op_chart = mo.as_html(_fig)
    plt.close(_fig)

    _op_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** The three operators with substantial coverage sit within a narrow band: Stagecoach Merseyside at 0.66 minutes median, First South and West Wales at 0.77 and Stagecoach South Wales at 0.84. The two operators with the highest medians, First Worcestershire and A2B Cars, have only a few thousand observations and a different operating footprint, so their position should not be read as a like-for-like ranking.

    **Why it matters.** The major operators differ by less than 0.2 minutes in the median but by much more in their interquartile range, which runs from 2.0 to 3.0 minutes. A league table built on average delay would therefore mislead. Reliability, measured as the spread of delay, separates them more clearly than lateness itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.7 Weather and temperature
    """)
    return


@app.cell
def _(hourly_dff, hourly_dfilter, mo, np, plt):
    # --- WEATHER & CLIMATE SENSITIVITY ANALYSIS (DERIVED FROM hourly_df) ---
    # WHAT: Analyze bus operational delay against continuous ambient temperature (°C) and
    # atmospheric weather conditions (Rain, Overcast, Clear) using merged Cardiff weather observations.
    # WHY: Avoids arbitrary temperature bins and pinpoints the true non-linear relationship between
    # temperature and operational delay alongside weather-induced travel friction.
    _has_weather = "temperature_c" in hourly_dfilter.columns and hourly_dff["temperature_c"].notnull().sum() > 0

    if _has_weather:
        _w_df = hourly_dfilter.dropna(subset=["temperature_c", "median_delay"]).copy()

        # 1. Continuous Temperature Profile: Group by rounded degree and compute percentiles
        _temp_profile = (
            _w_df.groupby(_w_df["temperature_c"].round(), observed=True)["median_delay"]
            .agg(
                median="median",
                p25=lambda s: s.quantile(0.25),
                p75=lambda s: s.quantile(0.75),
                count="count",
            )
            .reset_index()
        )
        # Exclude tail temperatures with fewer than 50 observations for statistical reliability
        _temp_profile = _temp_profile[_temp_profile["count"] >= 50].sort_values("temperature_c")

        # 2. Weather Condition Aggregation: Filter to conditions with at least 200 observations
        _cond_agg = (
            _w_df.dropna(subset=["condition"])
            .groupby("condition", observed=True)
            .agg(
                median_delay=("median_delay", "median"),
                n=("n_obs", "sum"),
            )
            .reset_index()
        )
        _cond_agg = _cond_agg[_cond_agg["n"] > 200].sort_values("median_delay")

        # 3. Create 2-Panel Figure
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=120, facecolor="#fafafa")
        _ax1.set_facecolor("#ffffff")
        _ax2.set_facecolor("#ffffff")

        # --- PANEL 1: Continuous Temperature vs Delay ---
        _ax1.fill_between(
            _temp_profile["temperature_c"],
            _temp_profile["p25"],
            _temp_profile["p75"],
            color="#38bdf8",
            alpha=0.25,
            label="Typical Range (IQR: 25th–75th %ile)",
            zorder=2,
        )
        _ax1.plot(
            _temp_profile["temperature_c"],
            _temp_profile["median"],
            color="#0f172a",
            linewidth=2.6,
            marker="o",
            markersize=5,
            markerfacecolor="#ffffff",
            markeredgewidth=2,
            markeredgecolor="#0f172a",
            label="Network Median Delay",
            zorder=4,
        )


        _ax1.set_title("Operational Delay vs. Ambient Temperature", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
        _ax1.set_xlabel("Ambient Temperature (°C)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
        _ax1.set_ylabel("Delay (Minutes)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
        _ax1.grid(True, linestyle="--", alpha=0.25, color="#64748b", zorder=0)
        _ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.8)

        for _sp in ["top", "right"]:
            _ax1.spines[_sp].set_visible(False)
        for _sp in ["left", "bottom"]:
            _ax1.spines[_sp].set_color("#cbd5e1")

        # --- PANEL 2: Operational Delay by Atmospheric Condition ---
        _y = np.arange(len(_cond_agg))
        _bars = _ax2.barh(_y, _cond_agg["median_delay"], color="#4a3aa7", alpha=0.85, height=0.55, zorder=3)
        _ax2.set_yticks(_y)
        _ax2.set_yticklabels(_cond_agg["condition"], fontsize=9.5, fontweight="bold", color="#334155")

        # Direct value annotations on bars
        for _idx, _bar in enumerate(_bars):
            _val = _cond_agg["median_delay"].iloc[_idx]
            _ax2.text(
                _val + 0.1,
                _bar.get_y() + _bar.get_height() / 2,
                f"{_val:.2f}m",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
            )

        _ax2.set_title("Operational Delay by Weather Condition", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
        _ax2.set_xlabel("Median Delay (Minutes)", fontsize=10.5, fontweight="bold", labelpad=8, color="#334155")
        _ax2.grid(axis="x", linestyle="--", alpha=0.25, color="#64748b", zorder=0)
        # _ax2.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.8)

        for _sp in ["top", "right"]:
            _ax2.spines[_sp].set_visible(False)
        for _sp in ["left", "bottom"]:
            _ax2.spines[_sp].set_color("#cbd5e1")

        plt.tight_layout()
        _w_chart = mo.as_html(_fig)
        plt.close(_fig)

        _view = _w_chart
    else:
        _view = mo.md("*Weather data integration not available.*")

    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** The temperature curve rises gently from slightly negative median delay in cool hours to about 1.2 minutes at 20°C and above, then flattens at the hottest extremes. The condition bars are nearly level: clear sky, overcast, drizzle and slight rain all sit between 0.7 and 0.9 minutes, and the differences are smaller than the day-to-day spread within any one condition.

    **Why it matters.** These are raw differences, and temperature is strongly confounded with time of day: hot hours fall in the midday plateau, while cool hours fall at night when buses run early. The pattern is therefore consistent with the operating-day effect in section 4.2 rather than with a separate weather penalty. The models in section 5 include weather alongside hour, day and operator, so any independent contribution is captured there.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Predictive findings: how much of delay can be explained?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Section 4 describes when delay happens. This section asks whether it can be anticipated. That is a different question, and a more demanding one: it requires the model to reproduce the level and variation of delay in hours it has never seen.

    Two metrics do the work. **R²** is the share of variance in delay that the model reproduces, where 1.0 is perfect and 0.0 is no better than predicting the average. **RMSE** is the typical error in minutes, in the same units as the outcome, so it can be compared directly with the 1.1 minute median and 4.2 minute interquartile range from section 4.

    The sequence is deliberately incremental: a linear model first, then gradient boosting on the same features. The gap between them measures what flexible machine learning adds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 What the models see

    The modelling table drops identifiers, counters and every punctuality share that is derived from the outcome itself, leaving 17 predictors across five families: time (hour, day of week), operations (operator, route line), weather (temperature, rainfall, humidity, wind, conditions), route geometry (stops, total and maximum reach distance, stop spacing, circuity, scheduled duration and speed) and distance from Cardiff. Categorical columns become indicator variables, which takes the table to 225 columns, of which 224 are predictors.

    One row is still one route-hour. The table below shows the shape of the data the models receive, with route geometry repeated across the hours of a given route and delay-varying hour by hour.
    """)
    return


@app.cell
def _(hourly_dfilter):
    model_df = hourly_dfilter.drop(columns=["timestamp_hour", "day", "period", "n_obs", "median_delay", "p25_delay", "p75_delay", "on_time_pct", "early_pct", "late_pct", "very_late_pct", "stop_index", "route_progress_pct", "max_planned_stops", "dest_dist_cardiff_km", "iqr_delay", "month", "daytype", "operator_name"])
    dum_vars = ["hour", "dow", "operator", "line", "condition"]
    return dum_vars, model_df


@app.cell
def _(dum_vars, model_df, pd):
    ## 1. Create dummy variables for categorical features
    model_dummy =  pd.get_dummies(model_df, columns = dum_vars, drop_first=True, dtype=int, prefix_sep="_")
    return (model_dummy,)


@app.cell
def _(dum_vars, model_df, pd):
    ml_model_df =  pd.get_dummies(model_df, columns = dum_vars, drop_first=False, dtype=int, prefix_sep="_")
    # ml_model_df = model_dummy.dropna().copy()
    treatment_var1 = "circuity_ratio"
    treatment_var2 = "avg_stop_spacing_km"
    treatment_var3 = "total_dist_km"
    outcome_var = "mean_delay"

    # 70/30 train/test split — tune on training data; test data kept aside for later
    from sklearn.model_selection import train_test_split

    ml_train_df, ml_test_df = train_test_split(
        ml_model_df, test_size=0.3, random_state=42
    )
    print(f"train: {ml_train_df.shape[0]} rows, test: {ml_test_df.shape[0]} rows")
    return (
        ml_test_df,
        ml_train_df,
        outcome_var,
        treatment_var1,
        treatment_var2,
        treatment_var3,
    )


@app.cell
def _(mo, model_df):
    mo.ui.table(model_df.round(3))
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What this shows.** Each row is a route-hour, carrying the outcome (`mean_delay`) and the variables that may explain it. Route geometry is a property of the route and repeats across all of its hours, while weather and time change hour by hour.

    **Why it matters.** That structure is what makes the causal question in section 6 delicate. Route geometry cannot be varied independently of the route that carries it, so a naive comparison of long and short routes would mix the effect of geometry with everything else about those routes. The next two sections handle that problem in stages.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Linear benchmark

    The first model is ordinary least squares on the same 224 predictors, with the categorical indicators entering as separate columns. It is not expected to win, but it provides a transparent lower bound and shows how much of the signal is linear.
    """)
    return


@app.cell
def _(json, load_json, model_dummy, save_local):
    # Estimating this OLS by pinv/SVD on ~165k x 200 takes minutes in the browser
    # (no BLAS in WASM), which trips marimo's RPC timeout before the page can paint.
    # Cache the fitted summary and the treatment coefficients in
    # lpip_results/ols_summary.json and reuse them.
    from types import SimpleNamespace

    _TREATMENTS = ("circuity_ratio", "avg_stop_spacing_km", "total_dist_km")

    try:
        _cached_ols = load_json("ols_summary.json")
    except Exception:
        _cached_ols = None

    if _cached_ols is not None:
        ols_results = SimpleNamespace(
            nobs=_cached_ols["nobs"],
            rsquared=_cached_ols["rsquared"],
            rsquared_adj=_cached_ols["rsquared_adj"],
            fvalue=_cached_ols["fvalue"],
            summary_html=_cached_ols["html"],
            coefficients=_cached_ols.get("coefficients", {}),
        )
        print(" Loaded OLS summary from lpip_results/ols_summary.json")
    else:
        y = model_dummy["mean_delay"]
        X = model_dummy.drop(columns=["mean_delay"])
        from statsmodels.api import OLS, add_constant

        X = add_constant(X)
        ols_model = OLS(y, X)
        ols_results = ols_model.fit()

        _ci = ols_results.conf_int()
        _coefficients = {
            name: {
                "coef": float(ols_results.params[name]),
                "std_err": float(ols_results.bse[name]),
                "p": float(ols_results.pvalues[name]),
                "ci_low": float(_ci.loc[name, 0]),
                "ci_high": float(_ci.loc[name, 1]),
            }
            for name in _TREATMENTS
        }
        _payload = json.dumps(
            {
                "nobs": float(ols_results.nobs),
                "rsquared": float(ols_results.rsquared),
                "rsquared_adj": float(ols_results.rsquared_adj),
                "fvalue": float(ols_results.fvalue),
                "html": ols_results.summary().as_html(),
                "coefficients": _coefficients,
            }
        )
        if save_local("ols_summary.json", _payload):
            print(" Saved OLS summary to lpip_results/ols_summary.json")
    return (ols_results,)


@app.cell
def _(mo, ols_results):
    _fit_card = mo.md(rf"""
    | Fit statistic | Value |
    |---|---|
    | Observations | {int(ols_results.nobs):,} |
    | R-squared | {ols_results.rsquared:.4f} |
    | Adjusted R-squared | {ols_results.rsquared_adj:.4f} |
    | F-statistic | {ols_results.fvalue:.2f} (p < 0.001) |
    """)

    mo.vstack([
        _fit_card,
        mo.accordion({
            "Full regression table": mo.Html(
                getattr(ols_results, "summary_html", None)
                or ols_results.summary().as_html()
            )
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows.** The linear model explains 15.4% of the variation in hourly delay (R² 0.154, adjusted 0.153) across 236,671 observations, with an F statistic of 193.9. That is a real signal rather than noise, but most of the variation is left unexplained.

    **Why it matters.** The specification is a useful floor, not a serious forecasting model. It shows that time, route and weather variables carry genuine information, while leaving room for the non-linearities that tree models can capture. The gap between this result and the next is the value added by flexible machine learning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.3 Gradient boosting benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second benchmark is gradient boosting, which learns non-linear effects and interactions without being told their form. For each prediction task, XGBoost and LightGBM are tuned by automatic hyperparameter search (Optuna) using cross-validation on the training set only, then scored once on the held-out test set.

    Tuning is not only a prediction exercise. Three of these models estimate delay conditional on the controls, and three estimate a geometry measure conditional on the same controls. Those six models are the nuisance functions of the causal model in section 6. The better they predict, the more completely they absorb confounding, which is why the tuned learners are reused there rather than refitted with defaults. A model that predicts delay well but treatment poorly would leave part of the confounding in the residual and bias the causal estimate.

    The table below reports the two all-feature models. The tabs that follow break the same comparison down by task: delay with one geometry variable excluded, and the geometry variables themselves, which is exactly the nuisance work the causal model in section 6 requires.
    """)
    return


@app.cell
def _(
    LGBMRegressor,
    XGBRegressor,
    load_csv,
    ml_test_df,
    ml_train_df,
    outcome_var,
    pd,
    save_local,
    tune_nuisance,
):
    # --- PREDICTIVE BENCHMARK: ALL VARIABLES -> mean_delay (XGBoost vs LIGHTGBM) ---
    # WHAT: Tune one XGBoost and one LightGBM regressor predicting `mean_delay` from the
    #       FULL feature set (route geometry, scheduled speed, weather, and all
    #       hour/dow/operator/line/condition dummies). Nothing is excluded, unlike the
    #       DML nuisance models which each drop one treatment variable.
    # WHY: Benchmarks the predictive ceiling of the panel: how well can hourly delay be
    #      predicted when every variable is available as a feature?
    #
    # CACHED RESULT: fitting these two 250-tree models over ~165k x 225 features takes
    # many minutes inside the browser (no BLAS/OpenMP in WASM), so the finished table is
    # stored as lpip_results/full_model_performance.csv and loaded when it exists. Set
    # force_refit=True (or delete that CSV) to recompute it.
    force_refit = False

    full_model_performance_df = None
    if not force_refit:
        try:
            full_model_performance_df = load_csv("full_model_performance.csv")
            print(" Loaded full-model performance from lpip_results/full_model_performance.csv")
        except Exception:
            full_model_performance_df = None

    if full_model_performance_df is None:
        # 1) Tune XGBoost using ALL variables (Optuna 3-fold CV; cached in lpip_results/tune_xgb_all.json)
        best_xgb_all_params, best_xgb_all_score = tune_nuisance(
            model="xgboost",
            df=ml_train_df,
            target_col=outcome_var,            # "mean_delay"
            exclude_cols=[],                   # keep ALL variables as features
            cache_name="tune_xgb_all",
            force_retune=False,
            n_trials=100,
            random_state=42,
        )

        # 2) Tune LightGBM using ALL variables (cached in lpip_results/tune_lgb_all.json)
        best_lgb_all_params, best_lgb_all_score = tune_nuisance(
            model="lightgbm",
            df=ml_train_df,
            target_col=outcome_var,
            exclude_cols=[],
            cache_name="tune_lgb_all",
            force_retune=False,
            n_trials=100,
            random_state=42,
        )

        # 3) Fit the tuned full models on the training split and score them on the held-out
        #    test split. These numbers feed the XGBoost vs LightGBM comparison table.
        from sklearn.metrics import root_mean_squared_error as _rmse_all, r2_score as _r2_all, mean_absolute_error as _mae_all

        _Xtr = ml_train_df.drop(columns=[outcome_var]).to_numpy()
        _ytr = ml_train_df[outcome_var].to_numpy()
        _Xte = ml_test_df.drop(columns=[outcome_var]).to_numpy()
        _yte = ml_test_df[outcome_var].to_numpy()

        xgb_all = XGBRegressor(**best_xgb_all_params)
        xgb_all.fit(_Xtr, _ytr)
        _xgb_preds = xgb_all.predict(_Xte)

        lgb_all = LGBMRegressor(**best_lgb_all_params)
        lgb_all.fit(_Xtr, _ytr)
        _lgb_preds = lgb_all.predict(_Xte)

        full_model_performance_df = pd.DataFrame(
            [
                {
                    "Model": "XGBoost",
                    "Feature Set": "All variables",
                    "CV RMSE (train)": round(abs(best_xgb_all_score), 4),
                    "Test RMSE": round(float(_rmse_all(_yte, _xgb_preds)), 4),
                    "Test MAE": round(float(_mae_all(_yte, _xgb_preds)), 4),
                    "Test R2": round(float(_r2_all(_yte, _xgb_preds)), 4),
                },
                {
                    "Model": "LightGBM",
                    "Feature Set": "All variables",
                    "CV RMSE (train)": round(abs(best_lgb_all_score), 4),
                    "Test RMSE": round(float(_rmse_all(_yte, _lgb_preds)), 4),
                    "Test MAE": round(float(_mae_all(_yte, _lgb_preds)), 4),
                    "Test R2": round(float(_r2_all(_yte, _lgb_preds)), 4),
                },
            ]
        )
        if save_local("full_model_performance.csv", full_model_performance_df.to_csv(index=False)):
            print(" Saved full-model performance to lpip_results/full_model_performance.csv")

    print(full_model_performance_df.to_string(index=False))
    return (full_model_performance_df,)


@app.cell
def _(
    best_lgb_lc,
    best_lgb_ld,
    best_lgb_ls,
    best_lgb_mc,
    best_lgb_md,
    best_lgb_ms,
    best_xgb_lc,
    best_xgb_ld,
    best_xgb_ls,
    best_xgb_mc,
    best_xgb_md,
    best_xgb_ms,
    data_url,
    full_model_performance_df,
    load_csv,
    ml_test_df,
    ml_train_df,
    mo,
    outcome_var,
    pd,
    save_local,
    treatment_var1,
    treatment_var2,
    treatment_var3,
):
    from sklearn.base import clone as _clone
    from sklearn.metrics import root_mean_squared_error as _rmse, r2_score as _r2

    # Nuisance estimation tasks: mapping targets, controls/exclusions, and tuned models
    nuisance_eval_tasks = [
        {
            "target_variable": "mean_delay (excl. circuity)",
            "target_col": outcome_var,
            "exclude_cols": [treatment_var1],
            "xgb_model": best_xgb_lc,
            "lgb_model": best_lgb_lc,
        },
        {
            "target_variable": "mean_delay (excl. stop spacing)",
            "target_col": outcome_var,
            "exclude_cols": [treatment_var2],
            "xgb_model": best_xgb_ls,
            "lgb_model": best_lgb_ls,
        },
        {
            "target_variable": "mean_delay (excl. total dist)",
            "target_col": outcome_var,
            "exclude_cols": [treatment_var3],
            "xgb_model": best_xgb_ld,
            "lgb_model": best_lgb_ld,
        },
        {
            "target_variable": "circuity_ratio",
            "target_col": treatment_var1,
            "exclude_cols": [outcome_var],
            "xgb_model": best_xgb_mc,
            "lgb_model": best_lgb_mc,
        },
        {
            "target_variable": "avg_stop_spacing_km",
            "target_col": treatment_var2,
            "exclude_cols": [outcome_var],
            "xgb_model": best_xgb_ms,
            "lgb_model": best_lgb_ms,
        },
        {
            "target_variable": "total_dist_km",
            "target_col": treatment_var3,
            "exclude_cols": [outcome_var],
            "xgb_model": best_xgb_md,
            "lgb_model": best_lgb_md,
        },
    ]

    def evaluate_models_on_test(
        tasks: list[dict],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_name: str = "accuracy_comparison_xgb_vs_lgb_v2.csv",
        rerun: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluates XGBoost and LightGBM nuisance models on the test set using RMSE and R-squared.
        Loads cached results from the notebook's results folder (local file or WASM
        URL on GitHub Pages) to avoid re-fitting; recomputes when unavailable.
        """
        if not rerun:
            try:
                _cached = load_csv(cache_name)
                print(f" Loaded accuracy comparison from {data_url / cache_name}")
                return _cached
            except Exception:
                pass

        results = []
        for task in tasks:
            t_var = task["target_variable"]
            t_col = task["target_col"]
            exclude = task["exclude_cols"]
            feature_cols = [c for c in train_df.columns if c not in exclude and c != t_col]

            X_train = train_df[feature_cols].to_numpy()
            y_train = train_df[t_col].to_numpy()
            X_test = test_df[feature_cols].to_numpy()
            y_test = test_df[t_col].to_numpy()

            # Fit and evaluate XGBoost
            xgb = _clone(task["xgb_model"])
            xgb.fit(X_train, y_train)
            xgb_preds = xgb.predict(X_test)
            xgb_rmse = float(_rmse(y_test, xgb_preds))
            xgb_r2 = float(_r2(y_test, xgb_preds))

            # Fit and evaluate LightGBM
            lgb = _clone(task["lgb_model"])
            lgb.fit(X_train, y_train)
            lgb_preds = lgb.predict(X_test)
            lgb_rmse = float(_rmse(y_test, lgb_preds))
            lgb_r2 = float(_r2(y_test, lgb_preds))

            # Improvement of LightGBM over XGBoost (positive = LightGBM better)
            lgb_improvement = round(((xgb_rmse - lgb_rmse) / xgb_rmse) * 100, 2)
            winner = "LightGBM" if lgb_rmse < xgb_rmse else ("XGBoost" if xgb_rmse < lgb_rmse else "Tie")

            results.append({
                "Target Variable": t_var,
                "Target Column": t_col,
                "XGBoost Test RMSE": round(xgb_rmse, 4),
                "LightGBM Test RMSE": round(lgb_rmse, 4),
                "XGBoost Test R2": round(xgb_r2, 4),
                "LightGBM Test R2": round(lgb_r2, 4),
                "Best Model": winner,
                "LGB Improvement over XGB (%)": lgb_improvement,
            })

        df_out = pd.DataFrame(results)
        if save_local(cache_name, df_out.to_csv(index=False)):
            print(f" Saved accuracy comparison to {data_url / cache_name}")
        else:
            print(" Skipped saving accuracy comparison (read-only location)")
        return df_out

    accuracy_comparison_df = evaluate_models_on_test(
        nuisance_eval_tasks, ml_train_df, ml_test_df, rerun=False
    )

    # --- Append the all-features full models (tuned and scored in the predictive tuning cell) ---
    _fmp = full_model_performance_df.set_index("Model")
    _xgb_r_all = float(_fmp.loc["XGBoost", "Test RMSE"])
    _lgb_r_all = float(_fmp.loc["LightGBM", "Test RMSE"])
    _xgb_r2_all = float(_fmp.loc["XGBoost", "Test R2"])
    _lgb_r2_all = float(_fmp.loc["LightGBM", "Test R2"])
    _lgb_impr_all = round(((_xgb_r_all - _lgb_r_all) / _xgb_r_all) * 100, 2)
    _best_all = "LightGBM" if _lgb_r_all < _xgb_r_all else ("XGBoost" if _xgb_r_all < _lgb_r_all else "Tie")
    accuracy_comparison_df = pd.concat(
        [
            accuracy_comparison_df,
            pd.DataFrame(
                [
                    {
                        "Target Variable": "mean_delay (all features)",
                        "Target Column": outcome_var,
                        "XGBoost Test RMSE": _xgb_r_all,
                        "LightGBM Test RMSE": _lgb_r_all,
                        "XGBoost Test R2": _xgb_r2_all,
                        "LightGBM Test R2": _lgb_r2_all,
                        "Best Model": _best_all,
                        "LGB Improvement over XGB (%)": _lgb_impr_all,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # Tabbed interface for in-cell interactive browsing
    _tabs_dict = {
        "All Targets": mo.ui.table(accuracy_comparison_df, selection=None),
    }
    for _idx, _row in accuracy_comparison_df.iterrows():
        _tabs_dict[_row["Target Variable"]] = mo.ui.table(pd.DataFrame([_row]), selection=None)

    interactive_accuracy_tabs = mo.ui.tabs(_tabs_dict)
    interactive_accuracy_tabs
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What this shows.** Both gradient boosting models roughly double the explanatory power of the linear benchmark. LightGBM reaches a test R² of 0.39 and an RMSE of 5.02 minutes with all features, against 0.36 and 5.13 for XGBoost. LightGBM wins all seven tasks, although the gain on delay itself is small at about 2% lower RMSE. The larger gains, 25% on stop spacing and 19% on circuity, appear where the target is close to a deterministic function of the timetable.

    **Why it matters.** The ceiling is the important number. Even with 224 predictors, about 61% of hourly delay variation remains unexplained, which points to causes the data does not record: incidents, roadworks, breakdowns, driver behaviour and passenger demand. Punctuality will not be fixed by better prediction alone. The causal analysis in section 6 narrows the question to the part of the variation that route design can actually influence.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.4 What would this service do? An interactive check

    The models above are benchmarks, not a tool. This section turns the best one, LightGBM with all features, into a simple check: choose an hour, a day, an operator, a route and a set of conditions, and see the delay the model expects, with an interval that reflects how wrong it usually is.

    The range around the prediction reflects how widely this model's errors vary. It is an illustration of model uncertainty, not a forecast for a specific journey.

    Prediction is where this report started, and it is not where it ends. A model can tell us which hours and routes are likely to be late; it cannot tell us what would happen if a route were redesigned. That is the causal question in section 6.
    """)
    return


@app.cell
def _(data_url, ml_train_df, outcome_var, pd, sys):
    import lightgbm as lgb

    _model_path = data_url / "lgb_full_model.txt"
    if sys.platform == "emscripten":
        from pyodide.http import open_url

        lgb_predictor = lgb.Booster(model_str=open_url(str(_model_path)).read())
    else:
        lgb_predictor = lgb.Booster(model_file=str(_model_path))

    _feature_columns = [c for c in ml_train_df.columns if c != outcome_var]
    _sample = ml_train_df.sample(n=20000, random_state=42)
    _sample_pred = lgb_predictor.predict(_sample[_feature_columns].to_numpy())
    lgb_residuals_sample = pd.Series(
        _sample[outcome_var].to_numpy() - _sample_pred,
        name="residual",
    ).sample(n=10000, random_state=42)
    print(f"Loaded prediction model and computed {len(lgb_residuals_sample):,} residuals from the training sample.")
    return lgb_predictor, lgb_residuals_sample


@app.cell
def _(DOW_ORDER, hourly_dfilter, mo, model_df):
    _hours = sorted(int(h) for h in model_df["hour"].dropna().unique())
    _operator_names = hourly_dfilter.groupby("operator")["operator_name"].first().to_dict()
    _operator_options = {name: code for code, name in sorted(_operator_names.items(), key=lambda kv: kv[1])}
    _lines = sorted(model_df["line"].dropna().unique())
    _conditions = sorted(model_df["condition"].dropna().unique())
    _default_operator = model_df["operator"].value_counts().index[0]


    def _num(col, step, label):
        return mo.ui.number(
            start=float(model_df[col].min()),
            stop=float(model_df[col].quantile(0.99)),
            step=step,
            value=round(float(model_df[col].median()), 2),
            label=label,
        )


    pred_hour = mo.ui.dropdown(options=_hours, value=12, label="Hour")
    pred_dow = mo.ui.dropdown(options=DOW_ORDER, value="Friday", label="Day of week")
    pred_operator = mo.ui.dropdown(options=_operator_options, value=_operator_names[_default_operator], label="Operator")
    pred_line = mo.ui.dropdown(options=_lines, value=model_df["line"].value_counts().index[0], label="Route line", searchable=True)
    pred_condition = mo.ui.dropdown(
        options=_conditions,
        value="Overcast" if "Overcast" in _conditions else _conditions[0],
        label="Weather",
    )

    pred_planned_stops = _num("planned_stops", 1, "Planned stops")
    pred_total_dist = _num("total_dist_km", 0.5, "Total distance (km)")
    pred_max_reach = _num("max_reach_km", 0.5, "Maximum reach (km)")
    pred_sched_duration = _num("sched_duration_min", 1, "Scheduled duration (min)")
    pred_origin_distance = _num("origin_dist_cardiff_km", 1, "Distance from Cardiff (km)")
    pred_temperature = _num("temperature_c", 0.5, "Temperature (C)")
    pred_precipitation = _num("precipitation_mm", 0.1, "Rainfall (mm)")
    pred_humidity = _num("relative_humidity_pct", 1, "Humidity (%)")
    pred_wind = _num("wind_speed_kmh", 1, "Wind speed (km/h)")

    mo.vstack([
        mo.md("**Service**"),
        mo.hstack([pred_hour, pred_dow, pred_operator, pred_line, pred_condition], wrap=True, gap=0.8, justify="start"),
        mo.md("**Route geometry and weather**"),
        mo.hstack([pred_planned_stops, pred_total_dist, pred_max_reach, pred_sched_duration], wrap=True, gap=0.8, justify="start"),
        mo.hstack([pred_origin_distance, pred_temperature, pred_precipitation, pred_humidity, pred_wind], wrap=True, gap=0.8, justify="start"),
    ])
    return (
        pred_condition,
        pred_dow,
        pred_hour,
        pred_humidity,
        pred_line,
        pred_max_reach,
        pred_operator,
        pred_origin_distance,
        pred_planned_stops,
        pred_precipitation,
        pred_sched_duration,
        pred_temperature,
        pred_total_dist,
        pred_wind,
    )


@app.cell(hide_code=True)
def _(
    dum_vars,
    lgb_predictor,
    lgb_residuals_sample,
    ml_train_df,
    mo,
    model_df,
    outcome_var,
    pd,
    pred_condition,
    pred_dow,
    pred_hour,
    pred_humidity,
    pred_line,
    pred_max_reach,
    pred_operator,
    pred_origin_distance,
    pred_planned_stops,
    pred_precipitation,
    pred_sched_duration,
    pred_temperature,
    pred_total_dist,
    pred_wind,
):
    _input = {
        "hour": pred_hour.value,
        "dow": pred_dow.value,
        "operator": pred_operator.value,
        "line": pred_line.value,
        "condition": pred_condition.value,
        "planned_stops": pred_planned_stops.value,
        "total_dist_km": pred_total_dist.value,
        "max_reach_km": pred_max_reach.value,
        "sched_duration_min": pred_sched_duration.value,
        "origin_dist_cardiff_km": pred_origin_distance.value,
        "temperature_c": pred_temperature.value,
        "precipitation_mm": pred_precipitation.value,
        "relative_humidity_pct": pred_humidity.value,
        "wind_speed_kmh": pred_wind.value,
    }

    for _key, _value in list(_input.items()):
        if _value is None and _key in model_df.columns:
            if model_df[_key].dtype.kind in "if":
                _input[_key] = float(model_df[_key].median())
            else:
                _input[_key] = model_df[_key].mode().iloc[0]

    # Derived geometry: these move automatically with the inputs above.
    _input["avg_stop_spacing_km"] = float(_input["total_dist_km"]) / max(1.0, float(_input["planned_stops"]) - 1.0)
    _input["circuity_ratio"] = float(_input["total_dist_km"]) / max(0.1, float(_input["max_reach_km"]))
    _input["sched_speed_kmh"] = float(_input["total_dist_km"]) / (max(1.0, float(_input["sched_duration_min"])) / 60.0)

    _input_row = pd.DataFrame([_input])
    _dummies = pd.get_dummies(_input_row, columns=dum_vars, drop_first=False, dtype=int, prefix_sep="_")
    _feature_cols = [c for c in ml_train_df.columns if c != outcome_var]
    _X = _dummies.reindex(columns=_feature_cols, fill_value=0)

    prediction_minutes = float(lgb_predictor.predict(_X.to_numpy())[0])
    _ci_low = float(prediction_minutes + lgb_residuals_sample.quantile(0.10))
    _ci_high = float(prediction_minutes + lgb_residuals_sample.quantile(0.90))

    mo.md(rf"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 2px 0;">
      <div style="flex:1; min-width:220px; border:1px solid #dbeafe; background:#f8fafc; border-radius:14px; padding:18px 22px;">
        <div style="font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#64748b;">Predicted delay</div>
        <div style="font-size:46px; font-weight:700; line-height:1.1; color:#0f172a;">{prediction_minutes:.1f}<span style="font-size:18px; font-weight:500;"> minutes</span></div>
      </div>
      <div style="flex:1; min-width:220px; border:1px solid #e2e8f0; background:#ffffff; border-radius:14px; padding:18px 22px;">
        <div style="font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#64748b;">Likely range</div>
        <div style="font-size:30px; font-weight:650; line-height:1.25; color:#0f172a;">{_ci_low:.1f} to {_ci_high:.1f}<span style="font-size:16px; font-weight:500;"> minutes</span> <span style="font-size:13px; font-weight:500; color:#64748b;">(10th to 90th percentile)</span></div>
      </div>
    </div>
    <p style="color:#64748b; font-size:13px; margin-top:6px;">0 means on time; negative values mean early.</p>
    <p style="color:#64748b; font-size:13px; margin-top:2px;">Derived: stop spacing {_input["avg_stop_spacing_km"]:.2f} km · circuity {_input["circuity_ratio"]:.2f} · scheduled speed {_input["sched_speed_kmh"]:.1f} km/h</p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Causal findings: does route geometry cause delay?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prediction and causation answer different questions. A model that predicts delay well may still be useless for policy, because it does not say what happens to delay if a route is redesigned. This section asks the causal question directly.

    The three treatments are the geometry choices a network planner can actually change: how indirect a route is, how far apart its stops are, and how long it is. They are fixed before the hour being observed, which makes them plausible causes of delay rather than consequences of it. Weather and time of day matter for prediction, but they are not levers a transport authority can pull.

    The problem is confounding. Long routes differ from short ones not only in length but in the operators that run them, the hours they serve, the terrain they cross and the timetable padding they carry. A naive regression would attribute all of that to geometry. OLS in particular forces every control to enter linearly and additively, so when delay responds non-linearly to hour, weather or route length, the geometry coefficients absorb the misfit. Adding interactions does not solve this: it requires knowing which interactions matter, and the model still imposes a functional form.

    Double machine learning avoids that by learning the two nuisance functions with gradient boosting, then estimating the geometry effect from their residuals after cross-fitting. Section 6.2 shows how much the answer changes: the linear model finds a large, significant circuity effect and a positive distance effect; the flexible causal model finds no circuity effect and a negative distance effect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Method: double machine learning

    The analysis uses the partially linear regression form of double machine learning (Chernozhukov et al., 2018). The outcome and the treatment are each split into a part explained by the controls and a residual part:

    $$Y = \theta D + g(X) + U, \qquad D = m(X) + V$$

    Here $Y$ is hourly mean delay, $D$ is one route geometry measure at a time, and $X$ is the full set of controls: weather, hour, day of week, operator and route. The nuisance functions $g(X)$ and $m(X)$ are learned with gradient boosting, and $\theta$ is estimated from their residuals using three-fold cross-fitting with five repetitions. Cross-fitting removes the bias that would arise if the same observations were used both to learn the nuisance functions and to estimate the effect.

    The quantity $\theta$ has a direct reading: the change in hourly delay, in minutes, associated with a one-unit change in the geometry measure, holding weather, time, operator and route constant. Three treatments are tested separately:

    | Treatment | Unit | Interpretation |
    |---|---|---|
    | `circuity_ratio` | ratio | route length relative to the straight-line distance to the farthest point reached |
    | `avg_stop_spacing_km` | km | average distance between consecutive stops |
    | `total_dist_km` | km | cumulative length of the route |

    Each treatment is estimated twice, once with XGBoost nuisance models and once with LightGBM, so the result can be judged on whether the two learners agree rather than on a single specification. A coefficient counts as evidence only if its 95% confidence interval excludes zero and both learners point the same way.
    """)
    return


@app.cell
def _(
    best_lgb_lc,
    best_lgb_ld,
    best_lgb_ls,
    best_lgb_mc,
    best_lgb_md,
    best_lgb_ms,
    best_xgb_lc,
    best_xgb_ld,
    best_xgb_ls,
    best_xgb_mc,
    best_xgb_md,
    best_xgb_ms,
    data_url,
    load_csv,
    ml_train_df,
    outcome_var,
    save_local,
    treatment_var1,
    treatment_var2,
    treatment_var3,
):
    # 1. Define clean modeling data (dropna to ensure complete cases for DML)

    # 2. Package into DoubleMLData
    # - y_col: outcome (e.g. 'mean_delay')
    # - d_cols: treatment variable (can be continuous or binary, e.g. 'precipitation_mm')
    # - x_cols: confounders/controls (all remaining numerical and dummy columns)
    def perform_dml(out_var, treatment_var, _ml_l, _ml_m, data_df=None):
        # Imported lazily: dml_result loads cached summaries for every published
        # run, and doubleml is a slow micropip install in Pyodide/WASM.
        import doubleml as dml

        _df = data_df if data_df is not None else ml_train_df
        feature_cols = [c for c in _df.columns if c not in [out_var, treatment_var]]

        dml_data = dml.DoubleMLData(
            _df,
            y_col=out_var,
            d_cols=treatment_var,
            x_cols=feature_cols
        )

        # 3. Instantiate the learner for BOTH ml_l and ml_m

        # 4. Initialize and fit DoubleMLPLR
        dml_plr = dml.DoubleMLPLR(
            obj_dml_data=dml_data,
            ml_l=_ml_l, # outcome for 
            ml_m=_ml_m,
            n_folds=3,
            n_rep=5,
            score="partialling out"
        )

        dml_plr.fit()
        return dml_plr.summary


    def dml_result(name, out_var, treatment_var, ml_l, ml_m, rerun=False):
        """Load a saved DML summary from the notebook's results folder, or rerun and save it.

        Reads `lpip_results/` next to the notebook: a plain file locally, and the
        notebook's public URL when running as a WASM app (e.g. GitHub Pages).
        Unlike mo.persistent_cache, this only depends on the cached file:
        small upstream changes do not trigger a rerun. Set rerun=True to refit.
        """
        _file = f"{name}.csv"
        if not rerun:
            try:
                _cached = load_csv(_file, index_col=0)
                print(f" Loaded DML result '{name}' from {data_url / _file}")
                return _cached
            except Exception:
                pass
        summary = perform_dml(out_var, treatment_var, ml_l, ml_m)
        if save_local(_file, summary.to_csv()):
            print(f" Saved DML result '{name}' to {data_url / _file}")
        else:
            print(f" Skipped saving DML result '{name}' (read-only location)")
        return summary


    # --- XGBoost learners ---
    dml_xgb_circuity = dml_result("dml_xgb_circuity", outcome_var, treatment_var1, best_xgb_lc, best_xgb_mc, rerun=False)
    dml_xgb_stop = dml_result("dml_xgb_stop", outcome_var, treatment_var2, best_xgb_ls, best_xgb_ms, rerun=False)
    dml_xgb_dist = dml_result("dml_xgb_dist", outcome_var, treatment_var3, best_xgb_ld, best_xgb_md, rerun=False)

    # --- LightGBM learners ---
    dml_lgb_circuity = dml_result("dml_lgb_circuity", outcome_var, treatment_var1, best_lgb_lc, best_lgb_mc, rerun=False)
    dml_lgb_stop = dml_result("dml_lgb_stop", outcome_var, treatment_var2, best_lgb_ls, best_lgb_ms, rerun=False)
    dml_lgb_dist = dml_result("dml_lgb_dist", outcome_var, treatment_var3, best_lgb_ld, best_lgb_md, rerun=False)
    return (
        dml_lgb_circuity,
        dml_lgb_dist,
        dml_lgb_stop,
        dml_xgb_circuity,
        dml_xgb_dist,
        dml_xgb_stop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Results
    """)
    return


@app.cell
def _(dml_summary, mo, ols_results, pd):
    _ols = ols_results.coefficients
    _dml = dml_summary.set_index(["Learner", "Treatment"])


    def _fmt(_coef, _p):
        _p_txt = "<0.001" if _p < 0.001 else f"{_p:.3f}"
        return f"{_coef:.2f} ({_p_txt})"


    _rows = []
    for _name, _label, _xgb, _lgb in [
        ("circuity_ratio", "Circuity ratio", "xgb_circuity", "lgb_circuity"),
        ("avg_stop_spacing_km", "Average stop spacing (km)", "xgb_stop", "lgb_stop"),
        ("total_dist_km", "Total route distance (km)", "xgb_dist", "lgb_dist"),
    ]:
        _rows.append({
            "Treatment": _label,
            "OLS (p)": _fmt(_ols[_name]["coef"], _ols[_name]["p"]),
            "DML XGB (p)": _fmt(_dml.loc[(_xgb, _name), "coef"], _dml.loc[(_xgb, _name), "p-value"]),
            "DML LGB (p)": _fmt(_dml.loc[(_lgb, _name), "coef"], _dml.loc[(_lgb, _name), "p-value"]),
        })

    ols_vs_dml = pd.DataFrame(_rows)

    mo.vstack([
        mo.md(r"""
    The linear model from section 5.2 is the natural comparison. It is adjusted for the same controls, but it fits one coefficient per variable. The table sets its geometry coefficients beside the causal estimates.
    """),
        ols_vs_dml,
        mo.md(r"""
    For circuity the OLS estimate is more than five times its standard error, yet the causal confidence interval spans zero for both learners. For total distance the sign reverses: OLS says longer routes are later, the causal model says slightly earlier. That reversal is the clearest evidence in the report that a significant regression coefficient is not a causal effect.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Each row is one learner and one treatment. `coef` is the estimated causal effect in minutes per unit of the treatment, `std err` describes its precision, `p-value` is the two-sided test, and `95% CI` gives the confidence interval.
    """)
    return


@app.cell
def _(
    dml_lgb_circuity,
    dml_lgb_dist,
    dml_lgb_stop,
    dml_xgb_circuity,
    dml_xgb_dist,
    dml_xgb_stop,
    pd,
):
    ## concat results: XGBoost + LightGBM, one row per learner x treatment
    dml_summary = (
        pd.concat(
            {
                "xgb_circuity": dml_xgb_circuity,
                "xgb_stop": dml_xgb_stop,
                "xgb_dist": dml_xgb_dist,
                "lgb_circuity": dml_lgb_circuity,
                "lgb_stop": dml_lgb_stop,
                "lgb_dist": dml_lgb_dist,
            },
            axis=0,
        )
        .rename_axis(["Learner", "Treatment"])
        .reset_index()
        .rename(columns={"P>|t|": "p-value", "2.5 %": "CI lower", "97.5 %": "CI upper"})
        .round(3)
    )

    # Compact view: keep the evidence columns visible in a narrow report page.
    _dml_display = dml_summary[["Learner", "Treatment", "coef", "std err", "p-value"]].copy()
    _dml_display["95% CI"] = (
        dml_summary["CI lower"].map(lambda v: f"{v:.3f}")
        + " to "
        + dml_summary["CI upper"].map(lambda v: f"{v:.3f}")
    )
    _dml_display
    return (dml_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    **What this shows.** Circuity and stop spacing are not distinguishable from zero: every confidence interval spans zero, and for stop spacing the two learners even disagree on the sign. Total route distance is the exception: both learners agree on a negative and statistically significant effect, at 0.086 minutes per kilometre with XGBoost (95% confidence interval −0.125 to −0.047) and 0.076 with LightGBM (−0.117 to −0.036).

    **Why it matters.** The effect is real but small. A route 10 kilometres longer is associated with roughly 0.8 minutes less delay in a given hour, against a standard deviation of 6.5 minutes in the outcome. The negative sign is best read as timetable slack rather than an operational advantage: longer rural routes tend to carry more recovery time per stop, which offsets the accumulated journey time. It is not a case for lengthening routes.

    The policy conclusion is the opposite of the common intuition that indirect, winding routes are the main source of rural lateness. On this panel, route shape is not the driver. What remains is the timetable, the operating context and the unrecorded incidents that section 5 could not capture.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. What this means for policy
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Five implications follow directly from the results, each tied to evidence in the preceding sections.

    1. **Treat early running as a failure in its own right.** One observation in four is early, morning services have negative median delay, and the earliest Sunday departures run more than four minutes ahead of timetable. A punctuality target built only around lateness will never see this. Hold-back discipline at rural timing points, and an early-share indicator reported alongside on-time performance, would make the problem visible and manageable.

    2. **Give Friday afternoon its own running times.** The weekly peak is Friday at 15:00, at 2.6 minutes median delay, against about 1.6 minutes at weekday midday. A separate Friday schedule with additional recovery time on the worst corridors would address a recurring pattern rather than a random bad day.

    3. **Judge operators on reliability, not on averages.** The major operators differ by less than 0.2 minutes in median delay but by around a minute in interquartile range. Contract metrics based on the 80th or 90th percentile of excess wait time would reward predictable service rather than a good average that hides bad days.

    4. **Do not expect route redesign to fix punctuality.** Circuity and stop spacing have no detectable causal effect. The only significant geometry effect is small, and its sign reflects timetable slack rather than an advantage of longer routes. The levers that remain are timetable padding, traffic management, bus priority and operational control.

    5. **Do not plan around weather, and do close the data gap.** Clear, overcast and rainy hours differ by about a quarter of a minute in median delay, and that raw gap is confounded with time of day. What the models cannot explain, roughly 60% of the variation, points instead to incidents, roadworks, breakdowns and demand, none of which the panel records. Linking the panel to operational incident logs is the single most valuable next data investment.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Reproducibility
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The notebook, the hourly panel and the cached results are public on GitHub: [github.com/mustafaslanCoto/marimo-outputs](https://github.com/mustafaslanCoto/marimo-outputs). The published version of this report is at [mustafaslancoto.github.io/marimo-outputs/apps/panctuality_report.html](https://mustafaslancoto.github.io/marimo-outputs/apps/panctuality_report.html). Running the notebook reproduces every figure and table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.1 Automatic tuning of the nuisance models

    The nuisance functions in section 6 are not specified by hand. For each outcome and treatment combination, a gradient boosting model is tuned over the parameters that control tree depth, learning rate, regularisation and sampling, using cross-validation on the training set only. Tuning uses a pruning search (Optuna) that stops unpromising trials early, and every finished search is cached as JSON in `lpip_results/`.

    The cell below holds the shared tuning logic. The two subsections that follow apply it to the six XGBoost and six LightGBM models and report the selected parameters.
    """)
    return


@app.cell
def _(data_url, load_json, np, pd, save_local):
    from typing import Any, Dict, Tuple
    import json as _json
    from sklearn.model_selection import KFold
    from sklearn.metrics import root_mean_squared_error
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor


    def _normalize_model_key(model: str | type) -> str:
        """Normalize a model name or estimator class to 'xgboost' or 'lightgbm'."""
        if model is XGBRegressor:
            return "xgboost"
        if model is LGBMRegressor:
            return "lightgbm"
        if isinstance(model, str):
            aliases = {
                "xgb": "xgboost",
                "xgboost": "xgboost",
                "lgb": "lightgbm",
                "lgbm": "lightgbm",
                "lightgbm": "lightgbm",
            }
            key = model.strip().lower()
            if key in aliases:
                return aliases[key]
        raise ValueError(
            f"Unsupported model {model!r}. Expected 'xgboost' ('xgb') or 'lightgbm' ('lgb'/'lgbm')."
        )


    _MODEL_SPECS = {
        "xgboost": {
            "estimator": XGBRegressor,
            "fixed_params": {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": 6,
            },
            "suggest_params": lambda trial: {
                "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256]),
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "max_leaves": trial.suggest_int("max_leaves", 15, 128),
                "min_child_weight": trial.suggest_int("min_child_weight", 30, 150),
                "subsample": trial.suggest_float("subsample", 0.50, 0.99),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.99),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.50, 0.99),
                "gamma": trial.suggest_float("gamma", 0.1, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            },
        },
        "lightgbm": {
            "estimator": LGBMRegressor,
            "fixed_params": {
                "objective": "regression",
                "n_jobs": 6,
                "verbose": -1,
            },
            "suggest_params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_int("num_leaves", 15, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.50, 0.99),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.99),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            },
        },
    }


    def tune_nuisance(
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: list[str],
        model: str | type = "xgboost",
        cache_name: str | None = None,
        force_retune: bool = False,
        n_trials: int = 100,
        random_state: int = 42,
        n_startup_trials: int = 3,
        show_progress_bar: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Unified Optuna tuning function for XGBoost and LightGBM nuisance regressors.

        Pass `model="xgboost"` (or `"xgb"` / `XGBRegressor`) or `model="lightgbm"`
        (or `"lgb"` / `"lgbm"` / `LGBMRegressor`). If `cache_name` is provided and
        `lpip_results/<cache_name>.json` exists next to the notebook (with
        `force_retune=False`), loads cached parameters without re-running Optuna.
        The cache is a plain file locally and the notebook's public URL when the
        notebook runs as a WASM app (e.g. GitHub Pages).
        """
        model_key = _normalize_model_key(model)
        spec = _MODEL_SPECS[model_key]
        estimator_cls = spec["estimator"]

        # Cache location: the results folder that ships next to the notebook
        # (data_url = mo.notebook_location() / "lpip_results", see the io_helpers cell).
        cache_file = (data_url / f"{cache_name}.json") if cache_name else None

        if cache_file and not force_retune:
            try:
                saved = load_json(f"{cache_name}.json")
                print(
                    f" Loaded cached {model_key} parameters for '{cache_name}' "
                    f"from {cache_file} (Best score: {saved['best_score']:.4f})"
                )
                return saved["best_params"], saved["best_score"]
            except Exception:
                pass  # no cache available where this runs; tune below

        # Imported lazily: cache hits (all published runs) never reach this line,
        # and optuna has no prebuilt Pyodide wheel, so importing it up front makes
        # the WASM app spend minutes installing packages it will not use.
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        feature_cols = [c for c in df.columns if c not in exclude_cols and c != target_col]
        X = df[feature_cols].to_numpy()
        y = df[target_col].to_numpy()

        fixed_params: Dict[str, Any] = {
            **spec["fixed_params"],
            "random_state": random_state,
        }

        def _objective(trial: optuna.Trial) -> float:
            params = {
                **fixed_params,
                **spec["suggest_params"](trial),
            }

            kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                est = estimator_cls(**params)
                est.fit(X[train_idx], y[train_idx])
                preds = est.predict(X[val_idx])

                score = -float(root_mean_squared_error(y[val_idx], preds))
                fold_scores.append(score)

                trial.report(float(np.mean(fold_scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=5)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True),
            pruner=pruner,
        )
        print(f"Running Optuna tuning ({model_key}) for '{cache_name or target_col}' ({n_trials} trials)...")
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        best_full_params = {**fixed_params, **study.best_params}

        if cache_file:
            _payload = _json.dumps(
                {"best_params": best_full_params, "best_score": study.best_value}, indent=2
            )
            if save_local(f"{cache_name}.json", _payload):
                print(f" Saved tuned {model_key} parameters to {cache_file}")
            else:
                print(" Skipped saving tuned parameters (read-only location)")

        return best_full_params, study.best_value


    return LGBMRegressor, XGBRegressor, tune_nuisance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.2 XGBoost nuisance models

    Six models are tuned with XGBoost: three outcome models for delay with one treatment excluded at a time, and three treatment models for the geometry variables. The parameters below are loaded from cache rather than refitted.
    """)
    return


@app.cell
def _(
    XGBRegressor,
    ml_train_df,
    outcome_var,
    treatment_var1,
    treatment_var2,
    treatment_var3,
    tune_nuisance,
):
    best_xgb_params_lc, best_xgb_score_lc = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var1],        # exclude treatment "circuity_ratio"
        cache_name="tune_xgb_outcome_c",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_lc = XGBRegressor(**best_xgb_params_lc)
    print("Best XGBoost Nuisance Model Parameters - circuity - for Outcome (mean_delay):")

    best_xgb_params_ls, best_xgb_score_ls = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var2],        # exclude treatment "avg_stop_spacing_km"
        cache_name="tune_xgb_outcome_s",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_ls = XGBRegressor(**best_xgb_params_ls)
    print("Best XGBoost Nuisance Model Parameters - avg_stop_spacing_km - for Outcome (mean_delay):")

    best_xgb_params_mc, best_xgb_score_mc = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=treatment_var1,            # "circuity_ratio"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_xgb_treatment_c",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_mc = XGBRegressor(**best_xgb_params_mc)
    print("Best XGBoost Nuisance Model Parameters for Treatment (circuity_ratio):")

    best_xgb_params_ms, best_xgb_score_ms = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=treatment_var2,            # "avg_stop_spacing_km"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_xgb_treatment_s",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_ms = XGBRegressor(**best_xgb_params_ms)

    best_xgb_params_ld, best_xgb_score_ld = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var3],        # exclude treatment "total_dist_km"
        cache_name="tune_xgb_outcome_d",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_ld = XGBRegressor(**best_xgb_params_ld)

    print("Best parameters for outcome:", best_xgb_params_ld)
    print("Best score for outcome:", best_xgb_score_ld)

    best_xgb_params_md, best_xgb_score_md = tune_nuisance(
        model="xgboost",
        df=ml_train_df,
        target_col=treatment_var3,            # "total_dist_km"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_xgb_treatment_d",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_xgb_md = XGBRegressor(**best_xgb_params_md)
    return (
        best_xgb_lc,
        best_xgb_ld,
        best_xgb_ls,
        best_xgb_mc,
        best_xgb_md,
        best_xgb_ms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.3 LightGBM nuisance models

    The same six models are tuned again with LightGBM, so that every causal estimate in section 6 is available from two independent learners.
    """)
    return


@app.cell
def _(
    LGBMRegressor,
    ml_train_df,
    outcome_var,
    treatment_var1,
    treatment_var2,
    treatment_var3,
    tune_nuisance,
):
    best_lgb_params_lc, best_lgb_score_lc = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var1],        # exclude treatment "circuity_ratio"
        cache_name="tune_lgb_outcome_c",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_lc = LGBMRegressor(**best_lgb_params_lc)
    print("Best LightGBM Nuisance Model Parameters - circuity - for Outcome (mean_delay):")

    best_lgb_params_ls, best_lgb_score_ls = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var2],        # exclude treatment "avg_stop_spacing_km"
        cache_name="tune_lgb_outcome_s",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_ls = LGBMRegressor(**best_lgb_params_ls)
    print("Best LightGBM Nuisance Model Parameters - avg_stop_spacing_km - for Outcome (mean_delay):")

    best_lgb_params_mc, best_lgb_score_mc = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=treatment_var1,            # "circuity_ratio"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_lgb_treatment_c",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_mc = LGBMRegressor(**best_lgb_params_mc)
    print("Best LightGBM Nuisance Model Parameters for Treatment (circuity_ratio):")

    best_lgb_params_ms, best_lgb_score_ms = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=treatment_var2,            # "avg_stop_spacing_km"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_lgb_treatment_s",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_ms = LGBMRegressor(**best_lgb_params_ms)

    best_lgb_params_ld, best_lgb_score_ld = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=outcome_var,              # "mean_delay"
        exclude_cols=[treatment_var3],        # exclude treatment "total_dist_km"
        cache_name="tune_lgb_outcome_d",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_ld = LGBMRegressor(**best_lgb_params_ld)

    print("Best parameters for outcome:", best_lgb_params_ld)
    print("Best score for outcome:", best_lgb_score_ld)

    best_lgb_params_md, best_lgb_score_md = tune_nuisance(
        model="lightgbm",
        df=ml_train_df,
        target_col=treatment_var3,            # "total_dist_km"
        exclude_cols=[outcome_var],          # exclude outcome "mean_delay"
        cache_name="tune_lgb_treatment_d",
        force_retune=False,
        n_trials=100,
        random_state=42,
    )

    best_lgb_md = LGBMRegressor(**best_lgb_params_md)
    return (
        best_lgb_lc,
        best_lgb_ld,
        best_lgb_ls,
        best_lgb_mc,
        best_lgb_md,
        best_lgb_ms,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### 8.4 How the prediction model and its interval were built

    The model used in section 5.4 is the LightGBM all-feature model, fitted once with the tuned parameters stored in `lpip_results/tune_lgb_all.json`:

    ```python
    params = json.load(open("lpip_results/tune_lgb_all.json"))["best_params"]
    features = [c for c in ml_train_df.columns if c != "mean_delay"]

    model = lgb.LGBMRegressor(**params)
    model.fit(ml_train_df[features], ml_train_df["mean_delay"])
    model.booster_.save_model("lpip_results/lgb_full_model.txt")
    ```

    The interval adds the model's own errors to the point prediction. In the notebook those errors are computed on load from a 20,000-row sample of the training data, and 10,000 are kept:

    ```python
    sample = ml_train_df.sample(n=20000, random_state=42)
    residuals = sample["mean_delay"] - model.predict(sample[features])
    residuals = residuals.sample(n=10000, random_state=42)
    ```

    The 10th and 90th percentiles of those residuals, added to the point prediction, give the range shown in section 5.4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Everything in this report is generated by the cells above. Cached results were produced with the same code on the full training set; deleting `lpip_results/` and setting the refit flags to `True` rebuilds them from the panel.*
    """)
    return


if __name__ == "__main__":
    app.run()
