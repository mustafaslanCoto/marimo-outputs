# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#     "doubleml>=0.11.4",
#     "lightgbm>=4.6.0",
#     "marimo>=0.23.3",
#     "marimo-lens>=0.2.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.4",
#     "optuna>=5.0.0",
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
    # Bus Punctuality Analysis in Rural Wales (Cymru Wledig LPIP)
    ## Multi-Day Hourly Operational Delay & Reliability Dashboard

    **Research Placement:** Cymru Wledig Local Partnership & Innovation Plan (Rural Wales)
    **Target Policy Focus:** Bus Service Punctuality, Headway Vulnerability, and Rural Accessibility
    **Live Telematics Data:** SIRI-VM (Automatic Vehicle Location tracking from local daily archives)
    **Scheduled Timetable:** GTFS (General Transit Feed Specification for Wales)
    **Meteorological Data:** Cardiff Hourly Weather (Temperature, Rain, Wind, Conditions)

    ---
    ### 1. Business & Policy Domain Context

    In urban agglomerations, bus networks operate on high-frequency "turn-up-and-go" schedules where a 5-minute delay is merely an inconvenience. In contrast, **rural Welsh bus networks** (e.g. across Powys, Ceredigion, Gwynedd, Carmarthenshire, and Pembrokeshire) operate under fundamentally different operational constraints:

    1. **Low Frequency & Long Headways:** Rural services often operate on 60-to-120-minute headways. A bus running 10–15 minutes late or failing to connect at a rural interchange leaves passengers stranded for hours.
    2. **Early Departures are Severe Service Failures:** In rural areas, a bus departing ahead of schedule ($< -1\text{ min}$) is worse than a moderate delay, as passengers arriving precisely on time miss their sole transport option.
    3. **UK Department for Transport (DfT) / Traffic Commissioner Standards:**
       - **Early:** $> 1$ minute early ($< -1.0\text{ min}$).
       - **On Time (Compliant Window):** Between 1 minute early and 5 minutes 59 seconds late ($-1.0\text{ to }+5.99\text{ min}$). Target: **95% of services on time**.
       - **Late:** 6 minutes to 14 minutes 59 seconds late ($+6.0\text{ to }+15.0\text{ min}$).
       - **Very Late:** $\ge 15$ minutes late ($> +15.0\text{ min}$), representing severe operational breakdown.
    4. **Predictability vs. Lateness:** A service that is consistently 4 minutes late is reliable and easy to plan around. A service with high variability (wide Inter-Quartile Range, IQR) induces anxiety and modal shift to private cars.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 2. Data Dictionary & Telematics Definitions

    - **`recorded_at`:** UTC timestamp when onboard vehicle telematics recorded GPS position.
    - **`origin_aimed_departure`:** Timetabled departure time from the route's initial terminus.
    - **`line_ref` / `LineRef`:** Commercial bus route number (e.g. `1`, `T4`, `T5`, `X4`, `40`).
    - **`vehicle_ref` / `VehicleRef`:** Unique vehicle tracker ID (e.g. `ACYM-1234`, `A2BV-24`), from which the Operator National Operator Code (NOC) is extracted.
    - **`timestamp_hour`:** Observation timestamp truncated to the operating hour in civil local time (`Europe/London`: BST = UTC+1).
    - **`delay_min`:** Telematics lateness from timetable: $(t_{\text{observed}} - t_{\text{aimed}})$, unwrapped around midnight.
    - **`temperature_c`:** Ambient air temperature (°C) merged from Cardiff meteorological observations.
    - **`precipitation_mm`:** Hourly rainfall amount (mm) merged from Cardiff meteorological observations.
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

    # Live data directory: local siri_zips archive
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
    ---
    ## 2. Timetable Reference: GTFS Schedule Pre-Indexing & Matching Architecture

    ### 🎯 What We Are Doing
    In this stage, we load the Welsh national bus timetable from the reference GTFS directory (`data2/20260520_bus_sch`) and build three high-performance in-memory lookup structures:
    1. **Scheduled Origin Minute (`origin_min`):** For each trip, we identify its very first scheduled stop (`stop_sequence == 1`) and compute its departure time in minutes from midnight (modulo 1440).
    2. **Representative Directional & Fallback Lookups (`rep` and `repfb`):**
       - `rep`: Primary directional match mapping `(agency_noc, route_short_name, origin_min, direction_id) -> trip_id`.
       - `repfb`: Direction-free fallback mapping `(agency_noc, route_short_name, origin_min) -> trip_id_fb`.
    3. **Per-Trip Stop Coordinates & Times (`trip_arrays`):** A fast dictionary mapping each `trip_id` to pre-compiled NumPy arrays of stop latitudes, longitudes, and scheduled timetable minutes `(slat, slon, smin)`.

    ### 💡 Why We Do It This Way
    - **AVL Telemetry Has No GTFS `trip_id`:** Live SIRI-VM feeds provide vehicle locations, lines, and terminal departure times, but do **not** carry GTFS `trip_id` foreign keys. Matching on `(operator, line, origin_min, direction)` reconstructs this connection unambiguously.
    - **Sub-Millisecond Vectorized Distance:** Iterating row-by-row or doing GeoPandas spatial joins on hundreds of thousands of live pings across 1.2M stop records would take hours. Pre-compiling stop arrays allows NumPy to calculate great-circle distances for an entire vehicle journey in milliseconds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Meteorological Integration: Cardiff Hourly Weather Feed

    ### 🎯 What We Are Doing
    We ingest hourly meteorological observations from Cardiff (`data2/cardiff_weather_april_to_september_2026.csv`), including temperature (°C), precipitation (mm), relative humidity (%), wind speed (km/h), and atmospheric conditions (Rain, Overcast, Clear).

    ### 💡 Why We Do It This Way
    - **Weather as an Exogenous Driver of Delay:** In rural Wales, severe weather (heavy rainfall, reduced visibility, gale-force winds) significantly reduces vehicular operating speeds, increases stopping distances, and causes passenger boarding delays.
    - **Hourly Timestamp Key:** Standardizing weather to `timestamp_hour` allows a seamless 1:1 join onto the bus delay panel, enabling direct empirical modeling of how punctuality degrades under adverse weather conditions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. End-to-End Processing Pipeline: Two-Loop Stream Ingestion & Nearest-Stop Delay

    ### 🎯 What We Are Doing
    We execute an end-to-end data processing pipeline that transforms raw daily SIRI-VM archives directly from `data/siri_zips` into an analysis-ready hourly panel:
    1. **Two Nested Loops:**
       - **Outer Loop:** Iterates across daily archives in `data/siri_zips`.
       - **Inner Loop:** Streams each individual 1-minute snapshot CSV file from the ZIP archive into RAM: `Archive: {_src.name}, CSV: {_csv}`.
    2. **Uniform 20% Sampling (`sample_frac=0.20`):** Uniformly samples 20% of rows from every snapshot CSV (`pd.read_csv(_csv).sample(frac=0.20, random_state=42)`) to preserve the complete 24-hour temporal distribution without overwhelming memory.
    3. **Timezone Normalization:** Converts UTC telemetry timestamps into British Summer Time (`Europe/London`), aligning with local clock timetables.
    4. **GTFS Trip Matching:** Links vehicle pings to scheduled trips via `(operator, line, origin_min, direction_id)` with direction-free fallback.
    5. **Nearest-Stop Distance & Filtering:** Computes the great-circle Haversine distance from the vehicle position to every stop on that trip, identifies the nearest stop, and retains observations within $200\text{ m}$.
    6. **Authentic Schedule Delay Calculation:**
       $$\text{delay\_min} = (t_{\text{observed, local}} - t_{\text{scheduled at closest stop}} + 720) \% 1440 - 720$$
    7. **Statutory Punctuality Classification & Hourly Panel Aggregation:** Assigns UK Department for Transport (DfT) punctuality bands and aggregates metrics by `(timestamp_hour, day, hour, dow, period, operator, line)`.
    8. **Weather Enrichment & Parquet Caching:** Merges Cardiff hourly weather and saves the master dataset to `results/hourly_bus_delay_panel.parquet`.

    ### 💡 Why We Do It This Way
    - **Streaming Over Monolithic Ingestion:** Storing a month of 1-minute snapshots in RAM would consume >40 GB and crash the system. Streaming file-by-file keeps RAM below 500 MB.
    - **Why 20% Uniform Sampling:** Bus GPS pings are recorded every 60 seconds. In rural routes, buses take several minutes between stops, so consecutive pings have high autocorrelation. A 20% uniform sample preserves the exact same delay distribution at 5x faster speed.
    - **Why Europe/London (BST = UTC+1):** GTFS timetables are published in local civil time. Ignoring daylight saving would shift all delay calculations by exactly 60 minutes.
    - **Why Nearest-Stop Matching Instead of Origin Departure:** Measuring delay against origin departure time computes *cumulative journey elapsed time*, falsely making on-time buses appear 30–45 minutes late mid-route. Comparing against the *nearest stop's scheduled arrival/departure time* measures **true schedule adherence**.
    - **Why $\le 200\text{ m}$ Filter:** Vehicles traveling in open countryside between stops have no timetable reference. Restricting analysis to $\le 200\text{ m}$ of a scheduled stop isolates authentic dwell and arrival events.
    - **Why Midnight Rollover Unwrapping:** The formula `(delay + 720) % 1440 - 720` maps clock minute differences across midnight continuously onto the standard interval $(-720, +720]$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Data Ingestion Ready: Local daily archives in `data/siri_zips` will be processed into `results/hourly_bus_delay_panel.parquet`.*
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
def _():
    # with mo.persistent_cache(name="all_days", save_path="results"):
    #     hourly_df = load_or_process_hourly_data(
    #         sample_frac=0.30)
    return


@app.cell
def _(hourly_df):
    hourly_df.groupby("hour").agg(
        count=("median_delay", "count"),
        median_delay=("median_delay", "median"),
    ).sort_values(by = "count")
    return


@app.cell
def _(hourly_df):
    hourly_dff = hourly_df[~hourly_df["hour"].isin([0, 1, 2, 3, 4])]
    return (hourly_dff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution of delay
    """)
    return


@app.cell
def _(hourly_dff, plt):
    # --- TRIMMED : mean_delay (1st - 99th percentile) ---
    _s = hourly_dff["mean_delay"].dropna()
    _q01, _q99 = _s.quantile([0.01, 0.99]).values
    _s_zoom = _s[(_s >= _q01) & (_s <= _q99)]

    _med = _s_zoom.median()
    _mean = _s_zoom.mean()

    _fig, _ax = plt.subplots(figsize=(8, 4.5), dpi=120, facecolor="#fafafa")
    _ax.set_facecolor("#ffffff")

    _ax.hist(_s_zoom, bins=60, color="#38bdf8", edgecolor="#0284c7", alpha=0.75, density=True)
    _ax.axvline(_med, color="#0f172a", lw=2, label=f"Median ({_med:.1f}m)")
    _ax.axvline(_mean, color="#ef4444", linestyle="--", lw=1.8, label=f"Mean ({_mean:.1f}m)")
    return


@app.cell
def _(hourly_dff):
    q_low, q_high = hourly_dff["mean_delay"].quantile([0.01, 0.99])
    hourly_dfilter = hourly_dff[hourly_dff["mean_delay"].between(q_low, q_high)].copy()
    return (hourly_dfilter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hourly pattern
    """)
    return


@app.cell
def _(hourly_dfilter, mo, plt):
    # 1. Compute median delay for every single calendar day by hour
    _daily_hourly = (
        hourly_dfilter
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

    mo.vstack([
        mo.md("### Diurnal Delay Distribution Across Operating Hours"),
        _clean_plot,
    ])
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

    mo.vstack([
        mo.md("## Day-of-Week Delay Profile (Monday – Sunday)"),
        _dow_plot,
    ])
    return


@app.cell
def _(hourly_df, hourly_dfilter, mo, plt):
    # 1. Filter out extreme outliers for visualization clarity
    _months = sorted(hourly_dfilter["month"].dropna().unique())
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
        tick_labels=_months,
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
        _vals = hourly_df[hourly_df["month"] == _m]["median_delay"].dropna()
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

    mo.vstack([
        mo.md("### Monthly Bus Delay Distributions"),
        _box_plot,
    ])
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

    mo.vstack([
        mo.md("""
        ---
        ## Temporal Congestion Matrix: Day of Week × Operating Hour
        """),
        _heat_chart,
    ])
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

    mo.vstack([
        mo.md("""
        ---
        ## Bus Operator Performance

        """),
        _op_chart,
        # mo.ui.table(op_df),
    ])
    return


@app.cell
def _():
    # # --- ROUTE CORRIDOR RELIABILITY ANALYSIS: TOP 10 DELAYED LINES (DERIVED FROM hourly_df) ---
    # # WHAT: Group hourly_df by (line, operator_name) to rank the 10 most delayed bus routes in rural Wales.
    # # WHY: Identifies priority corridors for local authority highway interventions, timetable padding, or bus priority lanes.
    # route_df = (
    #     hourly_dfilter.groupby(["line", "operator_name"], observed=True)
    #     .agg(
    #         observations=("n_obs", "sum"),
    #         median_delay=("median_delay", "median"),
    #         mean_delay=("mean_delay", "mean"),
    #         iqr_delay=("iqr_delay", "median"),
    #         on_time_pct=("on_time_pct", "mean"),
    #         very_late_pct=("very_late_pct", "mean"),
    #     )
    #     .reset_index()
    # )
    # # Filter to corridors with at least 100 observations to avoid statistical noise
    # top_delayed = route_df[route_df["observations"] >= 100].sort_values("median_delay", ascending=False).head(10).round(2)

    # mo.vstack([
    #     mo.md("""
    #     ---
    #     ## 12. Route Corridor Reliability Analysis: Top 10 Delayed Lines (Derived from `hourly_df`)

    #     ### 🎯 What We Are Doing
    #     We aggregate `hourly_df` by route line and operator to identify the 10 most delayed bus corridors in rural Wales.

    #     ### 💡 Why We Do It This Way
    #     Pinpointing chronic low-performing routes provides concrete operational evidence for local transport authorities to prioritize highway interventions, bus priority lanes, or timetable re-timings.
    #     """),
    #     mo.ui.table(top_delayed),
    # ])
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

        _view = mo.vstack([
            mo.md(r"""
            ---
            ## Weather & Climate Impact Analysis: Temperature & Atmospheric Sensitivity
            """),
            _w_chart,
        ])
    else:
        _view = mo.md("*Weather data integration not available.*")

    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    <!-- ---
    ## 14. Strategic Policy Recommendations for Cymru Wledig LPIP

    Based on the empirical findings across the multi-day hourly panel and ambient weather integration:

    #### 1. Enforce Hold-Back Protocols Against Early Departures
    - **The Issue:** A substantial portion of observations record buses running ahead of timetable ($< -1\text{ min}$), particularly on Sunday mornings and off-peak rural stretches.
    - **Policy Action:** Electronic ticket machines (ETMs) and telematics should actively chime to prevent drivers from departing rural timing points ahead of the published timetable.

    #### 2. Re-Calibrate Timetables for Friday Afternoon Congestion
    - **The Issue:** The Day-of-Week $\times$ Hour heatmap confirms a systemic Friday afternoon bottleneck (15:00–18:00) where delays and variability spike.
    - **Policy Action:** Separate Friday afternoon run-times from standard Monday–Thursday schedules, embedding 6–10 minutes of running time and terminal recovery margin on trunk corridors.

    #### 3. Shift Focus from Average Delay to Unpredictability (IQR)
    - **The Issue:** Operators with similar median delays exhibit dramatically different Inter-Quartile Ranges (IQR). A route with wide IQR imposes severe passenger stress.
    - **Policy Action:** Contracts awarded under the *Bus Services (Wales) Act* franchising regime should adopt Reliability Buffer metrics (80th/90th percentile excess wait time) rather than simple averages.

    #### 4. Weather & Summer Tourism Resilience Scheduling
    - **The Issue:** Punctuality degrades during warm summer periods (>20°C), with "Very Late" service breakdowns rising due to coastal tourist road volume and narrow-lane friction.
    - **Policy Action:** Introduce seasonal "Summer Peak" timetable variants for scenic and coastal corridors (e.g. Cambrian Coast, Pembrokeshire Coast, Brecon Beacons) with widened running times. -->
    """)
    return


@app.cell
def _(hourly_dfilter):
    model_df = hourly_dfilter.drop(columns=["timestamp_hour", "day", "period", "n_obs", "median_delay", "p25_delay", "p75_delay", "on_time_pct", "early_pct", "late_pct", "very_late_pct", "stop_index", "route_progress_pct", "max_planned_stops", "dest_dist_cardiff_km", "iqr_delay", "month", "daytype", "operator_name"])
    dum_vars = ["hour", "dow", "operator", "line", "condition"]
    return dum_vars, model_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of data we use in modeling
    """)
    return


@app.cell
def _(mo, model_df):
    mo.ui.table(model_df.round(3))
    return


@app.cell
def _(dum_vars, model_df, pd):
    ## 1. Create dummy variables for categorical features
    model_dummy =  pd.get_dummies(model_df, columns = dum_vars, drop_first=True, dtype=int, prefix_sep="_")
    return (model_dummy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Regression Results
    """)
    return


@app.cell
def _(model_dummy):
    y = model_dummy["mean_delay"]
    X = model_dummy.drop(columns=["mean_delay"])
    from statsmodels.api import OLS, add_constant
    X = add_constant(X)
    # Cell: Estimate OLS Model
    # Fit OLS with robust standard errors (HC1 / White-Huber)
    ols_model = OLS(y, X)
    ols_results = ols_model.fit()

    # # Display statutory regression metrics in marimo
    # mo.md(rf"""
    # ### OLS Regression Model Results
    # - **Dependent Variable ($$y$$):** `mean_delay` (Minutes)
    # - **Observations ($$N$$):** {int(ols_results.nobs):,}
    # - **R-squared ($$R^2$$):** {ols_results.rsquared:.4f}
    # - **Adjusted $$R^2$$:** {ols_results.rsquared_adj:.4f}
    # - **F-statistic:** {ols_results.fvalue:.2f} ($$p < 0.001$$)
    # """)
    return (ols_results,)


@app.cell
def _(mo, ols_results):
    _summary_card = mo.md(rf"""
    ### OLS Regression Model Results
    - **Dependent Variable ($$y$$):** `mean_delay` (Minutes)
    - **Observations ($$N$$):** {int(ols_results.nobs):,}
    - **R-squared ($$R^2$$):** {ols_results.rsquared:.4f}
    - **Adjusted $$R^2$$:** {ols_results.rsquared_adj:.4f}
    - **F-statistic:** {ols_results.fvalue:.2f} ($$p < 0.001$$)
    """)

    mo.vstack([
        # _summary_card,
        mo.Html(ols_results.summary().as_html()),
    ])
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Causal ML Nuisance Model Tuning (Optuna)

    Unified hyperparameter optimization for the DoubleML nuisance functions ($E[Y \mid X]$ and $E[D \mid X]$) using 5-fold cross-validation, `MedianPruner`, and disk-cached JSON results in `param_results/`.
    """)
    return


@app.cell
def _(data_url, load_json, np, pd, save_local):
    from typing import Any, Dict, Tuple
    import json as _json
    import optuna
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
    ### XGBoost Nuisance Tuning
    Tunes all 6 XGBoost nuisance models (outcome and treatment models for `circuity_ratio`, `avg_stop_spacing_km`, and `total_dist_km`) using `tune_nuisance(model="xgboost", ...)`.
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
    ### LightGBM Nuisance Tuning
    Tunes all 6 LightGBM nuisance models (outcome and treatment models for `circuity_ratio`, `avg_stop_spacing_km`, and `total_dist_km`) using `tune_nuisance(model="lightgbm", ...)`.
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

    target_selector = mo.ui.dropdown(
        options=["All Targets"] + list(accuracy_comparison_df["Target Variable"]),
        value="All Targets",
        label="🎯 Select Target Variable:",
    )

    # Tabbed interface for in-cell interactive browsing
    _tabs_dict = {
        "All Targets": mo.ui.table(accuracy_comparison_df, selection=None),
    }
    for _idx, _row in accuracy_comparison_df.iterrows():
        _tabs_dict[_row["Target Variable"]] = mo.ui.table(pd.DataFrame([_row]), selection=None)

    interactive_accuracy_tabs = mo.ui.tabs(_tabs_dict)

    mo.vstack([
        mo.md("### 📊 Test Data Accuracy Comparison: XGBoost vs LightGBM (RMSE & R2)"),
        interactive_accuracy_tabs,
    ])
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
    import doubleml as dml

    # 1. Define clean modeling data (dropna to ensure complete cases for DML)

    # 2. Package into DoubleMLData
    # - y_col: outcome (e.g. 'mean_delay')
    # - d_cols: treatment variable (can be continuous or binary, e.g. 'precipitation_mm')
    # - x_cols: confounders/controls (all remaining numerical and dummy columns)
    def perform_dml(out_var, treatment_var, _ml_l, _ml_m, data_df=None):
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
    <!-- ## Double Machine Learning (DML) Mathematical Formulation
    ### Partially Linear Regression (PLR) Framework

    To identify the true causal effect of network geometry on delay without suffering from omitted-variable bias or the curse of dimensionality, we employ the **Chernozhukov et al. (2018)** Double/Debiased Machine Learning framework.

    ---

    ### 1. Structural Equation Model

    We specify a Partially Linear Regression (PLR) model:

    $$Y = \theta D + g(X) + U, \quad \mathbb{E}[U \mid D, X] = 0$$

    $$D = m(X) + V, \quad \mathbb{E}[V \mid X] = 0$$

    * **$$Y$$ (Outcome):** Hourly corridor `mean_delay`
    * **$$D$$ (Treatment):** Geometric feature (e.g. `circuity_ratio`, `avg_stop_spacing_km`, `total_dist_km`)
    * **$$\theta$$ (Target Parameter):** Causal marginal impact of treatment on delay
    * **$$X$$ (High-Dimensional Controls):** Weather, temporal fixed effects (hour, day of week), operator, and line indicators
    * **$$g(X), m(X)$$ (Nuisance Functions):** Flexible, non-linear conditional expectations learned via XGBoost -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Causal Machine Learning Results: Causal impact of circuity, avg_stop_spacing and total_dist_km on mean delay
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
    dml_summary = pd.concat(
        {
            "xgb_circuity": dml_xgb_circuity,
            "xgb_stop": dml_xgb_stop,
            "xgb_dist": dml_xgb_dist,
            "lgb_circuity": dml_lgb_circuity,
            "lgb_stop": dml_lgb_stop,
            "lgb_dist": dml_lgb_dist,
        },
        axis=0,
    ).round(3)
    dml_summary
    return


if __name__ == "__main__":
    app.run()
