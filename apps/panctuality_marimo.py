# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#     "altair==6.3.0",
#     "doubleml>=0.11.4",
#     "duckdb==2.0.0",
#     "lightgbm>=4.6.0",
#     "marimo>=0.23.3",
#     "marimo-lens>=0.2.3",
#     "matplotlib>=3.11.2",
#     "mcp>=2.0.0",
#     "nbformat==5.11.1",
#     "numpy>=2.5.4",
#     "openai==3.45.0",
#     "optuna>=5.0.0",
#     "pandas>=3.1.0",
#     "polars[pyarrow]==2.1.0",
#     "pyarrow>=14.0.0",
#     "pytest==9.2.0",
#     "python-lsp-ruff==2.3.4",
#     "python-lsp-server==1.15.0",
#     "scikit-learn>=1.9.1",
#     "sqlglot==30.29.0",
#     "statsmodels>=0.15.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "websockets==17.4",
#     "xgboost>=3.4.1"
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
    BASE_DIR = Path("data/siri_zips")
    PROJECT_DIR = Path.cwd()
    GTFS_DIR = PROJECT_DIR / "data/20260520_bus_sch"
    OUT_DIR = PROJECT_DIR / "outputs_multiday"
    RESULTS_DIR = PROJECT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_CSV = "data/cardiff_weather_april_to_september_2026.csv"
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
    return (
        DOW_ORDER,
        GTFS_DIR,
        Path,
        RESULTS_DIR,
        WEATHER_CSV,
        json,
        np,
        pd,
        plt,
        sys,
    )


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

    return


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


@app.cell
def _(Path, np, pd):
    def load_and_index_gtfs(gtfs_dir: Path):
        """
        Loads Welsh GTFS Timetable Feeds and Pre-Indexes Spatial/Temporal Lookups:
        ========================================================================
        WHAT:
          1. Reads agency.txt, routes.txt, trips.txt, stops.txt, stop_times.txt.
          2. Derives scheduled origin departure minute (origin_min) from the first stop of every trip.
          3. Builds representative directional (rep) and direction-free fallback (repfb) trip lookup tables.
          4. Pre-compiles stop coordinates (lat, lon) and scheduled timetable minutes into contiguous
             NumPy arrays grouped by trip_id.

        WHY:
          - Telematics pings carry route numbers and terminal departure times but lack GTFS trip_id.
          - Pre-indexing allows instantaneous O(1) trip matching and vectorized C-speed Haversine distance
            calculations across hundreds of thousands of bus pings without pandas iteration overhead.
        """
        # --- STEP 1: LOAD RAW GTFS SPECIFICATION TABLES ---
        # WHAT: Load agency metadata containing operator codes and commercial trading names.
        # WHY: AVL feeds broadcast operator codes (NOCs) like 'FCYM' or 'SSWL'. Linking agency_id to agency_noc
        # is required to match live vehicles to scheduled operators.
        agency = pd.read_csv(gtfs_dir / "agency.txt", dtype=str)

        # WHAT: Load route definitions linking internal route_id to public line numbers (e.g., "132", "T4").
        # WHY: String dtype is enforced so alphanumeric line designations (e.g., "X4", "1A", "T1C") are not altered.
        routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str)

        # WHAT: Load trip schedules linking route_id to individual trip_id and direction_id (0=outbound, 1=inbound).
        # WHY: Each route has dozens of discrete daily trips. Preserving direction_id resolves outbound vs inbound runs.
        trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str)

        # WHAT: Load official bus stop coordinates (WGS84 latitude and longitude) across Wales.
        # WHY: Floating-point precision (float64) is essential for computing accurate Haversine spatial distances
        # to identify whether a bus is physically at a stop (within 200m).
        stops = pd.read_csv(
            gtfs_dir / "stops.txt",
            dtype={"stop_id": "str", "stop_lat": "float64", "stop_lon": "float64"},
        )

        # WHAT: Load scheduled arrival and departure times at each sequential stop along every trip.
        # WHY: String types are enforced for IDs to prevent numeric auto-casting from stripping leading zeroes
        # from NaPTAN stop identifiers (e.g., '0610CH1312').
        st = pd.read_csv(
            gtfs_dir / "stop_times.txt",
            usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
            dtype={"trip_id": "str", "arrival_time": "str", "departure_time": "str", "stop_id": "str", "stop_sequence": "int32"},
        )

        # --- STEP 2: TIMETABLE MINUTE-OF-DAY CONVERSION HELPER ---
        # WHAT: Convert 'HH:MM:SS' time strings into continuous minutes from midnight [0, 1440).
        # WHY: GTFS supports times beyond 24:00:00 (e.g. 24:30:00) for overnight trips belonging to the
        # previous service day. Using ((hours * 60 + minutes) % 1440) maps overnight trips continuously
        # into the standard 24-hour clock cycle.
        def _to_min(s):
            p = s.str.split(":", expand=True).astype("Int64")
            return ((p[0] * 60 + p[1]) % 1440).astype("float64")

        # --- STEP 3: EXTRACT ORIGIN DEPARTURE MINUTE PER TRIP ---
        # WHAT: Sort stop_times by (trip_id, stop_sequence) and extract the first scheduled stop (origin).
        # WHY: Live SIRI-VM vehicle tracking feeds broadcast 'OriginAimedDepartureTime' (when the bus was
        # scheduled to depart its first terminus). Finding this initial departure minute in GTFS creates
        # the critical foreign key needed to match live buses to specific timetable trips.
        fs = st.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id", as_index=False).first()
        fs["origin_min"] = _to_min(fs["departure_time"]).round().astype("float64")

        # --- STEP 4: CONSTRUCT COMPOSITE TRIP MATCHING KEYS ---
        # WHAT: Merge trips with route names, agency NOC, and origin departure minute.
        # WHY: Creates a unified reference table (tk) indexed by (agency_noc, route_short_name, origin_min, direction_id).
        tk = (
            trips.merge(routes[["route_id", "route_short_name", "agency_id"]], on="route_id")
            .merge(agency[["agency_id", "agency_noc", "agency_name"]], on="agency_id")
            .merge(fs[["trip_id", "origin_min"]], on="trip_id")
        )
        tk["agency_noc"] = tk["agency_noc"].astype(str)
        tk["route_short_name"] = tk["route_short_name"].astype(str)
        tk["direction_id"] = tk["direction_id"].fillna("0").astype(str)

        # --- STEP 5: BUILD PRIMARY DIRECTIONAL & FALLBACK TRIP LOOKUP TABLES ---
        # WHAT:
        #   - 'rep': Primary lookup matching on (operator, line, origin_min, direction_id).
        #   - 'repfb': Secondary fallback lookup matching on (operator, line, origin_min).
        # WHY: Live telematics often omit direction or broadcast unstandardized labels (e.g., 'inbound' vs '1').
        # The fallback table guarantees high matching coverage even when direction metadata is absent.
        rep = (
            tk.dropna(subset=["origin_min"])
            .drop_duplicates(["agency_noc", "route_short_name", "origin_min", "direction_id"])
            [["agency_noc", "route_short_name", "origin_min", "direction_id", "trip_id", "agency_name"]]
        )
        repfb = (
            tk.dropna(subset=["origin_min"])
            .drop_duplicates(["agency_noc", "route_short_name", "origin_min"])
            [["agency_noc", "route_short_name", "origin_min", "trip_id", "agency_name"]]
            .rename(columns={"trip_id": "trip_id_fb", "agency_name": "agency_name_fb"})
        )

        # --- STEP 6: PRE-COMPILE STOP COORDINATES & SCHEDULED TIMES PER TRIP ---
        # WHAT: Merge physical stop coordinates (lat, lon) into stop_times and pre-compile into contiguous
        # NumPy float64 arrays stored in a Python dictionary: trip_arrays[trip_id] = (slat, slon, smin).
        # WHY: Performing spatial joins or DataFrame slicing for millions of GPS pings in pandas is bottlenecked
        # by Python interpreter overhead. Pre-compiling stop coordinates into contiguous C-order NumPy arrays
        # enables instantaneous broadcasted Haversine calculations (slat[None, :], slon[None, :]) in C-speed.
        stp = st.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
        stp["sched_min"] = _to_min(stp["arrival_time"].fillna(stp["departure_time"]))
        stp = stp.dropna(subset=["stop_lat", "stop_lon", "sched_min"])

        trip_arrays = {} # to save trip_id -> (stop_lat, stop_lon, sched_min) arrays
        for tid, g in stp.groupby("trip_id"):
            trip_arrays[tid] = (
                g["stop_lat"].to_numpy(dtype=np.float64),
                g["stop_lon"].to_numpy(dtype=np.float64),
                g["sched_min"].to_numpy(dtype=np.float64),
            )

        return rep, repfb, trip_arrays

    return (load_and_index_gtfs,)


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


@app.cell
def _(Path, pd):
    def load_cardiff_weather(weather_csv: Path) -> pd.DataFrame:
        """
        Loads Cardiff Hourly Weather CSV and Indexes by Hourly Timestamp:
        ================================================================
        WHAT:
          1. Reads hourly weather observations from Cardiff meteorological station.
          2. Parses timestamps into timezone-naive datetime floored to the operating hour.
          3. Extracts key weather parameters: temperature (°C), rainfall (mm), humidity, wind, condition.
          4. Drops duplicate hourly rows to ensure a strict 1:1 join key.

        WHY:
          - Bus punctuality in rural Wales is heavily influenced by adverse weather (rain, high winds, mist).
          - Flooring timestamps to the hour allows a clean 1-to-1 merge onto the hourly bus delay panel.
        """
        # Guard check: if weather file does not exist, return an empty schema-compliant DataFrame
        if not (weather_csv and Path(weather_csv).exists()):
            return pd.DataFrame(columns=["timestamp_hour", "temperature_c", "precipitation_mm", "condition"])

        # Read raw weather observations from Cardiff station
        w_df = pd.read_csv(weather_csv)

        # WHAT: Parse timestamps into pandas datetime floored to the operating hour.
        # WHY: Ensures timestamp format matches timestamp_hour in the bus delay table for a clean 1:1 merge.
        w_df["timestamp_hour"] = pd.to_datetime(w_df["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

        # WHAT: Select key meteorological columns and enforce uniqueness on timestamp_hour.
        # WHY: Drops duplicate sensor readings so the subsequent left join on the bus panel never duplicates rows.
        cols = [c for c in ["timestamp_hour", "temperature_c", "precipitation_mm", "relative_humidity_pct", "wind_speed_kmh", "condition"] if c in w_df.columns]
        return w_df[cols].drop_duplicates("timestamp_hour")

    return (load_cardiff_weather,)


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


@app.cell
def _(
    GTFS_DIR,
    Path,
    RESULTS_DIR,
    WEATHER_CSV,
    load_and_index_gtfs,
    load_cardiff_weather,
    np,
    pd,
):
    def load_or_process_hourly_data(
        sample_frac: float = 0.20,
        at_stop_radius: float = 200.0,
        gtfs_dir: Path = GTFS_DIR,
        weather_csv: Path = WEATHER_CSV,
        cache_file: Path = RESULTS_DIR / "hourly_bus_delay_panel.parquet",
        data_dir: Path = Path("data/siri_zips"),
        sample_n_csv: int = None,
        max_days: int = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Unified Ingestion & Stop-Level Delay Processing Pipeline:
        ========================================================
        WHAT:
          - Automatically discovers daily SIRI-VM archives in `data/siri_zips` (e.g. 20260401.zip).
          - Directly streams each CSV snapshot from the ZIP archive into RAM without unzipping to disk.
          - Outer loop: iterates over daily archives (chronologically sorted).
          - Inner loop: iterates over CSV snapshots:
            * Samples 20% of pings: csv_df.sample(frac=sample_frac, random_state=42)
            * Standardizes columns (snake_case, CamelCase, SIRI raw tags)
            * Normalizes operator NOCs and aliases (e.g. A2BV -> A2BC, ANWE -> ANSL)
            * Validates spatial coordinates within the Wales bounding box
            * Converts UTC timestamps to local British Summer Time (BST = UTC+1)
            * Matches pings to GTFS scheduled trips (directional with fallback)
            * Computes vectorized Haversine distance from bus GPS to all trip stops
            * Identifies nearest scheduled stop and computes true delay: (rec_min - smin[j] + 720) % 1440 - 720
            * Filters strictly to at-stop pings (distance <= 200m)
            * Computes route geometry features (total_dist_km, max_reach_km, circuity_ratio, etc.)
            * Appends verified at-stop pings to day_at_stop_pings
          - Groups day_at_stop_pings by (timestamp_hour, day, hour, dow, period, operator, operator_name, line)
          - Assigns DfT statutory punctuality bands (Early, On Time, Late, Very Late)
          - Merges Cardiff hourly weather observations on timestamp_hour
          - Collects daily summaries into master list and materializes to Parquet cache

        WHY:
          - Streaming directly from compressed ZIP archives eliminates tens of gigabytes of disk usage.
          - Removing raw latitude/longitude from hourly aggregation produces authentic route-hour statistics.
          - Measuring delay against the nearest scheduled stop prevents the artificial 30-45 min delay bias.
        """
        import zipfile
        import random
        random.seed(42)

        # --- STEP 2: LOAD AND PRE-INDEX REFERENCE DATASETS (GTFS & WEATHER) ---
        if verbose:
            print("Loading and indexing GTFS timetable...")
        rep, repfb, trip_arrays = load_and_index_gtfs(gtfs_dir)
        weather_df = load_cardiff_weather(weather_csv)

        # Operational time period classification helper (standard UK transport planning periods)
        def period_fn(h):
            if 7 <= h <= 9: return "AM peak"
            if 10 <= h <= 15: return "Interpeak"
            if 16 <= h <= 18: return "PM peak"
            if 19 <= h <= 21: return "Evening"
            return "Night"

        # Great-circle Haversine spherical distance formula in meters (Earth radius R = 6,371,000 m)
        R = 6371000.0
        CARDIFF_LAT = 51.4780 # Cardiff city center latitude
        CARDIFF_LON = -3.1780 # Cardiff city center longitude
        def hav(la1, lo1, la2, lo2):
            p1 = np.radians(la1); p2 = np.radians(la2)
            dp = np.radians(la2 - la1); dl = np.radians(lo2 - lo1)
            a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
            return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        # Operator alias mapping: maps live feed operator codes to GTFS National Operator Codes (NOCs)
        op_alias = {"A2BV": "A2BC", "ANWE": "ANSL"}

        # Column mapping across CamelCase (SIRI raw) and snake_case formats
        col_map = {
            "line_ref": "line", "LineRef": "line", "MonitoredVehicleJourney_LineRef": "line",
            "vehicle_ref": "vehicle", "VehicleRef": "vehicle", "MonitoredVehicleJourney_VehicleRef": "vehicle",
            "operator_ref": "operator", "OperatorRef": "operator", "MonitoredVehicleJourney_OperatorRef": "operator",
            "recorded_at": "recorded_at", "RecordedAtTime": "recorded_at", "recorded_at_dt": "recorded_at",
            "origin_aimed_departure": "aimed_dep", "OriginAimedDeparture": "aimed_dep",
            "origin_aimed_departure_time": "aimed_dep", "OriginAimedDepartureTime": "aimed_dep",
            "MonitoredVehicleJourney_OriginAimedDepartureTime": "aimed_dep",
            "latitude": "latitude", "Latitude": "latitude", "MonitoredVehicleJourney_VehicleLocation_Latitude": "latitude",
            "longitude": "longitude", "Longitude": "longitude", "MonitoredVehicleJourney_VehicleLocation_Longitude": "longitude",
            "direction_ref": "direction_ref", "DirectionRef": "direction_ref", "MonitoredVehicleJourney_DirectionRef": "direction_ref",
        }

        # --- STEP 3: DISCOVER DAILY ZIP ARCHIVES ---
        data_dir = Path(data_dir or "data/siri_zips")
        day_sources = sorted([f for f in data_dir.glob("*.zip") if f.name.replace(".zip", "").isdigit()])
        print(f"Discovered {len(day_sources)} daily ZIP archives in {data_dir}")

        if day_sources:
            print(f"First day: {day_sources[0].stem}, Last day: {day_sources[-1].stem}")
        if max_days is not None:
            day_sources = day_sources[:max_days]

        all_days_hourly_list = []

        # Helper generator to stream sampled DataFrames directly from ZIP archive
        def iter_day_csvs(src):
            with zipfile.ZipFile(src, "r") as z:
                csv_names = sorted([n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")])
                if sample_n_csv is not None:
                    csv_names = random.sample(csv_names, min(len(csv_names), sample_n_csv))
                print(f"Archive: {src.name}, CSV files discovered: {len(csv_names)}")
                for idx, name in enumerate(csv_names):
                    if idx % 800 == 0 and verbose:
                        print(f"Archive: {src.name}, {idx + 1}/{len(csv_names)}: Processing CSV: {name}")
                    try:
                        with z.open(name) as f:
                            yield pd.read_csv(f).sample(frac=sample_frac, random_state=42)
                    except Exception:
                        continue

        # --- STEP 4: PROCESSING LOOP ACROSS DAILY ARCHIVES ---
        for _src in day_sources:
            day_label = _src.stem
            day_at_stop_pings = []

            for csv_df in iter_day_csvs(_src): # stream sampled CSVs directly from ZIP to RAM without unzipping to disk. This is done to avoid excessive disk usage and I/O overhead.
                try:
                    csv_df = csv_df.drop_duplicates()
                    if csv_df.empty:
                        continue

                    # --- STEP 4a: STANDARDIZE COLUMN NAMES ---
                    for old, new in col_map.items():
                        if old in csv_df.columns and new not in csv_df.columns:
                            csv_df[new] = csv_df[old]

                    # --- STEP 4b: EXTRACT & NORMALIZE OPERATOR NOC ---
                    if "operator" not in csv_df.columns and "vehicle" in csv_df.columns:
                        csv_df["operator"] = csv_df["vehicle"].astype(str).str.split("-").str[0]
                    elif "operator" not in csv_df.columns:
                        csv_df["operator"] = "UNKNOWN"

                    csv_df["operator"] = csv_df["operator"].astype(str).map(op_alias).fillna(csv_df["operator"].astype(str))
                    csv_df["line"] = csv_df["line"].astype(str)

                    # --- STEP 4c: SPATIAL VALIDATION (WELSH BOUNDING BOX) ---
                    csv_df["latitude"] = pd.to_numeric(csv_df["latitude"], errors="coerce")
                    csv_df["longitude"] = pd.to_numeric(csv_df["longitude"], errors="coerce")
                    csv_df = csv_df.dropna(subset=["latitude", "longitude"])
                    csv_df = csv_df[csv_df["latitude"].between(49.8, 60.9) & csv_df["longitude"].between(-8.7, 1.8)]

                    if "recorded_at" not in csv_df.columns or "aimed_dep" not in csv_df.columns or csv_df.empty:
                        continue

                    # --- STEP 4d: TIMEZONE CONVERSION (UTC -> BRITISH SUMMER TIME) ---
                    t_obs = pd.to_datetime(csv_df["recorded_at"], utc=True, errors="coerce").dt.tz_convert("Europe/London")
                    t_aim = pd.to_datetime(csv_df["aimed_dep"], utc=True, errors="coerce").dt.tz_convert("Europe/London")

                    csv_df["dt"] = t_obs
                    csv_df["aim_dt"] = t_aim
                    csv_df = csv_df.dropna(subset=["dt", "aim_dt"])
                    if csv_df.empty:
                        continue

                    # --- STEP 4e: COMPUTE DEPARTURE & OBSERVATION MINUTES OF DAY ---
                    csv_df["origin_min"] = (csv_df["aim_dt"].dt.hour * 60 + csv_df["aim_dt"].dt.minute).round().astype("float64")
                    csv_df["rec_min"] = (csv_df["dt"].dt.hour * 60 + csv_df["dt"].dt.minute + csv_df["dt"].dt.second / 60.0).astype("float64")

                    # --- STEP 4f: MATCH LIVE VEHICLE JOURNEYS TO SCHEDULED GTFS TRIPS ---
                    if "direction_ref" in csv_df.columns:
                        dir_map = {"inbound": "1", "outbound": "0", "1": "1", "0": "0"}
                        csv_df["dir_id"] = csv_df["direction_ref"].astype(str).str.lower().map(dir_map).fillna("0")
                        m = csv_df.merge(
                            rep,
                            left_on=["operator", "line", "origin_min", "dir_id"],
                            right_on=["agency_noc", "route_short_name", "origin_min", "direction_id"],
                            how="left",
                        )
                        m = m.merge(
                            repfb,
                            left_on=["operator", "line", "origin_min"],
                            right_on=["agency_noc", "route_short_name", "origin_min"],
                            how="left",
                        )
                        m["trip_id"] = m["trip_id"].fillna(m["trip_id_fb"])
                        m["agency_name"] = m["agency_name"].fillna(m["agency_name_fb"]).fillna(m["operator"])
                    else:
                        m = csv_df.merge(
                            repfb,
                            left_on=["operator", "line", "origin_min"],
                            right_on=["agency_noc", "route_short_name", "origin_min"],
                            how="left",
                        )
                        m["trip_id"] = m["trip_id_fb"]
                        m["agency_name"] = m["agency_name_fb"].fillna(m["operator"])

                    m = m.dropna(subset=["trip_id"])
                    if m.empty:
                        continue

                    # --- STEP 4g: VECTORIZED NEAREST-SCHEDULED-STOP & DELAY CALCULATION ---
                    for tid, g in m.groupby("trip_id"):
                        if tid not in trip_arrays:
                            continue
                        slat, slon, smin = trip_arrays[tid]
                        plat = g["latitude"].to_numpy()
                        plon = g["longitude"].to_numpy()
                        n_planned_stops = len(slat)

                        # Great-circle Haversine distance matrix between pings and trip stops
                        D = hav(plat[:, None], plon[:, None], slat[None, :], slon[None, :])
                        j = D.argmin(axis=1)
                        dist = D[np.arange(len(g)), j]

                        # Schedule delay unwrapped continuously around midnight
                        delays = (g["rec_min"].to_numpy() - smin[j] + 720) % 1440 - 720

                        # --- STEP 4h: FILTER TO VERIFIED AT-STOP OBSERVATIONS (DIST <= 200M) ---
                        valid = dist <= at_stop_radius
                        if np.any(valid):
                            sub = g.iloc[valid].copy()
                            total_dist = hav(slat[:-1], slon[:-1], slat[1:], slon[1:]).sum() / 1000.0
                            max_reach  = hav(slat[0], slon[0], slat, slon).max() / 1000.0
                            j_valid = j[valid]
                            sched_dur = (smin[-1] - smin[0] + 1440) % 1440

                            sub["delay_min"] = delays[valid]
                            sub["dist_m"] = dist[valid]
                            sub["planned_stops"] = n_planned_stops
                            sub["stop_index"] = j_valid
                            sub["route_progress_pct"] = (j_valid / max(1, n_planned_stops - 1)) * 100.0
                            sub["total_dist_km"] = total_dist
                            sub["max_reach_km"] = max_reach
                            sub["avg_stop_spacing_km"] = total_dist / max(1, n_planned_stops - 1)
                            sub["circuity_ratio"] = total_dist / max(0.1, max_reach)
                            sub["sched_duration_min"] = sched_dur
                            sub["sched_speed_kmh"] = (total_dist / max(0.1, sched_dur / 60.0)) if sched_dur > 0 else np.nan
                            sub["origin_dist_cardiff_km"] = hav(slat[0], slon[0], CARDIFF_LAT, CARDIFF_LON) / 1000.0
                            sub["dest_dist_cardiff_km"]   = hav(slat[-1], slon[-1], CARDIFF_LAT, CARDIFF_LON) / 1000.0
                            day_at_stop_pings.append(sub)
                except Exception:
                    print(f"Error processing CSV in {day_label}, skipping this file.")
                    continue

            if not day_at_stop_pings:
                print(f"Skipping {day_label}: 0 matched at-stop pings.")
                continue

            # --- STEP 4i: CONSOLIDATE VERIFIED AT-STOP PINGS & ASSIGN STATUTORY BANDS ---
            pings_df = pd.concat(day_at_stop_pings, ignore_index=True)
            pings_df["timestamp_hour"] = pings_df["dt"].dt.floor("h").dt.tz_localize(None)
            pings_df["day"] = pings_df["dt"].dt.strftime("%Y-%m-%d")
            pings_df["hour"] = pings_df["dt"].dt.hour
            pings_df["dow"] = pings_df["dt"].dt.day_name()
            pings_df["period"] = pings_df["hour"].map(period_fn)

            # Senior Traffic Commissioner & DfT Statutory Punctuality Brackets (Vectorized 0-100 floats for C-mean)
            pings_df["is_ontime"] = ((pings_df["delay_min"] >= -1.0) & (pings_df["delay_min"] <= 5.99)).astype(float) * 100.0
            pings_df["is_early"] = (pings_df["delay_min"] < -1.0).astype(float) * 100.0
            pings_df["is_late"] = ((pings_df["delay_min"] > 5.99) & (pings_df["delay_min"] <= 15.0)).astype(float) * 100.0
            pings_df["is_very_late"] = (pings_df["delay_min"] > 15.0).astype(float) * 100.0
            pings_df["operator_name"] = pings_df["agency_name"].fillna(pings_df["operator"])

            # --- STEP 4j: AGGREGATE TO HOURLY ROUTE GRANULARITY (WITHOUT LAT/LON) ---
            hourly_day = (
                pings_df.groupby(
                    ["timestamp_hour", "day", "hour", "dow", "period", "operator", "operator_name", "line"],
                    observed=True,
                )
                .agg(
                    n_obs=("delay_min", "count"),
                    median_delay=("delay_min", "median"),
                    mean_delay=("delay_min", "mean"),
                    p25_delay=("delay_min", lambda s: s.quantile(0.25)),
                    p75_delay=("delay_min", lambda s: s.quantile(0.75)),
                    on_time_pct=("is_ontime", "mean"),
                    early_pct=("is_early", "mean"),
                    late_pct=("is_late", "mean"),
                    very_late_pct=("is_very_late", "mean"),
                    planned_stops=("planned_stops", "median"),
                    stop_index=("stop_index", "median"),
                    route_progress_pct=("route_progress_pct", "median"),
                    max_planned_stops=("planned_stops", "max"),
                    total_dist_km=("total_dist_km", "max"),
                    max_reach_km=("max_reach_km", "max"),
                    avg_stop_spacing_km=("avg_stop_spacing_km", "median"),
                    circuity_ratio=("circuity_ratio", "median"),
                    sched_duration_min=("sched_duration_min", "median"),
                    sched_speed_kmh=("sched_speed_kmh", "median"),
                    origin_dist_cardiff_km=("origin_dist_cardiff_km", "min"),
                    dest_dist_cardiff_km=("dest_dist_cardiff_km", "max")
                )
                .reset_index()
            )
            hourly_day["iqr_delay"] = hourly_day["p75_delay"] - hourly_day["p25_delay"]

            # --- STEP 4k: MERGE AMBIENT CARDIFF WEATHER ATTRIBUTES ---
            if not weather_df.empty:
                hourly_day = hourly_day.merge(weather_df, on="timestamp_hour", how="left")

            all_days_hourly_list.append(hourly_day)
            print(f"Successfully processed {day_label}: {len(hourly_day)} hourly route records.")

        # --- STEP 5: CONCATENATE ALL DAILY HOURLY TABLES INTO MASTER HOURLY PANEL ---
        if not all_days_hourly_list:
            print("No hourly records could be generated from available archives.")
            return pd.DataFrame()

        hourly_df = pd.concat(all_days_hourly_list, ignore_index=True)
        ## drop duplicates in case of overlapping archives or repeated CSVs
        hourly_df = hourly_df.drop_duplicates()
        hourly_df["month"] = pd.to_datetime(hourly_df["day"]).dt.strftime("%Y-%m")
        hourly_df["daytype"] = hourly_df["dow"].apply(lambda d: "Weekend" if d in ["Saturday", "Sunday"] else "Weekday")

        # --- STEP 6: MATERIALIZE MASTER PANEL TO PARQUET CACHE ---
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        hourly_df.to_parquet(cache_file, index=False)
        print(f"Saved {len(hourly_df)} hourly records across {hourly_df['day'].nunique()} unique days to {cache_file}.")
        return hourly_df

    return


@app.cell
def _():
    # ## example usag
    # hourly_example = load_or_process_hourly_data(
    #         sample_frac=0.20,
    #         max_days=1,
    #         sample_n_csv=5,
    #     )
    return


@app.cell
def _():
    # hourly_example[["median_delay", "avg_stop_spacing_km", "circuity_ratio", "avg_stop_spacing_km", "max_planned_stops", "origin_dist_cardiff_km", "dest_dist_cardiff_km"]].corr()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Data Ingestion Ready: Local daily archives in `data/siri_zips` will be processed into `results/hourly_bus_delay_panel.parquet`.*
    """)
    return


@app.cell
def _(mo, pd):
    ## read from parquet file
    # Load the panel from the data folder next to the notebook. This works both
    # locally and as a WASM app on GitHub Pages (mo.notebook_location resolves
    # to the notebook's directory / public URL).
    hourly_df = pd.read_parquet(
        mo.notebook_location() / "lpip_data" / "hourly_bus_delay_panel.parquet"
    )
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
def _(np, pd):
    from typing import Any, Dict, Tuple
    from pathlib import Path as _Path
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
        save_dir: str | _Path = "param_results",
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
        `<save_dir>/<cache_name>.json` exists (with `force_retune=False`), loads
        cached parameters directly from disk without re-running Optuna.
        """
        model_key = _normalize_model_key(model)
        spec = _MODEL_SPECS[model_key]
        estimator_cls = spec["estimator"]

        save_path = _Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        cache_file = (save_path / f"{cache_name}.json") if cache_name else None

        if cache_file and cache_file.exists() and not force_retune:
            with open(cache_file, "r") as f:
                saved = _json.load(f)
            print(
                f" Loaded cached {model_key} parameters for '{cache_name}' "
                f"from {cache_file} (Best score: {saved['best_score']:.4f})"
            )
            return saved["best_params"], saved["best_score"]

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
            with open(cache_file, "w") as f:
                _json.dump({"best_params": best_full_params, "best_score": study.best_value}, f, indent=2)
            print(f" Saved tuned {model_key} parameters to {cache_file}")

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
    ml_test_df,
    ml_train_df,
    outcome_var,
    pd,
    tune_nuisance,
):
    # --- PREDICTIVE TUNING: ALL VARIABLES -> mean_delay (XGBoost vs LIGHTGBM) ---
    # WHAT: Tune one XGBoost and one LightGBM regressor predicting `mean_delay` from the
    #       FULL feature set (route geometry, scheduled speed, weather, and all
    #       hour/dow/operator/line/condition dummies). Nothing is excluded, unlike the
    #       DML nuisance models which each drop one treatment variable.
    # WHY: Benchmarks the predictive ceiling of the panel: how well can hourly delay be
    #      predicted when every variable is available as a feature?

    # 1) Tune XGBoost using ALL variables (Optuna 3-fold CV; cached in param_results/tune_xgb_all.json)
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

    # 2) Tune LightGBM using ALL variables (cached in param_results/tune_lgb_all.json)
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
    #    test split. These numbers feed the XGBoost vs LightGBM comparison table (cell
    #    yOPj), where they appear as the "mean_delay (all features)" row.
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
    full_model_performance_df,
    ml_test_df,
    ml_train_df,
    mo,
    outcome_var,
    pd,
    treatment_var1,
    treatment_var2,
    treatment_var3,
):
    from pathlib import Path as _Path
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
        cache_path: str = "results/accuracy_comparison_xgb_vs_lgb_v2.csv",
        rerun: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluates XGBoost and LightGBM nuisance models on the test set using RMSE and R-squared.
        Loads cached results from disk if available to avoid re-fitting.
        """
        cache_file = _Path(cache_path)
        if cache_file.exists() and not rerun:
            print(f" Loaded accuracy comparison from {cache_file}")
            return pd.read_csv(cache_file)

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
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(cache_file, index=False)
        print(f" Saved accuracy comparison to {cache_file}")
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
    Path,
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
    ml_train_df,
    outcome_var,
    pd,
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


    def dml_result(name, out_var, treatment_var, ml_l, ml_m, rerun=False, save_dir="results"):
        """Load a saved DML summary from disk, or rerun and save it.

        Unlike mo.persistent_cache, this only depends on the file on disk:
        small upstream changes do not trigger a rerun. Set rerun=True to refit.
        """
        cache_file = Path(save_dir) / f"{name}.csv"
        if cache_file.exists() and not rerun:
            print(f" Loaded DML result '{name}' from {cache_file}")
            return pd.read_csv(cache_file, index_col=0)
        summary = perform_dml(out_var, treatment_var, ml_l, ml_m)
        summary.to_csv(cache_file)
        print(f" Saved DML result '{name}' to {cache_file}")
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
