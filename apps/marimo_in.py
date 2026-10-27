# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.23.3",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
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
    ## Model Specifications
    """)
    return


@app.cell
def _():
    import io
    import pickle
    from urllib.request import urlopen
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd
    import numpy as np

    def fetch(url):
        with urlopen(url) as r:
            return io.BytesIO(r.read())

    # Load model results directly from GitHub raw repository (or local fallback)
    _DATA_URL = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/sphere/ms_arr_model_results.pkl"
    try:
        model_res = pickle.load(fetch(_DATA_URL))
    except Exception:
        with open("talks/sphere/ms_arr_model_results.pkl", "rb") as _f:
            model_res = pickle.load(_f)

    hmm_df = model_res["hmm_df"]

    # Load dataset
    all_ead = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/sphere/ead_data.csv"
    try:
        ead = pd.read_csv(fetch(all_ead))
    except Exception:
        ead = pd.read_csv("talks/sphere/ead_data.csv")

    ead["date"] = pd.to_datetime(ead["date"])
    ead.set_index("date", inplace=True)
    return ead, fetch, hmm_df, model_res, np, pd, plt


@app.cell
def _(hmm_df, mo):
    _dates = hmm_df.index.unique().sort_values()
    _n_dates = len(_dates)
    _min_d = _dates.min().strftime("%b %d, %Y")
    _max_d = _dates.max().strftime("%b %d, %Y")

    date_range_bar = mo.ui.range_slider(
        start=0,
        stop=_n_dates - 1,
        value=(max(0, _n_dates - 180), _n_dates - 1),
        step=1,
        full_width=True,
        label=f"📅 **Drag bar to select Date Range** *({_min_d} to {_max_d})*:",
    )
    date_range_bar
    return (date_range_bar,)


@app.cell(hide_code=True)
def _(date_range_bar, hmm_df, mo):
    _dates = hmm_df.index.unique().sort_values()
    _i0, _i1 = date_range_bar.value
    _d0 = _dates[min(_i0, len(_dates) - 1)].strftime("%b %d, %Y")
    _d1 = _dates[min(_i1, len(_dates) - 1)].strftime("%b %d, %Y")
    mo.md(
        f"<div style='font-size:22px;font-weight:bold;color:#0B363C'>"
        f"📅 Selected Period: {_d0} <span style='color:#20808D'>→</span> {_d1}</div>"
    )
    return


@app.cell
def _(date_range_bar, hmm_df, mo):
    import altair as alt

    _dates = hmm_df.index.unique().sort_values()
    _n_dates = len(_dates)
    _start_i, _stop_i = date_range_bar.value
    _start_i = max(0, min(_start_i, _n_dates - 1))
    _stop_i = max(0, min(_stop_i, _n_dates - 1))
    _start_d = _dates[_start_i].date()
    _stop_d = _dates[_stop_i].date()
    _df_chart = hmm_df.loc[str(_start_d):str(_stop_d)].copy().reset_index()
    _df_chart["date_str"] = _df_chart["date"].dt.strftime("%b %d, %Y")

    if len(_df_chart) < 2:
        alt_chart_display = mo.md("⚠️ *Select at least 2 days to display chart.*")
    else:
        _hover = alt.selection_point(
            fields=["date"],
            nearest=True,
            on="pointerover",
            empty=False,
            clear="pointerout",
        )

        _x_scale = alt.Scale(
            domain=[_df_chart["date"].min(), _df_chart["date"].max()]
        )
        _x_axis = alt.Axis(
            format="%b %d, %Y",
            labelAngle=-25,
            titleColor="#0B363C",
            labelColor="#0B363C",
            grid=True,
            gridDash=[3, 3],
            gridOpacity=0.35,
            gridColor="#cbd5e1",
        )

        _area_top = (
            alt.Chart(_df_chart)
            .mark_area(opacity=0.08, color="#20808D")
            .encode(
                x=alt.X("date:T", scale=_x_scale, axis=_x_axis, title=None),
                y=alt.Y(
                    "ead:Q", title="Avoidable Deaths", scale=alt.Scale(zero=True)
                ),
            )
        )

        _points_top = (
            alt.Chart(_df_chart)
            .mark_circle(size=50, stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("date:T", scale=_x_scale),
                y=alt.Y("ead:Q"),
                color=alt.Color(
                    "regime_2:Q",
                    scale=alt.Scale(
                        range=["#20808D", "#D97706", "#C93B2B"],
                        domain=[0, 0.5, 1.0],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("date_str:N", title="Date"),
                    alt.Tooltip("ead:Q", title="Avoidable Deaths", format=".2f"),
                    alt.Tooltip("regime_2:Q", title="P(Peak Risk)", format=".3f"),
                    alt.Tooltip("regime_1:Q", title="P(Normal)", format=".3f"),
                ],
            )
        )

        _rule_top = (
            alt.Chart(_df_chart)
            .mark_rule(color="#1e293b", strokeWidth=1.8, strokeDash=[4, 3])
            .encode(x=alt.X("date:T", scale=_x_scale))
            .transform_filter(_hover)
        )

        _chart_top = (
            _area_top + _points_top + _rule_top
        ).properties(
            width=880,
            height=250,
            title=alt.TitleParams(
                text=f"Estimated Avoidable Deaths with Regime Changes ({_start_d.strftime('%b %d, %Y')} – {_stop_d.strftime('%b %d, %Y')})",
                fontSize=16,
                fontWeight="bold",
                color="#0B363C",
            ),
        )

        _area_surge = (
            alt.Chart(_df_chart)
            .mark_area(opacity=0.22, color="#C93B2B")
            .encode(
                x=alt.X("date:T", scale=_x_scale, axis=_x_axis, title="Date"),
                y=alt.Y(
                    "regime_2:Q",
                    title="Posterior Probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
            )
        )

        _line_surge = (
            alt.Chart(_df_chart)
            .mark_line(color="#C93B2B", strokeWidth=2.6)
            .encode(x=alt.X("date:T", scale=_x_scale), y=alt.Y("regime_2:Q"))
        )

        _line_normal = (
            alt.Chart(_df_chart)
            .mark_line(color="#20808D", strokeWidth=2.2, strokeDash=[5, 4])
            .encode(x=alt.X("date:T", scale=_x_scale), y=alt.Y("regime_1:Q"))
        )

        _points_bottom = (
            alt.Chart(_df_chart)
            .mark_circle(size=60, opacity=0)
            .encode(
                x=alt.X("date:T", scale=_x_scale),
                y=alt.Y("regime_2:Q"),
                tooltip=[
                    alt.Tooltip("date_str:N", title="Date"),
                    alt.Tooltip(
                        "regime_2:Q", title="P(Peak Risk | Past)", format=".3f"
                    ),
                    alt.Tooltip(
                        "regime_1:Q", title="P(Normal | Past)", format=".3f"
                    ),
                ],
            )
            .add_params(_hover)
        )

        _rule_bottom = (
            alt.Chart(_df_chart)
            .mark_rule(color="#1e293b", strokeWidth=1.8, strokeDash=[4, 3])
            .encode(x=alt.X("date:T", scale=_x_scale))
            .transform_filter(_hover)
        )

        _chart_bottom = (
            _area_surge + _line_surge + _line_normal + _points_bottom + _rule_bottom
        ).properties(width=880, height=180)

        alt_chart_display = (
            alt.vconcat(_chart_top, _chart_bottom)
            .configure_view(stroke=None)
        )

    _selected_days = len(_df_chart)
    _range_info = mo.md(
        f"**Window:** `{_selected_days} days`"
    )

    mo.vstack([_range_info, alt_chart_display])
    return (alt,)


@app.cell
def _(ead, model_res):
    # === MODEL TRAINING / PERSISTENT CACHE (Commented out; uncomment if you need to rerun and save) ===
    # from peshbeen.model_selection import SplitTimeSeries
    # from peshbeen.models import ms_arr
    # from peshbeen.transformations import rolling_quantile, fourier_terms
    # from peshbeen.metrics import MSE
    # from sklearn.preprocessing import OneHotEncoder

    # val_len = 65
    # train_ead = ead[:-val_len]

    # tsv_opt = SplitTimeSeries(n_splits=90, test_size=10, step_size=1)
    # ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    # transformation_function = [rolling_quantile(window_size=49, quantile=0.2, shift=10, min_samples=49)]
    # cat_vars = ["day", "month", "holiday"]

    # def run_msar():
    #     model_specs = {"trans_matrix": [], "std": []}
    #     for idx, (train_index, test_index) in enumerate(tsv_opt.split(train_ead)):
    #         if idx == 0:
    #             ms_mod = ms_arr(n_components=2, lags=[4, 8, 10], target_col="ead",
    #                             cat_variables=cat_vars, categorical_encoder=ohe_encoder, n_iter=100,
    #                             random_state=42, lag_transform=transformation_function)
    #         train = train_ead.iloc[train_index]
    #         test = train_ead.iloc[test_index]
    #         ms_mod.fit(train)
    #         if idx % 10 == 0:
    #             print(f"Iteration: {idx} is done")
    #         model_specs["trans_matrix"].append(ms_mod.A)
    #         model_specs["std"].append(ms_mod.stds)
    #     return model_specs

    # with mo.persistent_cache(name="ms_res", save_path="marimo_res"):
    #     res = run_msar()

    # === LOAD PRE-COMPUTED MODEL RESULTS DIRECTLY FROM GITHUB ===
    import io as _io
    import pickle as _pickle
    import urllib.request as _urllib_req

    _MS_CACHE_URL = "https://raw.githubusercontent.com/mustafaslanCoto/marimo-outputs/main/apps/marimo_res/ms_res/X_2K4ad7Mx7HLG8VphbAIHCJHm0j4P4BicaMmLPmkr5IU.pickle"
    try:
        with _urllib_req.urlopen(_MS_CACHE_URL) as _resp:
            _cache_obj = _pickle.load(_io.BytesIO(_resp.read()))
            res = _cache_obj.defs["res"]
    except Exception:
        try:
            with open("marimo_res/ms_res/X_2K4ad7Mx7HLG8VphbAIHCJHm0j4P4BicaMmLPmkr5IU.pickle", "rb") as _f:
                _cache_obj = _pickle.load(_f)
                res = _cache_obj.defs["res"]
        except Exception:
            # Fallback to model_res already in memory
            res = model_res["res"]

    val_len = 65
    train_ead = ead[:-val_len]
    tsv_opt = None
    return res, train_ead, tsv_opt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How transition matrix change
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_res, res, train_ead, tsv_opt):
    # Extract transition probabilities and split cutoffs
    if tsv_opt is not None:
        cv_dates = [train_ead.index[train_idx[-1]] for train_idx, _ in tsv_opt.split(train_ead)]
    elif "cv_dates" in model_res:
        cv_dates = model_res["cv_dates"]
    else:
        # 90 splits, test_size=10, step_size=1
        _n_splits = len(res["trans_matrix"])
        _total_eval = 10 + _n_splits - 1
        _start_idx = len(train_ead) - _total_eval
        cv_dates = [train_ead.index[_start_idx + _i] for _i in range(_n_splits)]

    p_stay_regime0 = [A[0, 0] for A in res["trans_matrix"]]
    p_stay_regime1 = [A[1, 1] for A in res["trans_matrix"]]

    cv_idx_slider = mo.ui.slider(
        start=0,
        stop=len(cv_dates) - 1,
        value=len(cv_dates) - 1,
        step=1,
        label="🔍 **Inspect CV Split / Date:**",
        full_width=True
    )
    cv_idx_slider
    return cv_dates, cv_idx_slider, p_stay_regime0, p_stay_regime1


@app.cell(hide_code=True)
def _(cv_dates, cv_idx_slider, np, p_stay_regime0, p_stay_regime1, plt):
    import matplotlib.dates as mdates

    _n_pts = min(len(cv_dates), len(p_stay_regime0), len(p_stay_regime1))
    fig_trans, ax_trans = plt.subplots(figsize=(11, 4.8))

    if _n_pts == 0:
        ax_trans.text(0.5, 0.5, "No CV persistence data available.", ha="center", va="center")
    else:
        _valid_dates = cv_dates[:_n_pts]
        _valid_p00 = p_stay_regime0[:_n_pts]
        _valid_p11 = p_stay_regime1[:_n_pts]

        # Safely clamp selected index to available data range
        _idx = max(0, min(cv_idx_slider.value, _n_pts - 1))
        _sel_date = _valid_dates[_idx]
        _sel_p00 = _valid_p00[_idx]
        _sel_p11 = _valid_p11[_idx]

        ax_trans.plot(
            _valid_dates, _valid_p00,
            label=r"Regime 0 Persistence ($P_{00}$ - Baseline)",
            color="#006699", linewidth=2.2, marker="o", markersize=3.5, alpha=0.9
        )
        ax_trans.plot(
            _valid_dates, _valid_p11,
            label=r"Regime 1 Persistence ($P_{11}$ - Surge)",
            color="#d95f02", linewidth=2.2, marker="s", markersize=3.5, alpha=0.9
        )

        # Reference benchmark / mean lines
        mean_p00 = np.mean(_valid_p00)
        mean_p11 = np.mean(_valid_p11)
        ax_trans.axhline(mean_p00, color="#006699", linestyle="--", alpha=0.45, label=f"Mean $P_{{00}}$ ({mean_p00:.2f})")
        ax_trans.axhline(mean_p11, color="#d95f02", linestyle="--", alpha=0.45, label=f"Mean $P_{{11}}$ ({mean_p11:.2f})")

        # Set Y limits accommodating target range (0.80 to 0.95) and data
        _y_data = _valid_p00 + _valid_p11
        _y_min = min(0.80, min(_y_data) - 0.02)
        _y_max = max(0.95, max(_y_data) + 0.02)
        ax_trans.set_ylim(_y_min, _y_max)

        # Vertical inspection line
        ax_trans.axvline(_sel_date, color="#1e293b", linestyle="-", linewidth=2.0, alpha=0.9, zorder=10)

        # Highlight points at selected index
        ax_trans.scatter([_sel_date], [_sel_p00], color="#006699", s=90, edgecolor="white", linewidth=1.5, zorder=11)
        ax_trans.scatter([_sel_date], [_sel_p11], color="#d95f02", s=90, edgecolor="white", linewidth=1.5, zorder=11)

        # Floating callout text box next to vertical line
        _x_lim = ax_trans.get_xlim()
        _date_num = mdates.date2num(_sel_date)
        _x_span = _x_lim[1] - _x_lim[0]

        # If cursor is near the right edge, place text on the left; otherwise right
        if _x_span > 0 and (_date_num - _x_lim[0]) / _x_span > 0.68:
            _ha = "right"
            _offset = -0.015 * _x_span
        else:
            _ha = "left"
            _offset = 0.015 * _x_span if _x_span > 0 else 0.5

        _text_str = (
            f"Cutoff: {_sel_date.strftime('%b %d, %Y')}\n"
            f"P₀₀ (Baseline): {_sel_p00:.4f}\n"
            f"P₁₁ (Surge):     {_sel_p11:.4f}"
        )

        _y_lim = ax_trans.get_ylim()
        _text_y = _y_lim[1] - 0.03 * (_y_lim[1] - _y_lim[0])

        ax_trans.text(
            _date_num + _offset, _text_y, _text_str,
            ha=_ha, va="top", fontsize=10, fontweight="bold", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.5, alpha=0.95),
            zorder=12
        )

        ax_trans.set_title("Stability of Regime Persistence Across CV Training Windows", fontsize=13, fontweight="bold", pad=12)
        ax_trans.set_xlabel("CV Window End Date (Training Cutoff)", fontsize=11, fontweight="medium")
        ax_trans.set_ylabel("Probability of Staying in Regime ($P_{ii}$)", fontsize=11, fontweight="medium")
        ax_trans.grid(True, linestyle=":", alpha=0.6)
        ax_trans.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=9.5, loc="lower right", ncol=2)

    for spine in ax_trans.spines.values():
        spine.set_color("#cccccc")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Can Calibrated Uncertainty Bounds Capture Peaks in EAD?
    """)
    return


@app.cell(hide_code=True)
def _(fetch, mo, pd):
    # CSV instead of pickle: pickles are pandas-version specific and fail in the WASM (Pyodide) build
    _url = "https://raw.githubusercontent.com/mustafaslanCoto/marimo-outputs/main/apps/public/cv_results.csv"
    try:
        cv_eval_df = pd.read_csv(fetch(_url), parse_dates=["cutoff", "fold_index"])
    except Exception:
        cv_eval_df = pd.read_csv("public/cv_results.csv", parse_dates=["cutoff", "fold_index"])

    bound_radio = mo.ui.radio(
        options={"Upper 90% Bound (Ceiling)": "90", "Upper 80% Bound": "80"},
        value="Upper 90% Bound (Ceiling)",
        inline=True,
    )
    th_slider = mo.ui.slider(
        steps=[0.8, 0.9, 1.0, 1.1, 1.2],
        value=1.0,
        label="Alert Threshold (EAD)",
        show_value=False,
        full_width=True,
    )
    _th_styled = mo.Html(
        f"""
        <style>
        .th-slider {{ width: 800px; max-width: 100%; }}
        </style>
        <div class="th-slider">{th_slider}</div>
        """
    )
    mo.vstack([bound_radio, _th_styled], gap=1)
    return bound_radio, cv_eval_df, th_slider


@app.cell
def _(mo, th_slider):
    mo.Html(
        f'<div style="font-size:2rem;font-weight:700;color:#2563eb">Alert Threshold: {th_slider.value:.1f} EAD</div>'
    )
    return


@app.cell(hide_code=True)
def _(alt, bound_radio, cv_eval_df, mo, pd, th_slider):
    _b = bound_radio.value
    _th = th_slider.value
    _col = f"upper_{_b}"
    _d = cv_eval_df.copy()
    _d["date_str"] = _d["fold_index"].dt.strftime("%b %d, %Y")
    _surge = _d["y_true"] > _th
    _inside = _d["y_true"] <= _d[_col]
    _d["signal"] = "Safe Demand"
    _d.loc[_surge & _inside, "signal"] = f"Correct Surge (≤ Upper {_b}%)"
    _d.loc[_surge & ~_inside, "signal"] = f"Overrun Breach (> Upper {_b}%)"
    _d["bound"] = _d[_col]

    _n = len(_d)
    _n_surges = int(_surge.sum())
    _corr = int((_surge & _inside).sum())
    _wrong = int((_surge & ~_inside).sum())
    _safe = int((~_surge & _inside).sum())
    _pt_miss_mask = (_d["y_pred"] <= _th) & _surge
    _pt_capt = int(((_d["y_pred"] >= _th) & _surge).sum())
    _denom = _corr + _wrong
    _rate = _corr / _denom * 100 if _denom else 100.0
    _capt_pct = _pt_capt / _n_surges * 100 if _n_surges else 0.0

    _x = alt.X(
        "fold_index:T",
        title="Date (Rolling 120-Day Evaluation Horizon | 24 updates × 5 days)",
        axis=alt.Axis(format="%b %d, %Y", labelAngle=-25),
    )
    _y = alt.Y(
        "y_true:Q",
        title="Emergency Avoidable Deaths (EAD)",
        scale=alt.Scale(domain=[0.15, 1.9]),
    )
    _base = alt.Chart(_d).encode(x=_x)

    _actual = _base.mark_line(color="#C93B2B", strokeWidth=3.5).encode(y=_y)
    _pred = _base.mark_line(
        color="#2563EB", strokeWidth=2.8, strokeDash=[6, 4], point=alt.OverlayMarkDef(shape="square", size=60, color="#2563EB")
    ).encode(
        y="y_pred:Q",
        tooltip=[alt.Tooltip("date_str:N", title="Date"), alt.Tooltip("y_pred:Q", title="Point Forecast", format=".2f")],
    )
    _bound = _base.mark_line(
        color="#0B363C" if _b == "90" else "#20808D",
        strokeWidth=2.8,
        strokeDash=[] if _b == "90" else [2, 3],
        point=alt.OverlayMarkDef(shape="triangle-up" if _b == "90" else "triangle-down", size=50),
    ).encode(
        y="bound:Q",
        tooltip=[alt.Tooltip("date_str:N", title="Date"), alt.Tooltip("bound:Q", title=f"Upper {_b}% Bound", format=".2f")],
    )

    _sig_domain = ["Safe Demand", f"Correct Surge (≤ Upper {_b}%)", f"Overrun Breach (> Upper {_b}%)"]
    _points = _base.mark_point(filled=True, size=110, opacity=1).encode(
        y=_y,
        color=alt.Color(
            "signal:N",
            scale=alt.Scale(domain=_sig_domain, range=["#10B981", "#0284C7", "#DC2626"]),
            legend=alt.Legend(title="Signal", orient="top"),
        ),
        shape=alt.Shape(
            "signal:N",
            scale=alt.Scale(domain=_sig_domain, range=["circle", "diamond", "cross"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("date_str:N", title="Date"),
            alt.Tooltip("y_true:Q", title="Actual EAD", format=".2f"),
            alt.Tooltip("signal:N", title="Signal"),
        ],
    )
    _miss = (
        alt.Chart(_d[_pt_miss_mask])
        .mark_point(shape="circle", size=300, filled=False, color="#DC2626", strokeWidth=2.6)
        .encode(x=_x, y=_y, tooltip=[alt.Tooltip("date_str:N", title="Point Forecast Missed Surge")])
    )

    _rule = (
        alt.Chart(pd.DataFrame({"th": [_th]}))
        .mark_rule(color="#DC2626", strokeWidth=2.2, strokeDash=[6, 4])
        .encode(y="th:Q")
    )
    _rule_label = (
        alt.Chart(pd.DataFrame({"th": [_th], "t": [f"Alert Threshold: {_th:.1f} EAD"]}))
        .mark_text(align="right", baseline="bottom", dy=-4, dx=-4, color="#DC2626", fontSize=15, fontWeight="bold")
        .encode(y="th:Q", text="t:N", x=alt.value(1000))
    )

    cv_alt_chart = (
        (_actual + _pred + _bound + _rule + _rule_label + _points + _miss)
        .properties(width=1000, height=520)
        .interactive()
    )

    _kpi = mo.md(
        f"**Peak Containment Rate [Correct / (Wrong + Correct)]:** "
        f"<span style='color:#2CA0AB;font-size:28px'><b>{_rate:.1f}%</b></span> "
        f"**({_corr}/{_denom} surges)**<br>"
        f"• Safe EAD: **{_safe}/{_n} ({_safe/_n*100:.1f}%)** &nbsp;|&nbsp; "
        f"• Correct Signals: **{_corr}/{_n} ({_corr/_n*100:.1f}%)** &nbsp;|&nbsp; "
        f"• Wrong Signals: **{_wrong}/{_n} ({_wrong/_n*100:.1f}%)** &nbsp;|&nbsp; "
        f"• Point Forecast Capture: **{_pt_capt}/{_n_surges} ({_capt_pct:.0f}%)**"
    )
    mo.vstack([mo.ui.altair_chart(cv_alt_chart) if False else cv_alt_chart, _kpi])
    return


if __name__ == "__main__":
    app.run()
