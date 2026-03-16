"""
============================================================
  QuanSen — Streamlit GUI  (Main Entry Point)
  Module 4 of 4: Page configuration, sidebar, hero area,
  and all seven application tabs.
  
  Wraps the Quantitative Portfolio Optimizer engine
  by Amatra Sen without modifying the engine.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as _yf

from engines.market_state_engine import detect_market_state
from engines.ticker_search import get_best_ticker
from engines.config import RF_ANNUAL, SHRINKAGE_ALPHA

# ── Import QuanSen GUI modules ──────────────────────────────
from engines.config_and_state import (
    APP_VERSION, _ICO_B64, _LOGO_B64,
    SECTOR_PRESETS, APP_DATA_PATH,
    init_state, sync_user_store_to_session, persist_session_user_store,
    consume_bridge_symbols,
    current_portfolio_payload, load_portfolio_payload,
    reset_portfolio_outputs,
    normalize_symbol_list, parse_uploaded_symbols, apply_ticker_set,
    compute_max_w,
)
from engines.data_and_compute import (
    load_data, optimizer_expected_returns,
    cached_fetch_tape_quotes, cached_compute_momentum,
    cached_utility_portfolio, cached_tangency_portfolio,
    cached_min_risk_portfolio, cached_compute_frontier,
    cached_asset_risks, cached_correlation_matrix,
    cached_portfolio_stats, cached_portfolio_cumulative_returns,
    plotly_heatmap, search_ticker_api, get_ticker_date_range,
    generate_pdf,
)
from engines.ui_components import (
    GLOBAL_CSS,
    render_perf_status_panel, render_session_strip,
    show_flash_notice, render_hero_action_hub,
    render_build_workflow_overview, render_universe_helper,
    evaluate_alerts, render_ticker_tape,
    metric_card, weight_bar, make_weights_table,
    portfolio_metrics_row, render_insight_note,
    render_portfolio_spotlight, render_comparison_spotlights,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="QuanSen · Portfolio Optimizer",
    page_icon="\U0001f4d0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global styles ─────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────
init_state()

if not st.session_state.user_store_loaded:
    sync_user_store_to_session()
    st.session_state.user_store_loaded = True

consume_bridge_symbols()

try:
    _helper_mode = st.query_params.get("helper", "")
except Exception:
    _helper_mode = ""
if _helper_mode in {"upload", "sector"}:
    render_universe_helper(_helper_mode)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'''<div style="text-align:center;padding:0.4rem 0 1rem">
            <img src="data:image/png;base64,{_LOGO_B64}"
                 style="width:90px;height:90px;border-radius:18px;
                        box-shadow:0 0 18px rgba(0,180,255,0.35);">
            <div style="font-family:'Syne',sans-serif;font-size:0.65rem;
                        font-weight:700;color:#4a6a90;letter-spacing:0.18em;
                        text-transform:uppercase;margin-top:6px">
                Portfolio Optimizer v{APP_VERSION}
            </div>
        </div>''',
        unsafe_allow_html=True
    )

    st.markdown('<div class="card-title">⚙ Control Deck</div>', unsafe_allow_html=True)

    with st.expander("Universe Builder", expanded=True):
        if st.button("Load Test Portfolio (20 stocks)", use_container_width=True):
            st.session_state.tickers = [
                'BEL.NS','VEDL.NS','BAJFINANCE.NS','BEML.NS','ADVAIT.BO',
                'ADANIENT.NS','COALINDIA.NS','CROMPTON.NS','KINGFA.NS','KRISHNADEF.NS',
                'LT.NS','LUPIN.NS','MAZDOCK.NS','PENIND.NS','PNB.NS',
                'RELIANCE.NS','SHUKRAPHAR.BO','TARIL.NS','HDFCBANK.NS','SBIN.NS'
            ]
            st.session_state.start_date = "2021-02-01"
            st.session_state.end_date = "2026-03-06"
            reset_portfolio_outputs()
            st.rerun()

        use_sector_helper = st.toggle(
            "Use helper window for sectors",
            value=True,
            key="sector_helper_toggle",
            help="Opens the sector picker in a separate tab so this sidebar stays clean.",
        )
        if use_sector_helper:
            st.markdown(
                """
                <a class="builder-helper-link" href="?helper=sector" target="_blank">
                    <div class="builder-helper-title">Open Sector Basket Desk</div>
                    <div class="builder-helper-copy">Choose a sector preset in a separate tab and send it back here.</div>
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            sector_names = ["Custom"] + sorted(SECTOR_PRESETS.keys())
            selected_sector = st.selectbox(
                "Sector preset",
                options=sector_names,
                key="sector_preset_select",
                help="Load a ready-made basket instead of typing symbols one by one.",
            )
            sector_symbols = SECTOR_PRESETS.get(selected_sector, [])
            if sector_symbols:
                st.caption(f"{len(sector_symbols)} symbols ready in {selected_sector}.")
                col_sector_1, col_sector_2 = st.columns(2)
                with col_sector_1:
                    if st.button("Replace Basket", key="replace_sector_btn", use_container_width=True):
                        apply_ticker_set(sector_symbols, mode="replace")
                        st.success(f"Loaded {selected_sector}")
                        st.rerun()
                with col_sector_2:
                    if st.button("Append Sector", key="append_sector_btn", use_container_width=True):
                        apply_ticker_set(sector_symbols, mode="append")
                        st.success(f"Appended {selected_sector}")
                        st.rerun()

        if st.session_state.builder_prefill is not None:
            st.session_state.builder_symbols_raw = st.session_state.builder_prefill
            st.session_state.builder_prefill = None
        builder_default = st.session_state.get("builder_symbols_raw") or ", ".join(st.session_state.tickers[:20])
        builder_symbols_raw = st.text_area(
            "Paste symbols",
            value=builder_default,
            key="builder_symbols_raw",
            height=85,
            placeholder="SBIN.NS, HDFCBANK.NS, ICICIBANK.NS",
            help="Comma, newline, or semicolon separated symbols are all fine.",
        )
        use_upload_helper = st.toggle(
            "Use helper window for file upload",
            value=True,
            key="upload_helper_toggle",
            help="Opens the file uploader in a separate tab for CSV/TXT/Excel imports.",
        )
        imported_symbols = []
        if use_upload_helper:
            st.markdown(
                """
                <a class="builder-helper-link" href="?helper=upload" target="_blank">
                    <div class="builder-helper-title">Open File Upload Desk</div>
                    <div class="builder-helper-copy">Upload a basket file in a separate tab and send the cleaned symbols back here.</div>
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            upload = st.file_uploader(
                "Import symbols from CSV/XLSX/TXT",
                type=["csv", "txt", "xlsx", "xls"],
                key="builder_file_upload",
            )
            imported_symbols = parse_uploaded_symbols(upload)
        if imported_symbols:
            st.caption(f"Imported {len(imported_symbols)} symbols from file.")

        manual_symbols = normalize_symbol_list(builder_symbols_raw)
        candidate_symbols = imported_symbols or manual_symbols
        if candidate_symbols:
            st.caption(f"Builder preview: {len(candidate_symbols)} symbols")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Replace Basket", key="replace_basket_btn", use_container_width=True):
                if candidate_symbols:
                    apply_ticker_set(candidate_symbols, mode="replace")
                    st.success(f"Loaded {len(candidate_symbols)} symbols")
                    st.rerun()
                st.warning("Paste or import symbols first.")
        with b2:
            if st.button("Append Basket", key="append_basket_btn", use_container_width=True):
                if candidate_symbols:
                    apply_ticker_set(candidate_symbols, mode="append")
                    st.success(f"Appended {len(candidate_symbols)} symbols")
                    st.rerun()
                st.warning("Paste or import symbols first.")
        if st.button("Clear Builder", key="clear_builder_btn", use_container_width=True):
            st.session_state.builder_prefill = ""
            st.rerun()

        if st.session_state.tickers:
            st.markdown(
                '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;'
                'margin:0.65rem 0 0.45rem">Selected Basket</div>',
                unsafe_allow_html=True
            )
            preview = ", ".join(st.session_state.tickers[:12])
            if len(st.session_state.tickers) > 12:
                preview += f" ... +{len(st.session_state.tickers) - 12} more"
            st.caption(preview)
            st.caption(f"{len(st.session_state.tickers)} tickers active")
            if st.button("🗑 Clear All Tickers", use_container_width=True):
                st.session_state.tickers = []
                reset_portfolio_outputs()
                st.rerun()

    with st.expander("Date & Rules", expanded=True):
        date_mode = st.radio(
            "Date selection mode",
            ["📅 Start / End dates", "⏱ Days back from today"],
            label_visibility="collapsed",
            horizontal=False,
            key="date_mode"
        )

        if date_mode == "📅 Start / End dates":
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Start")
                st.session_state.start_date = st.text_input(
                    "Start", value=st.session_state.start_date,
                    label_visibility="collapsed", placeholder="YYYY-MM-DD", key="si_start"
                )
            with col2:
                st.caption("End")
                st.session_state.end_date = st.text_input(
                    "End", value=st.session_state.end_date,
                    label_visibility="collapsed", placeholder="YYYY-MM-DD", key="si_end"
                )
        else:
            import datetime as _dt
            days_back = st.number_input(
                "Number of days back from today", min_value=30, max_value=7300,
                value=365 * 3, step=30, key="days_back_input"
            )
            _today = _dt.date.today()
            _start = _today - _dt.timedelta(days=int(days_back))
            st.session_state.start_date = _start.strftime("%Y-%m-%d")
            st.session_state.end_date = _today.strftime("%Y-%m-%d")
            st.caption(f"▶ {st.session_state.start_date}  →  {st.session_state.end_date}  ({int(days_back)} days)")

        n_t = max(len(st.session_state.tickers), 1)
        auto_max = compute_max_w(n_t)
        st.caption(f"Auto max-weight for {n_t} assets: {auto_max*100:.0f}%")
        min_w_pct = st.slider("Min weight %", 0, 10, int(st.session_state.min_w * 100), 1)
        max_w_pct = st.slider("Max weight %", 5, 50, int(auto_max * 100), 5)
        st.session_state.min_w = min_w_pct / 100
        st.session_state.max_w = max_w_pct / 100

        target_pct = st.slider("Target return %", 1, 60, int(st.session_state.target_return * 100), 1)
        st.session_state.target_return = target_pct / 100

        # ─────────────────────────────────────────────
        # SHRINKAGE CONTROL
        # ─────────────────────────────────────────────

        auto_alpha = st.toggle(
            "Automatic α (market regime)",
            value=st.session_state.get("auto_alpha", True),
            key="auto_alpha_toggle",
            help="Automatically determine shrinkage α using the market regime engine."
        )

        st.session_state.auto_alpha = auto_alpha


        # ───────── AUTO MODE ─────────
        if auto_alpha and st.session_state.tickers:

            alpha_regime, _, regime_meta = detect_market_state(
                tuple(st.session_state.tickers),
                st.session_state.start_date,
                st.session_state.end_date
            )

            st.session_state.shrinkage_alpha = alpha_regime

            alpha_pct = int(alpha_regime * 100)

            st.caption(
                f"Automatic α = **{alpha_pct}%**  ·  regime: {max(regime_meta['regime_probabilities'], key=regime_meta['regime_probabilities'].get)}"
            )


        # ───────── MANUAL MODE ─────────
        else:

            alpha_pct = st.slider(
                "Shrinkage α",
                0,
                100,
                int(st.session_state.get("shrinkage_alpha", 0.7) * 100),
                5,
                help=(
                    "α controls how much to trust each stock's own historical mean vs the market index.\n\n"
                    "α=100%  → pure history\n"
                    "α=70%   → 70% history + 30% benchmark\n"
                    "α=0%    → fully benchmark driven"
                )
            )

            st.session_state.shrinkage_alpha = alpha_pct / 100


        # ───────── Display benchmark info ─────────
        _bm = st.session_state.get("bm_sym")

        st.caption(
            f"α = {int(st.session_state.shrinkage_alpha*100)}%  ·  benchmark = {_bm if _bm else '(auto after load)'}"
        )

        

    with st.expander("Momentum", expanded=False):
        
        

        mom_enabled = st.toggle(
        "Enable momentum tilt",
        value=st.session_state.momentum_enabled,
        key="mom_toggle",
        help="When ON, the optimizer uses β-blended expected returns."
    )

    st.session_state.momentum_enabled = mom_enabled


    # ─────────────────────────────────────────────
    # MOMENTUM CONTROLS
    # ─────────────────────────────────────────────
    if mom_enabled:

        # Toggle for automatic regime detection
        auto_regime = st.toggle(
            "Automatic regime detection",
            value=st.session_state.get("auto_regime", True),
            key="auto_regime_toggle",
            help="Use the market_state_engine to automatically determine α and β."
        )

        st.session_state.auto_regime = auto_regime


        # ─────────────────────────────────────────
        # AUTOMATIC REGIME MODE
        # ─────────────────────────────────────────
        if auto_regime and st.session_state.tickers:

            alpha_regime, beta_regime, regime_meta = detect_market_state(
                tuple(st.session_state.tickers),
                st.session_state.start_date,
                st.session_state.end_date
            )

            # Update parameters automatically
            st.session_state.shrinkage_alpha = alpha_regime
            st.session_state.momentum_beta = beta_regime


            st.markdown("#### 📊 Market Regime")

            trend = regime_meta["trend"]
            vol_ratio = regime_meta["vol_ratio"]
            drawdown = regime_meta["drawdown"]
            probs = regime_meta["regime_probabilities"]

        # ----------------------------
        # Metric Panel
        # ----------------------------
            st.markdown(
            f"""
            <div style="font-size:13px;line-height:1.7;margin-bottom:8px;">
            <b>Trend</b> <span style="float:right;">{trend:.3f}</span><br>
            <b>Volatility</b> <span style="float:right;">{vol_ratio:.2f}</span><br>
            <b>Drawdown</b> <span style="float:right;">{abs(drawdown):.2%}</span>
            </div>
            """,
            unsafe_allow_html=True
            )

            # ----------------------------
            # Probability Bars
            # ----------------------------
            colors = {
                "bull": "#00ff9c",
                "sideways": "#ffd166",
                "bear": "#ff4d6d",
                "crisis": "#9b5de5"
            }

            st.markdown("**Regime probabilities**")

            for regime, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):

                color = colors.get(regime, "#4a6a90")

                st.markdown(
                    f"""
                    <div style="margin-top:6px;font-size:12px;">
                    {regime.capitalize()} — {p*100:.1f}%
                    </div>

                    <div style="
                        background:#1a2132;
                        border-radius:6px;
                        overflow:hidden;
                        height:8px;
                        box-shadow:0 0 6px {color};
                    ">
                        <div style="
                            width:{p*100}%;
                            height:100%;
                            background:{color};
                            box-shadow:0 0 12px {color};
                        "></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ----------------------------
            # Dominant Regime
            # ----------------------------
            dominant = max(probs, key=probs.get)
            regime_scale = {
            "crisis": 0.1,
            "bear": 0.3,
            "sideways": 0.6,
            "bull": 0.9
                    }

            gauge_value = regime_scale.get(dominant, 0.5)

            labels = {
                "bull": "🟢 Bull Market",
                "sideways": "🟡 Sideways Market",
                "bear": "🟠 Bear Market",
                "crisis": "🔴 Crisis Market"
            }

            badge_colors = {
                "bull": "#00ff9c",
                "sideways": "#ffd166",
                "bear": "#ff4d6d",
                "crisis": "#9b5de5"
            }

            badge_color = badge_colors.get(dominant)

            st.markdown(
            f"""
            <div style="
            padding:8px;
            margin-top:15px;
            border-radius:8px;
            text-align:center;
            background:linear-gradient(135deg,{badge_color}33,#0f1724);
            border:1px solid {badge_color};
            box-shadow:0 0 10px {badge_color};
            font-weight:500;
            ">
            Detected Regime: {labels[dominant]}
            </div>
            """,
            unsafe_allow_html=True
            )

            # ----------------------------
            # Regime Gauge
            # ----------------------------

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                
                number={
                    "suffix": " score",
                    "font":{"size":28,"color":"#e6f0ff"}
                },

                gauge={
                    "axis":{
                        "range":[0,1],
                        "tickvals":[0.1,0.35,0.6,0.85],
                        "ticktext":["Crisis","Bear","Sideways","Bull"],
                        "tickfont":{"size":11,"color":"#cbd5e1"},
                    },

                    # thinner arc
                    "bar":{
                        "color":"#00e5ff",   # pointer colour
                        "thickness":0.12
                    },

                    "steps":[
                        {"range":[0,0.2],"color":"#7c3aed"},
                        {"range":[0.2,0.45],"color":"#ef4444"},
                        {"range":[0.45,0.75],"color":"#facc15"},
                        {"range":[0.75,1],"color":"#22c55e"}
                    ]
                }
            ))

            fig.update_layout(
                height=160,
                margin=dict(l=20,r=20,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color":"#e2e8f0","family":"Inter"}
            )

            st.plotly_chart(fig, use_container_width=True)
            # ----------------------------
            # Help / Formula Guide
            # ----------------------------
            with st.expander("How market regime is calculated"):
                st.markdown(
                """
            The regime detection engine uses three signals:

            **Trend**
            - distance of price from 200-day moving average  
            - slope of the 200-MA

            **Volatility**
            - 30-day volatility relative to 250-day volatility

            **Drawdown**
            - percentage drop from the historical peak in the selected window

            These signals are converted into regime probabilities for:

            • Bull  
            • Sideways  
            • Bear  
            • Crisis

            The dominant probability determines the detected regime.
                """
                )


        # ─────────────────────────────────────────
        # MANUAL MOMENTUM MODE
        # ─────────────────────────────────────────
        else:

            _lookback_options = list(range(42, 253, 21))

            current_lb = st.session_state.get("momentum_lookback", 252)

            lb_choice = st.select_slider(
                "Lookback window",
                options=_lookback_options,
                value=current_lb if current_lb in _lookback_options else 252,
                format_func=lambda v: f"{v}d (~{v//21}m)",
                help="Measured backward from the selected end date."
            )

            st.session_state.momentum_lookback = lb_choice


            beta_pct = st.slider(
                "β — history vs momentum",
                0,
                100,
                int(st.session_state.get("momentum_beta", 0.6) * 100),
                5
            )

            st.session_state.momentum_beta = beta_pct / 100

            st.caption(
                f"β = {beta_pct}%  ·  lookback = {lb_choice}d  ·  skip = 21d"
            )


        # ─────────────────────────────────────────
        # COMPUTE MOMENTUM
        # ─────────────────────────────────────────
        if st.session_state.data_loaded:

            momentum_requested = (
                st.button("⚡ Compute Momentum", use_container_width=True)
                or st.session_state.pop("request_compute_momentum", False)
            )

            if momentum_requested:

                with st.spinner("Computing momentum signals…"):

                    try:

                        scores, signals, final_er, meta = cached_compute_momentum(
                            tuple(st.session_state.tickers),
                            st.session_state.start_date,
                            st.session_state.end_date,
                            st.session_state.expected_returns,
                            lookback=st.session_state.get("momentum_lookback", 252),
                            skip=21,
                            beta=st.session_state.momentum_beta,
                            auto_lookback=st.session_state.auto_regime
                        )

                        st.session_state.momentum_scores = scores
                        st.session_state.momentum_signals = signals
                        st.session_state.momentum_final_er = final_er
                        st.session_state.momentum_meta = meta

                        st.session_state.asset_returns = final_er.reindex(
                            st.session_state.tickers
                        ) * 252

                        st.session_state.portfolios_computed = False
                        st.session_state.frontier_computed = False

                        n_strong = (signals == "Strong").sum()
                        n_weak = (signals == "Weak").sum()

                        st.success(
                            f"✔ Momentum computed  ·  🟢 {n_strong} Strong  ·  🔴 {n_weak} Weak"
                        )

                    except Exception as e:

                        import traceback

                        st.error(f"Momentum compute failed: {e}")
                        st.code(traceback.format_exc())

        else:
            st.caption("⚠ Load data first to compute momentum.")
    with st.expander("💾 Saved Portfolios", expanded=False):
        save_name = st.text_input("Portfolio name", key="save_portfolio_name", placeholder="e.g. India Momentum Basket")
        csave1, csave2 = st.columns(2)
        with csave1:
            if st.button("Save Current", key="save_current_portfolio", use_container_width=True):
                if st.session_state.tickers and save_name.strip():
                    st.session_state.saved_portfolios[save_name.strip()] = {
                        **current_portfolio_payload(),
                        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.active_saved_portfolio = save_name.strip()
                    persist_session_user_store()
                    st.success(f"Saved portfolio: {save_name.strip()}")
                else:
                    st.warning("Add tickers and enter a portfolio name first.")
        with csave2:
            if st.button("Quick Save", key="save_test_portfolio", use_container_width=True):
                name = save_name.strip() or f"Portfolio {pd.Timestamp.now().strftime('%H%M%S')}"
                st.session_state.saved_portfolios[name] = {
                    **current_portfolio_payload(),
                    "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.active_saved_portfolio = name
                persist_session_user_store()
                st.success(f"Saved portfolio: {name}")

        saved_names = sorted(st.session_state.saved_portfolios.keys())
        if saved_names:
            selected_saved = st.selectbox(
                "Saved sets",
                options=saved_names,
                index=saved_names.index(st.session_state.active_saved_portfolio) if st.session_state.active_saved_portfolio in saved_names else 0,
                key="saved_portfolio_picker",
            )
            st.session_state.active_saved_portfolio = selected_saved
            meta = st.session_state.saved_portfolios[selected_saved]
            st.caption(
                f"{len(meta.get('tickers', []))} tickers  ·  {meta.get('start_date')} → {meta.get('end_date')}  ·  "
                f"updated {meta.get('updated_at', 'N/A')}"
            )
            pl1, pl2 = st.columns(2)
            with pl1:
                if st.button("Load Saved", key="load_saved_portfolio", use_container_width=True):
                    load_portfolio_payload(meta)
                    st.success(f"Loaded: {selected_saved}")
                    st.rerun()
            with pl2:
                if st.button("Delete Saved", key="delete_saved_portfolio", use_container_width=True):
                    st.session_state.saved_portfolios.pop(selected_saved, None)
                    st.session_state.active_saved_portfolio = None
                    persist_session_user_store()
                    st.rerun()
        else:
            st.caption("No saved portfolios yet.")

    with st.expander("👀 Watchlists & Alerts", expanded=False):
        wl_name = st.text_input("Watchlist name", key="watchlist_name", placeholder="e.g. Defence Radar")
        if st.session_state.watchlist_prefill is not None:
            st.session_state.watchlist_symbols_raw = st.session_state.watchlist_prefill
            st.session_state.watchlist_prefill = None
        wl_symbols_default = st.session_state.get("watchlist_symbols_raw")
        if wl_symbols_default is None:
            wl_symbols_default = ", ".join(st.session_state.tickers[:12]) if st.session_state.tickers else ""
        wl_symbols_raw = st.text_area(
            "Symbols (comma separated)",
            value=wl_symbols_default,
            key="watchlist_symbols_raw",
            height=70,
        )
        w1, w2 = st.columns(2)
        with w1:
            if st.button("Save Watchlist", key="save_watchlist", use_container_width=True):
                symbols = [s.strip().upper() for s in wl_symbols_raw.split(",") if s.strip()]
                if wl_name.strip() and symbols:
                    st.session_state.watchlists[wl_name.strip()] = {
                        "symbols": symbols,
                        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.active_watchlist = wl_name.strip()
                    persist_session_user_store()
                    st.success(f"Saved watchlist: {wl_name.strip()}")
                else:
                    st.warning("Enter a watchlist name and at least one symbol.")
        with w2:
            if st.button("Use Current", key="watchlist_use_current", use_container_width=True):
                if st.session_state.tickers:
                    st.session_state.watchlist_prefill = ", ".join(st.session_state.tickers)
                    st.rerun()

        watch_names = sorted(st.session_state.watchlists.keys())
        if watch_names:
            selected_watch = st.selectbox(
                "Saved watchlists",
                options=watch_names,
                index=watch_names.index(st.session_state.active_watchlist) if st.session_state.active_watchlist in watch_names else 0,
                key="watchlist_picker",
            )
            st.session_state.active_watchlist = selected_watch
            watch_symbols = st.session_state.watchlists[selected_watch].get("symbols", [])
            st.caption(f"{len(watch_symbols)} symbols  ·  updated {st.session_state.watchlists[selected_watch].get('updated_at', 'N/A')}")
            watch_quotes = cached_fetch_tape_quotes(tuple(watch_symbols))
            if watch_quotes:
                watch_df = pd.DataFrame([
                    {
                        "Symbol": q["symbol"],
                        "Price": round(q["price"], 3),
                        "Chg %": round(q["chg"], 2),
                    }
                    for q in watch_quotes
                ])
                st.dataframe(watch_df, use_container_width=True, height=min(220, 45 + 35 * len(watch_df)))
            ww1, ww2 = st.columns(2)
            with ww1:
                if st.button("Load to Tickers", key="load_watchlist_tickers", use_container_width=True):
                    st.session_state.tickers = list(watch_symbols)
                    st.session_state.data_loaded = False
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed = False
                    st.rerun()
            with ww2:
                if st.button("Delete Watch", key="delete_watchlist", use_container_width=True):
                    st.session_state.watchlists.pop(selected_watch, None)
                    st.session_state.active_watchlist = None
                    persist_session_user_store()
                    st.rerun()
        else:
            st.caption("No watchlists saved yet.")

        st.markdown("---")
        st.caption("Price alerts")
        alert_symbol_default = st.session_state.tickers[0] if st.session_state.tickers else ""
        a1, a2 = st.columns(2)
        with a1:
            alert_symbol = st.text_input("Symbol", value=alert_symbol_default, key="alert_symbol")
        with a2:
            alert_condition = st.selectbox("Condition", ["above", "below"], key="alert_condition")
        a3, a4 = st.columns([1.2, 1])
        with a3:
            alert_threshold = st.number_input("Threshold", min_value=0.0, value=100.0, step=1.0, key="alert_threshold")
        with a4:
            alert_note = st.text_input("Note", key="alert_note", placeholder="optional")
        if st.button("Add Alert", key="add_alert_btn", use_container_width=True):
            if alert_symbol.strip():
                st.session_state.alerts.append({
                    "symbol": alert_symbol.strip().upper(),
                    "condition": alert_condition,
                    "threshold": float(alert_threshold),
                    "note": alert_note.strip(),
                    "enabled": True,
                })
                persist_session_user_store()
                st.success(f"Alert added for {alert_symbol.strip().upper()}")
                st.rerun()

        alert_hits, _quote_map = evaluate_alerts(st.session_state.alerts)
        if alert_hits:
            st.markdown('<div class="status-box status-warn">Triggered alerts</div>', unsafe_allow_html=True)
            for hit in alert_hits[:6]:
                st.caption(
                    f"{hit['symbol']} at {hit['price']:.3f} is {hit['condition']} {hit['threshold']:.3f}"
                    + (f" · {hit['note']}" if hit['note'] else "")
                )

        if st.session_state.alerts:
            for idx, alert in enumerate(st.session_state.alerts):
                c_alert1, c_alert2 = st.columns([4, 1])
                with c_alert1:
                    st.markdown(
                        f"`{alert['symbol']}`  {alert['condition']}  `{alert['threshold']:.3f}`"
                        + (f"  ·  {alert['note']}" if alert.get("note") else "")
                    )
                with c_alert2:
                    if st.button("✕", key=f"del_alert_{idx}"):
                        st.session_state.alerts.pop(idx)
                        persist_session_user_store()
                        st.rerun()

    with st.expander("Monitoring", expanded=False):
        st.markdown(
            '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;'
            'margin-bottom:0.5rem">📡 Live Indices Tape</div>',
            unsafe_allow_html=True
        )

        _ALL_INDICES = {
            "Nifty 50": "^NSEI", "BSE Sensex": "^BSESN", "Nifty Bank": "^NSEBANK",
            "Nifty IT": "^CNXIT", "Nifty Midcap": "NIFTY_MIDCAP_100.NS",
            "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI",
            "Russell 2000": "^RUT", "VIX": "^VIX", "FTSE 100": "^FTSE",
            "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI",
            "Shanghai Comp.": "000001.SS", "Gold": "GC=F", "Silver": "SI=F",
            "Crude Oil WTI": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F",
            "USD/INR": "INR=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
            "USD/JPY": "USDJPY=X", "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        }

        _default_indices = ["Nifty 50", "BSE Sensex", "Gold", "S&P 500", "Bitcoin", "USD/INR"]
        _tape_presets = {
            "India": ["Nifty 50", "BSE Sensex", "Nifty Bank", "Nifty IT", "Nifty Midcap"],
            "Global": ["S&P 500", "Nasdaq 100", "Dow Jones", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng"],
            "Macro": ["Gold", "Silver", "Crude Oil WTI", "Brent Crude", "Natural Gas", "USD/INR", "VIX"],
            "Crypto": ["Bitcoin", "Ethereum", "Gold", "USD/JPY"],
        }
        if "tape_indices" not in st.session_state:
            st.session_state.tape_indices = _default_indices

        st.caption("Pick your own strip or load a preset.")
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            if st.button("India", key="tape_preset_india", use_container_width=True):
                st.session_state.tape_indices = _tape_presets["India"]
                st.rerun()
        with p2:
            if st.button("Global", key="tape_preset_global", use_container_width=True):
                st.session_state.tape_indices = _tape_presets["Global"]
                st.rerun()
        with p3:
            if st.button("Macro", key="tape_preset_macro", use_container_width=True):
                st.session_state.tape_indices = _tape_presets["Macro"]
                st.rerun()
        with p4:
            if st.button("Crypto", key="tape_preset_crypto", use_container_width=True):
                st.session_state.tape_indices = _tape_presets["Crypto"]
                st.rerun()
        with p5:
            if st.button("Clear", key="tape_preset_clear", use_container_width=True):
                st.session_state.tape_indices = []
                st.rerun()

        if st.button("↻ Refresh Live Tape", key="tape_refresh_btn", use_container_width=True):
            cached_fetch_tape_quotes.clear()
            st.rerun()

        chosen = st.multiselect(
            "Choose indices for the tape",
            options=list(_ALL_INDICES.keys()),
            default=st.session_state.tape_indices,
            key="tape_multiselect",
            label_visibility="collapsed"
        )
        st.session_state.tape_indices = chosen

        st.markdown("---")
        render_perf_status_panel()

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.65rem;color:#263a56;text-align:center;margin-top:1rem">MPT · CVXPY · SciPy<br>Amatra Sen — QuanSen v{APP_VERSION}</div>', unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════
# ── Hero ──────────────────────────────────────────────────────
import datetime as _hero_dt
_now_str = _hero_dt.datetime.now().strftime("%A, %d %B %Y  ·  %H:%M:%S")
st.markdown(f'''
<div class="hero-banner" style="display:flex;align-items:center;gap:2rem;padding:1.4rem 2rem;">
    <img src="data:image/png;base64,{_LOGO_B64}"
         style="width:82px;height:82px;border-radius:16px;flex-shrink:0;
                box-shadow:0 0 24px rgba(0,180,255,0.45);
                animation:logoPulse 3s ease-in-out infinite;">
    <div style="flex:1">
        <div class="hero-title" style="margin-bottom:0.15rem">
            QUAN<span>SEN</span>
        </div>
        <div class="hero-subtitle">Quantitative Portfolio Optimizer · MPT Engine</div>
        <div class="hero-badge">Ultimate Risk Management Tool</div>
    </div>
    <div style="text-align:right;flex-shrink:0">
        <div style="font-size:0.7rem;color:#4a6a90;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:2px">Session time</div>
        <div id="qs-clock" style="font-family:'DM Mono',monospace;font-size:0.85rem;
                    color:#00b4ff;letter-spacing:0.05em">{_now_str}</div>
    </div>
</div>
<script>
(function() {{
    function pad(n){{ return n<10?'0'+n:n; }}
    function tick(){{
        var d=new Date();
        var days=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
        var months=['January','February','March','April','May','June',
                    'July','August','September','October','November','December'];
        var s=days[d.getDay()]+', '+pad(d.getDate())+' '+months[d.getMonth()]+' '+d.getFullYear()
             +'  ·  '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
        var el=document.getElementById('qs-clock');
        if(el) el.textContent=s;
    }}
    tick(); setInterval(tick,1000);
}})();
</script>
''', unsafe_allow_html=True)

show_flash_notice()
render_hero_action_hub()
render_session_strip()


# Tape prices refresh purely client-side via JS — no st.rerun() needed

# ── Live Ticker Tape ──────────────────────────────────────────
render_ticker_tape(
    st.session_state.get("tape_indices", []),
    {
        "Nifty 50":"^NSEI","BSE Sensex":"^BSESN","Nifty Bank":"^NSEBANK",
        "Nifty IT":"^CNXIT","Nifty Midcap":"NIFTY_MIDCAP_100.NS",
        "S&P 500":"^GSPC","Nasdaq 100":"^NDX","Dow Jones":"^DJI",
        "Russell 2000":"^RUT","VIX":"^VIX","FTSE 100":"^FTSE",
        "DAX":"^GDAXI","Nikkei 225":"^N225","Hang Seng":"^HSI",
        "Shanghai Comp.":"000001.SS","Gold":"GC=F","Silver":"SI=F",
        "Crude Oil WTI":"CL=F","Brent Crude":"BZ=F","Natural Gas":"NG=F",
        "USD/INR":"INR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
        "USD/JPY":"USDJPY=X","Bitcoin":"BTC-USD","Ethereum":"ETH-USD",
    }
)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_build, tab_data, tab_portfolios, tab_frontier, tab_analysis, tab_backtest, tab_export = st.tabs([
    "📋 Build Portfolio",
    "📊 Market Data",
    "🏆 Portfolios",
    "📈 Efficient Frontier",
    "🔬 Analysis",
    "🧪 Backtest",
    "💾 Export",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — BUILD PORTFOLIO
# ══════════════════════════════════════════════════════════════
with tab_build:
    render_build_workflow_overview()
    st.markdown(
        f"""
        <div class="build-action-rail">
            <div class="build-action-card">
                <div class="workflow-kicker">Quick Flow</div>
                <div class="headline">Build → Load → Tilt → Run</div>
                <div class="copy">Use the search desk, lock the common market window, optionally compute momentum, then fire the full optimizer stack.</div>
            </div>
            <div class="build-action-card">
                <div class="workflow-kicker">Basket</div>
                <div class="headline">{len(st.session_state.tickers)} symbols</div>
                <div class="copy">Enough names for a proper optimization set? Aim for at least 5 if you want diversification to show up.</div>
            </div>
            <div class="build-action-card">
                <div class="workflow-kicker">Data</div>
                <div class="headline">{'Loaded' if st.session_state.data_loaded else 'Awaiting load'}</div>
                <div class="copy">The optimizer only trusts the strict common-overlap window after a successful data load.</div>
            </div>
            <div class="build-action-card">
                <div class="workflow-kicker">Desk</div>
                <div class="headline">{'Results live' if st.session_state.portfolios_computed or st.session_state.frontier_computed else 'Ready to stage'}</div>
                <div class="copy">Run `ALL` when you want the quickest comparison across utility, tangency, frontier, and min-risk.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="workflow-section-shell">
            <div class="workflow-section-topline">
                <div>
                    <div class="workflow-section-kicker">Stage 1</div>
                    <div class="section-heading" style="margin:0.2rem 0 0 0">Build The Universe</div>
                </div>
                <div class="workflow-section-copy">Search fast, add manually, then tighten the basket before you load the window.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_search, col_manual = st.columns([1.1, 1])

    # ── Ticker search ──────────────────────────────────────────
    with col_search:
        st.markdown('<div class="section-heading">Search & Add Tickers</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="status-box status-info">Desk search: press Enter or click Search to query Yahoo Finance.</div>',
            unsafe_allow_html=True
        )
        with st.form("ticker_search_form", clear_on_submit=False):
            search_query = st.text_input(
                "Company name",
                placeholder="e.g. Reliance Industries",
                key="search_input"
            )
            search_btn = st.form_submit_button("🔍 Search", use_container_width=True)

        if search_btn and search_query.strip():
            with st.spinner("Querying Yahoo Finance..."):
                results = search_ticker_api(search_query.strip())
                st.session_state.search_results = results

        if st.session_state.search_results:
            st.markdown('<div style="font-size:0.7rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.4rem">Results</div>', unsafe_allow_html=True)
            for q in st.session_state.search_results:
                name   = q.get('shortname', q.get('longname', 'N/A'))
                symbol = q.get('symbol', '')
                exch   = q.get('exchDisp', '')
                if not symbol:
                    continue
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div style="padding:0.35rem 0">
                        <span style="color:#e8f4ff;font-size:0.82rem">{name}</span><br>
                        <span class="ticker-chip">{symbol}</span>
                        <span style="font-size:0.68rem;color:#4a6a90;margin-left:5px">{exch}</span>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if st.button("Add", key=f"add_{symbol}_{name[:6]}"):
                        if symbol not in st.session_state.tickers:
                            # Resolve best exchange for Indian stocks
                            with st.spinner(f"Resolving {symbol}..."):
                                best = get_best_ticker(
                                    symbol,
                                    st.session_state.start_date,
                                    st.session_state.end_date
                                )
                            st.session_state.tickers.append(best)
                            st.session_state.data_loaded = False
                            st.session_state.portfolios_computed = False
                            st.session_state.frontier_computed = False
                            st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                            st.rerun()
                        else:
                            st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

    # ── Manual entry + remove ──────────────────────────────────
    with col_manual:
        st.markdown('<div class="section-heading">Manual Entry & Manage</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="status-box status-info">Quick add: broker-style symbols like `SBIN` will be auto-resolved when possible.</div>',
            unsafe_allow_html=True
        )
        with st.form("manual_add_form", clear_on_submit=False):
            manual_ticker = st.text_input(
                "Enter ticker directly",
                placeholder="e.g. AAPL, TCS.NS",
                key="manual_input"
            )
            manual_btn = st.form_submit_button("➕ Add Manually", use_container_width=True)
        if manual_btn:
            t = manual_ticker.strip().upper()
            if t:
                if t not in st.session_state.tickers:
                    with st.spinner(f"Resolving {t}..."):
                        best = get_best_ticker(t, st.session_state.start_date, st.session_state.end_date)
                    st.session_state.tickers.append(best)
                    st.session_state.data_loaded = False
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed = False
                    st.markdown(f'<div class="status-box status-success">✔ Added {best}</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="status-box status-warn">Already in portfolio.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="workflow-section-shell">
            <div class="workflow-section-topline">
                <div>
                    <div class="workflow-section-kicker">Stage 2</div>
                    <div class="section-heading" style="margin:0.2rem 0 0 0">Load And Execute</div>
                </div>
                <div class="workflow-section-copy">Freeze the market window, confirm the return model, then fire the optimizer stack.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(st.session_state.tickers) < 2:
        st.markdown('<div class="status-box status-warn">⚠ Add at least 2 tickers to load data.</div>', unsafe_allow_html=True)
    else:
        load_requested = (
            st.button("⬇ Download & Load Data", use_container_width=True) or
            st.session_state.pop("request_load_data", False)
        )
        if load_requested:
            with st.spinner("Downloading price data and computing shrinkage-adjusted returns…"):
                try:
                    
                    # Detect market regime
                    alpha_regime, beta_regime, regime_meta = detect_market_state(
                        tuple(st.session_state.tickers),
                        st.session_state.start_date,
                        st.session_state.end_date
                    )

                    st.session_state.beta_regime = beta_regime
                    st.session_state.beta_regime = max(0.25, min(0.9, beta_regime))

                    (tickers_out,
                     returns,
                     expected_returns,
                     cov_matrix,
                     raw_er,
                     bm_sym) = load_data(
                        tuple(st.session_state.tickers),
                        st.session_state.start_date,
                        st.session_state.end_date,
                        alpha=alpha_regime,)
                    
                    st.session_state.tickers          = tickers_out
                    st.session_state.returns          = returns
                    st.session_state.expected_returns = expected_returns
                    st.session_state.cov_matrix       = cov_matrix
                    st.session_state.raw_er           = raw_er
                    st.session_state.bm_sym           = bm_sym
                    st.session_state.asset_returns    = expected_returns * 252
                    st.session_state.asset_risks      = cached_asset_risks(returns)
                    st.session_state.data_loaded      = True
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed   = False
                    # Clear momentum — stale signals don't apply to new data
                    st.session_state.momentum_scores    = None
                    st.session_state.momentum_signals   = None
                    st.session_state.momentum_final_er  = None
                    st.session_state.momentum_meta      = None

                    alpha = st.session_state.shrinkage_alpha
                    if len(returns) <= 0:
                        raise ValueError(
                            "No usable return history remained after cleaning. "
                            "Try a later start date or reduce the basket."
                        )
                    overlap_note = ""
                    if len(returns.index) > 0:
                        overlap_note = (
                            f" Effective optimizer window: "
                            f"**{returns.index[0].strftime('%Y-%m-%d')}** -> "
                            f"**{returns.index[-1].strftime('%Y-%m-%d')}**."
                        )
                    if bm_sym:
                        message = (
                            f"✔ Loaded {len(returns)} trading days for {len(tickers_out)} assets. "
                            f"Returns shrunk toward **{bm_sym}**  (α={alpha:.0%})."
                            f"{overlap_note}"
                        )
                        level = "success"
                    else:
                        message = (
                            f"✔ Loaded {len(returns)} trading days for {len(tickers_out)} assets. "
                            "Benchmark unavailable — using raw expected returns."
                            f"{overlap_note}"
                        )
                        level = "warning"
                    st.session_state.flash_notice = (level, message)
                    st.rerun()
                except Exception as e:
                    st.error(f"Data load failed: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── Step 2: Run optimizations ──────────────────────────────
    # Resolve which expected returns the optimizer should use
    def _active_er():
        return optimizer_expected_returns()

    if st.session_state.data_loaded:
        exec_info_1, exec_info_2, exec_info_3 = st.columns(3)
        with exec_info_1:
            st.markdown(
                f"""
                <div class="workflow-metric">
                    <div class="workflow-metric-label">Data State</div>
                    <div class="workflow-metric-value">Loaded</div>
                    <div class="workflow-metric-copy">{len(st.session_state.returns)} trading rows across {len(st.session_state.tickers)} assets</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with exec_info_2:
            active_model = "Momentum blend" if (
                st.session_state.momentum_enabled and st.session_state.momentum_final_er is not None
            ) else "Shrinkage core"
            st.markdown(
                f"""
                <div class="workflow-metric">
                    <div class="workflow-metric-label">Model In Play</div>
                    <div class="workflow-metric-value">{active_model}</div>
                    <div class="workflow-metric-copy">the optimizer is using {active_model.lower()} expected returns</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with exec_info_3:
            momentum_state = (
                "Ready" if st.session_state.momentum_final_er is not None else
                "Enabled" if st.session_state.momentum_enabled else
                "Off"
            )
            st.markdown(
                f"""
                <div class="workflow-metric">
                    <div class="workflow-metric-label">Momentum Desk</div>
                    <div class="workflow-metric-value">{momentum_state}</div>
                    <div class="workflow-metric-copy">compute momentum when you want the optimizer to tilt off raw shrinkage</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.6rem">Optimization Actions</div>', unsafe_allow_html=True)
        run_col1, run_col2, run_col3 = st.columns(3)

        with run_col1:
            if st.button("⚡ Utility Portfolio", use_container_width=True):
                with st.spinner("Optimizing utility portfolio…"):
                    try:
                        w = cached_utility_portfolio(
                            _active_er(),
                            st.session_state.cov_matrix,
                            tuple(st.session_state.tickers),
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_utility = w
                        st.success("Utility portfolio computed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with run_col2:
            if st.button("🌟 Tangency Portfolio", use_container_width=True):
                with st.spinner("Maximising Sharpe ratio…"):
                    try:
                        w, tr, tk, ts = cached_tangency_portfolio(
                            _active_er(),
                            st.session_state.cov_matrix,
                            tuple(st.session_state.tickers),
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_tan  = w
                        st.session_state.tan_return   = tr
                        st.session_state.tan_risk     = tk
                        st.session_state.tan_sharpe   = ts
                        st.success("Tangency portfolio computed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with run_col3:
            if st.button("🎯 Min-Risk Portfolio", use_container_width=True):
                with st.spinner("Minimising risk for target return…"):
                    try:
                        w = cached_min_risk_portfolio(
                            _active_er(),
                            st.session_state.cov_matrix,
                            tuple(st.session_state.tickers),
                            st.session_state.target_return,
                            st.session_state.min_w,
                            st.session_state.max_w
                        )
                        st.session_state.weights_min = w
                        if w is not None:
                            st.success(f"Min-risk portfolio computed (target: {st.session_state.target_return*100:.1f}%).")
                        else:
                            st.warning("No feasible solution for this target. Try a lower target return.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        run_all = (
            st.button("🚀 Run ALL Optimizations + Frontier", use_container_width=True) or
            st.session_state.pop("request_run_all", False)
        )
        if run_all:
            progress = st.progress(0, text="Running all optimizations…")
            try:
                progress.progress(10, "Utility portfolio…")
                w_u = cached_utility_portfolio(
                    _active_er(), st.session_state.cov_matrix,
                    tuple(st.session_state.tickers), st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_utility = w_u
                progress.progress(35, "Tangency portfolio…")
                w_t, tr, tk, ts = cached_tangency_portfolio(
                    _active_er(), st.session_state.cov_matrix,
                    tuple(st.session_state.tickers), st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_tan  = w_t
                st.session_state.tan_return   = tr
                st.session_state.tan_risk     = tk
                st.session_state.tan_sharpe   = ts
                progress.progress(60, "Efficient frontier…")
                fr, ret = cached_compute_frontier(
                    _active_er(), st.session_state.cov_matrix,
                    st.session_state.min_w, st.session_state.max_w)
                st.session_state.frontier_risks    = fr
                st.session_state.frontier_returns  = ret
                st.session_state.frontier_computed = True
                progress.progress(85, "Min-risk portfolio…")
                w_m = cached_min_risk_portfolio(
                    _active_er(), st.session_state.cov_matrix,
                    tuple(st.session_state.tickers), st.session_state.target_return,
                    st.session_state.min_w, st.session_state.max_w)
                st.session_state.weights_min = w_m
                st.session_state.portfolios_computed = True
                progress.progress(100, "Done ✔")
                st.success("All optimizations complete. Navigate the tabs above.")
            except Exception as e:
                st.error(f"Error during run: {e}")

        # ── Frontier separately ────────────────────────────────
        frontier_requested = (
            st.button("📉 Compute Efficient Frontier Only", use_container_width=True) or
            st.session_state.pop("request_frontier", False)
        )
        if frontier_requested:
            with st.spinner("Computing efficient frontier (100 points)…"):
                try:
                    fr, ret = cached_compute_frontier(
                        _active_er(),
                        st.session_state.cov_matrix,
                        st.session_state.min_w,
                        st.session_state.max_w
                    )
                    st.session_state.frontier_risks   = fr
                    st.session_state.frontier_returns = ret
                    st.session_state.frontier_computed = True
                    st.success(f"Frontier computed: {len(fr)} feasible points.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown(
        """
        <div class="workflow-section-shell">
            <div class="workflow-section-topline">
                <div>
                    <div class="workflow-section-kicker">Stage 3</div>
                    <div class="section-heading" style="margin:0.2rem 0 0 0">Audit The Basket</div>
                </div>
                <div class="workflow-section-copy">Review listing coverage, prune weak names, and sanity-check the sample before deeper analysis.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.session_state.tickers:
        st.markdown(
            '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem">Ticker Coverage Audit</div>',
            unsafe_allow_html=True
        )
        for i, t in enumerate(st.session_state.tickers):
            first_date, last_date, colour, age = get_ticker_date_range(t)
            age_label = f"{age}d ago" if age < 9999 else "unknown"
            try:
                import datetime as _dt2
                chosen_start = _dt2.date.fromisoformat(st.session_state.start_date)
                ticker_start = _dt2.date.fromisoformat(first_date) if first_date != "N/A" else None
                start_warn = ticker_start and chosen_start < ticker_start
            except Exception:
                start_warn = False
            c1, c2 = st.columns([5, 1])
            with c1:
                warn_html = (
                    f'<span style="font-size:0.65rem;color:#ff5252;margin-left:6px">'
                    f'⚠ data starts {first_date} — your start date is earlier</span>'
                ) if start_warn else ""
                st.markdown(
                    f'<span class="ticker-chip">{t}</span>'
                    f'<span style="font-size:0.7rem;color:#4a90d9;margin-left:8px">from </span>'
                    f'<span style="font-size:0.7rem;color:#a0c8e8;">{first_date}</span>'
                    f'<span style="font-size:0.7rem;color:#4a90d9;margin-left:6px"> to </span>'
                    f'<span style="font-size:0.7rem;color:{colour};">{last_date}</span>'
                    f'<span style="font-size:0.65rem;color:#4a6a90;margin-left:5px">({age_label})</span>'
                    + warn_html,
                    unsafe_allow_html=True
                )
            with c2:
                if st.button("✕", key=f"rm_{t}_{i}"):
                    st.session_state.tickers.remove(t)
                    st.session_state.data_loaded = False
                    st.session_state.portfolios_computed = False
                    st.session_state.frontier_computed = False
                    st.rerun()
    else:
        st.markdown('<div class="status-box status-info">No tickers yet. Search or add manually.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — MARKET DATA
# ══════════════════════════════════════════════════════════════
with tab_data:
    if not st.session_state.data_loaded:
        render_insight_note("This tab becomes your diagnostic layer after a data load. It explains what the return engine believes about each asset before the optimizer starts assigning weights.")
    else:
        returns          = st.session_state.returns
        expected_returns = st.session_state.expected_returns
        asset_returns    = st.session_state.asset_returns
        asset_risks      = st.session_state.asset_risks
        tickers          = st.session_state.tickers

        # ── Shrinkage comparison panel ─────────────────────────
        raw_er = st.session_state.raw_er
        bm_sym = st.session_state.bm_sym
        alpha  = st.session_state.shrinkage_alpha

        if raw_er is not None and bm_sym is not None:
            st.markdown('<div class="section-heading">Return Shrinkage Adjustment</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-box status-info">' +
                f'Shrinkage  <b>α = {alpha:.0%}</b>  ·  Benchmark anchored to  <b>{bm_sym}</b>.<br>' +
                f'Each stock\'s expected return = {alpha:.0%} × its own mean  +  {1-alpha:.0%} × {bm_sym} mean.' +
                '</div>',
                unsafe_allow_html=True
            )
            shrink_df = pd.DataFrame({
                "Ticker":       tickers,
                "Raw ann. %":   (raw_er.values * 252 * 100).round(2),
                "Adj ann. %":   (expected_returns.values * 252 * 100).round(2),
                "Δ (pp)":       ((expected_returns.values - raw_er.values) * 252 * 100).round(2),
            }).set_index("Ticker")

            def _colour_delta(val):
                if val < -5:   return "color: #ff5252; font-weight:600"
                elif val < 0:  return "color: #ff8a65"
                elif val > 5:  return "color: #69f0ae; font-weight:600"
                elif val > 0:  return "color: #b9f6ca"
                return ""

            st.dataframe(
                shrink_df.style
                    .background_gradient(cmap="Blues",  subset=["Adj ann. %"])
                    .background_gradient(cmap="Oranges", subset=["Raw ann. %"])
                    .applymap(_colour_delta, subset=["Δ (pp)"]),
                use_container_width=True
            )
            st.caption(
                "Δ (pp) = adjusted − raw.  "
                "Negative = return pulled down (outlier tamed).  "
                "Large red values = stocks the raw optimizer would have over-weighted."
            )
            st.markdown("---")
        elif raw_er is not None and bm_sym is None:
            st.markdown(
                '<div class="status-box status-warn">⚠ Benchmark unavailable — raw expected returns used (no shrinkage applied).</div>',
                unsafe_allow_html=True
            )

        # ── Momentum Signal Table ──────────────────────────────
        if (st.session_state.momentum_enabled and
                st.session_state.momentum_signals is not None):

            scores   = st.session_state.momentum_scores
            signals  = st.session_state.momentum_signals
            final_er = st.session_state.momentum_final_er
            meta     = st.session_state.momentum_meta or {}
            beta     = st.session_state.momentum_beta
            lb       = st.session_state.momentum_lookback

            st.markdown('<div class="section-heading">Momentum Signals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-box status-info">' +
                f'Lookback <b>{lb} trading days</b> from the selected end date  ·  Skip last <b>21 days</b>  ·  ' +
                f'β = <b>{beta:.0%}</b>  (history weight)<br>' +
                f'μ_final = {beta:.0%} × μ_shrinkage  +  {1-beta:.0%} × momentum_scaled' +
                '</div>',
                unsafe_allow_html=True
            )
            if meta.get("entry_date") and meta.get("exit_date"):
                st.caption(
                    f"Effective momentum window: {meta['entry_date']} → {meta['exit_date']} "
                    f"(selected sample end date: {meta.get('end_date', st.session_state.end_date)})."
                )

            def _signal_icon(s):
                return {"Strong": "🟢 Strong", "Neutral": "🟡 Neutral", "Weak": "🔴 Weak"}.get(s, s)

            mom_df = pd.DataFrame({
                "Ticker":           tickers,
                "Momentum %":       (scores.reindex(tickers).values * 100).round(2),
                "Signal":           [_signal_icon(signals.get(t, "Neutral")) for t in tickers],
                "μ_shrinkage ann%": (expected_returns.values * 252 * 100).round(2),
                "μ_final ann%":     (final_er.reindex(tickers).values * 252 * 100).round(2),
                "Δ vs shrinkage":   ((final_er.reindex(tickers).values - expected_returns.values) * 252 * 100).round(2),
            }).set_index("Ticker")

            def _col_momentum(val):
                if val > 15:   return "color:#00e676;font-weight:600"
                elif val > 0:  return "color:#b9f6ca"
                elif val < -15:return "color:#ff5252;font-weight:600"
                elif val < 0:  return "color:#ff8a65"
                return ""

            def _col_delta(val):
                if val > 2:    return "color:#00e676;font-weight:600"
                elif val > 0:  return "color:#b9f6ca"
                elif val < -2: return "color:#ff5252;font-weight:600"
                elif val < 0:  return "color:#ff8a65"
                return ""

            st.dataframe(
                mom_df.style
                    .applymap(_col_momentum, subset=["Momentum %"])
                    .applymap(_col_delta,    subset=["Δ vs shrinkage"])
                    .background_gradient(cmap="RdYlGn", subset=["μ_final ann%"]),
                use_container_width=True
            )

            # Summary counts
            n_strong  = (signals == "Strong").sum()
            n_neutral = (signals == "Neutral").sum()
            n_weak    = (signals == "Weak").sum()
            st.caption(
                f"🟢 Strong: {n_strong}  ·  🟡 Neutral: {n_neutral}  ·  🔴 Weak: {n_weak}  ·  "
                f"Optimizer is using μ_final (momentum-blended) returns."
            )
            st.markdown("---")

        elif st.session_state.momentum_enabled and st.session_state.momentum_signals is None:
            st.markdown(
                '<div class="status-box status-warn">⚠ Momentum enabled but not yet computed — ' +
                'click ⚡ Compute Momentum in the sidebar.</div>',
                unsafe_allow_html=True
            )

        # ── Summary table ──────────────────────────────────────
        # Always reflect the active expected returns:
        # momentum-blended if enabled+computed, else shrinkage
        _mom_on = (st.session_state.momentum_enabled and
                   st.session_state.momentum_final_er is not None)
        _active  = (st.session_state.momentum_final_er.reindex(tickers)
                    if _mom_on else expected_returns)
        _er_label = "μ_final ann% (mom)" if _mom_on else "μ_shrinkage ann%"

        st.markdown('<div class="section-heading">Asset Summary</div>', unsafe_allow_html=True)
        if _mom_on:
            st.caption("⚡ Showing momentum-blended μ_final — these are the returns the optimizer uses.")
        summary_df = pd.DataFrame({
            "Ticker"        : tickers,
            _er_label       : (_active.values * 252 * 100).round(2),
            "Ann. Risk %"   : (asset_risks.values * 100).round(2),
            "Sharpe"        : ((_active.values * 252 - RF_ANNUAL) / asset_risks.values).round(3),
            "Daily Mean %"  : (_active.values * 100).round(4),
        })
        summary_df = summary_df.set_index("Ticker")
        st.dataframe(summary_df.style
            .background_gradient(cmap="Blues",  subset=[_er_label])
            .background_gradient(cmap="Reds",   subset=["Ann. Risk %"])
            .background_gradient(cmap="Greens", subset=["Sharpe"]),
            use_container_width=True)

        # ── Correlation heatmap (Plotly) ───────────────────────
        st.markdown('<div class="section-heading">Correlation Matrix</div>', unsafe_allow_html=True)
        render_insight_note("High positive clusters here are where diversification starts to disappear. If many names move together, the optimizer has fewer true ways to spread risk.")
        corr = cached_correlation_matrix(returns)
        fig_corr = plotly_heatmap(
            corr, "Asset Correlation Matrix",
            colorscale="RdBu", zmid=0, fmt=".2f",
            height=max(420, len(tickers) * 28 + 120)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Hover over any cell for full ticker names and exact value.  "
                   "Labels show abbreviated ticker (exchange suffix stripped).")

        # ── Returns distribution ───────────────────────────────
        st.markdown('<div class="section-heading">Return Distributions</div>', unsafe_allow_html=True)
        render_insight_note("These histograms show how noisy each name is day to day. Wide, skewed, or jumpy distributions often explain why a stock gets capped or downweighted.")
        n_cols = min(4, len(tickers))
        n_rows = (len(tickers) + n_cols - 1) // n_cols
        fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8))
        fig_dist.patch.set_facecolor("#0a0e17")
        axes_flat = np.array(axes).flatten() if len(tickers) > 1 else [axes]
        for idx, t in enumerate(tickers):
            ax2 = axes_flat[idx]
            ax2.set_facecolor("#0d1525")
            ax2.hist(returns[t].dropna(), bins=40, color="#0066cc", alpha=0.75, edgecolor="#003388")
            ax2.axvline(returns[t].mean(), color="#00e676", linestyle="--", linewidth=1.2, label="Mean")
            ax2.set_title(t, color="#80c8ff", fontsize=8, pad=4)
            ax2.tick_params(colors="#4a6a90", labelsize=6)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#1a2d4d")
        for idx in range(len(tickers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle("Daily Return Distributions", color="#80b0d0", fontsize=11, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_dist)

        # ── Risk-Return scatter ────────────────────────────────
        st.markdown('<div class="section-heading">Risk–Return Scatter</div>', unsafe_allow_html=True)
        render_insight_note("This is the raw opportunity map for individual assets. The optimizer generally prefers points that sit higher for the same level of risk.")
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=asset_risks.values  * 100,
            y=asset_returns.values * 100,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            textfont=dict(size=10, color="#80c8ff"),
            marker=dict(
                size=12,
                color=(asset_returns.values - RF_ANNUAL) / asset_risks.values,
                colorscale='RdYlGn',
                colorbar=dict(title="Sharpe", tickfont=dict(color="#80b0d0")),
                showscale=True,
                line=dict(width=1, color="#001428")
            ),
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
        ))
        fig_scat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#0d1525",
            xaxis_title="Annual Risk σ (%)",
            yaxis_title="Annual Return μ (%)",
            title=dict(text="Individual Asset Risk–Return", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            height=440,
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig_scat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIOS
# ══════════════════════════════════════════════════════════════
with tab_portfolios:
    if not st.session_state.data_loaded:
        render_insight_note("Load data in the Build tab first. Once the common market window is locked, this desk will show allocation spotlights and portfolio rankings.")
    else:
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
        cov     = st.session_state.cov_matrix
        render_insight_note("This desk is for allocation decisions. Utility shows the most comfortable risk-adjusted utility mix, Tangency shows the cleanest excess-return efficiency, and Min-Risk shows the calmest feasible path to your target.")

        comparison_seed = {}
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            comparison_seed["Utility"] = {"Return %": ann_ret * 100, "Risk %": ann_risk * 100, "Sharpe": sharpe}
        if st.session_state.weights_tan is not None:
            comparison_seed["Tangency"] = {
                "Return %": st.session_state.tan_return * 100,
                "Risk %": st.session_state.tan_risk * 100,
                "Sharpe": st.session_state.tan_sharpe,
            }
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            comparison_seed["Min-Risk"] = {"Return %": ann_ret * 100, "Risk %": ann_risk * 100, "Sharpe": sharpe}

        best_sharpe_name = None
        if comparison_seed:
            best_sharpe_name = max(comparison_seed.items(), key=lambda item: item[1]["Sharpe"])[0]

        st.markdown('<div class="section-heading">Utility-Maximized Portfolio</div>', unsafe_allow_html=True)
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            fig_pie = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers))},60%,50%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                  title=dict(text="Utility Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                  showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                  height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Utility-Maximized Portfolio",
                "Best when you want the optimizer to respect the risk penalty and still keep expected return in play.",
                w, ann_ret, ann_risk, sharpe, fig_pie,
                badge="Risk utility lens",
                badge_gold=(best_sharpe_name == "Utility")
            )
        else:
            render_insight_note("Utility is waiting for its first run. Use this when you want the most comfortable risk-adjusted mix under your constraints.")

        st.markdown("---")
        st.markdown('<div class="section-heading">Tangency Portfolio (Max Sharpe)</div>', unsafe_allow_html=True)
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            tr = st.session_state.tan_return
            tk = st.session_state.tan_risk
            ts = st.session_state.tan_sharpe
            fig_pie2 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+40)},65%,52%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie2.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                   title=dict(text="Tangency Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                   showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                   height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Tangency Portfolio",
                "Best when you want the highest excess return per unit of risk and a clean benchmark against every other portfolio.",
                w, tr, tk, ts, fig_pie2,
                badge="Best Sharpe" if best_sharpe_name == "Tangency" else "Efficiency lens",
                badge_gold=(best_sharpe_name == "Tangency")
            )
        else:
            render_insight_note("Tangency has not been computed yet. This is the portfolio to use when you want the sharpest return-to-risk benchmark.")

        st.markdown("---")
        st.markdown(f'<div class="section-heading">Min-Risk Portfolio (Target: {st.session_state.target_return*100:.1f}%)</div>', unsafe_allow_html=True)
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            fig_pie3 = go.Figure(go.Pie(
                labels=tickers, values=w * 100, hole=0.45, textinfo='label+percent', textfont=dict(size=10),
                marker=dict(colors=[f"hsl({int(i*360/len(tickers)+80)},60%,48%)" for i in range(len(tickers))],
                            line=dict(color='#0a0e17', width=2))
            ))
            fig_pie3.update_layout(template="plotly_dark", paper_bgcolor="#0a0e17",
                                   title=dict(text="Min-Risk Portfolio Allocation", font=dict(color="#a0c8e8", size=13)),
                                   showlegend=True, legend=dict(font=dict(size=9, color="#80b0d0")),
                                   height=380, margin=dict(l=10, r=10, t=50, b=10))
            render_portfolio_spotlight(
                "Min-Risk Portfolio",
                "Best when the brief is capital preservation first and you only want enough return to clear the target hurdle.",
                w, ann_ret, ann_risk, sharpe, fig_pie3,
                badge="Target discipline",
                badge_gold=(best_sharpe_name == "Min-Risk")
            )
        else:
            render_insight_note("Min-Risk will appear once you solve a feasible target-return problem. Lower the target if the desk cannot find a clean solution.")

        # ── Side-by-side comparison ────────────────────────────
        any_computed = any([
            st.session_state.weights_utility is not None,
            st.session_state.weights_tan is not None,
            st.session_state.weights_min is not None,
        ])
        if any_computed:
            st.markdown("---")
            st.markdown('<div class="section-heading">Portfolio Comparison</div>', unsafe_allow_html=True)

            comp_data = {}
            if st.session_state.weights_utility is not None:
                w = st.session_state.weights_utility
                ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
                comp_data["Utility"] = {
                    "Return %": round(ann_ret * 100, 2),
                    "Risk %":   round(ann_risk * 100, 2),
                    "Sharpe":   round(sharpe, 3),
                }
            if st.session_state.weights_tan is not None:
                comp_data["Tangency"] = {
                    "Return %": round(st.session_state.tan_return * 100, 2),
                    "Risk %":   round(st.session_state.tan_risk   * 100, 2),
                    "Sharpe":   round(st.session_state.tan_sharpe, 3),
                }
            if st.session_state.weights_min is not None:
                w = st.session_state.weights_min
                ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
                comp_data["Min-Risk"] = {
                    "Return %": round(ann_ret * 100, 2),
                    "Risk %":   round(ann_risk * 100, 2),
                    "Sharpe":   round(sharpe, 3),
                }

            if comp_data:
                comp_df = pd.DataFrame(comp_data).T
                render_insight_note("Use the cards first for ranking intuition, then the table and chart for exact deltas. In most sessions, Tangency is the benchmark portfolio to beat.")
                render_comparison_spotlights(comp_df)
                st.dataframe(comp_df.style
                    .background_gradient(cmap="Greens", subset=["Return %"])
                    .background_gradient(cmap="Reds_r", subset=["Risk %"])
                    .background_gradient(cmap="Blues",  subset=["Sharpe"]),
                    use_container_width=True)

                # Bar chart comparison
                fig_bar = go.Figure()
                metrics_list = ["Return %", "Risk %", "Sharpe"]
                colors = ["#00b4ff", "#ff5252", "#ffd54f"]
                for metric, color in zip(metrics_list, colors):
                    fig_bar.add_trace(go.Bar(
                        name=metric,
                        x=list(comp_data.keys()),
                        y=[comp_data[p][metric] for p in comp_data],
                        marker_color=color,
                        text=[f"{comp_data[p][metric]}" for p in comp_data],
                        textposition='outside',
                        textfont=dict(size=9),
                    ))
                fig_bar.update_layout(
                    barmode='group',
                    template="plotly_dark",
                    paper_bgcolor="#0a0e17",
                    plot_bgcolor="#0d1525",
                    title=dict(text="Portfolio Metrics Comparison", font=dict(color="#a0c8e8")),
                    font=dict(family="DM Mono", color="#80b0d0"),
                    legend=dict(font=dict(size=9)),
                    height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════
with tab_frontier:
    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data first.</div>', unsafe_allow_html=True)
    elif not st.session_state.frontier_computed:
        st.markdown('<div class="status-box status-warn">Compute the efficient frontier first (Build Portfolio tab).</div>', unsafe_allow_html=True)
    else:
        fr  = st.session_state.frontier_risks
        ret = st.session_state.frontier_returns
        ar  = st.session_state.asset_risks
        are = st.session_state.asset_returns
        tickers = st.session_state.tickers

        # Tangency data
        tan_risk   = st.session_state.tan_risk
        tan_return = st.session_state.tan_return
        tan_sharpe = st.session_state.tan_sharpe

        # ── Interactive Plotly (2D) ────────────────────────────
        st.markdown('<div class="section-heading">Interactive Efficient Frontier</div>', unsafe_allow_html=True)

        fr_pct   = [r * 100 for r in ret]
        fsk_pct  = [r * 100 for r in fr]
        ar_pct   = np.array([are[t] * 100 for t in tickers])
        ask_pct  = np.array([ar[t]  * 100 for t in tickers])

        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(
            x=fsk_pct, y=fr_pct, mode='lines', name='Efficient Frontier',
            line=dict(color="#00b4ff", width=3),
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        fig2d.add_trace(go.Scatter(
            x=ask_pct, y=ar_pct, mode='markers+text', text=tickers,
            textposition="top center", textfont=dict(size=9, color="#80c8ff"),
            marker=dict(size=10, color="#ff5252", line=dict(width=1, color="#ff0000")),
            name='Assets',
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        if tan_risk and tan_return and tan_sharpe:
            cml_x = np.linspace(0, max(fsk_pct) * 1.3, 100)
            cml_y = RF_ANNUAL * 100 + tan_sharpe * cml_x
            fig2d.add_trace(go.Scatter(
                x=cml_x, y=cml_y, mode='lines', name='Capital Market Line',
                line=dict(color='#00e676', dash='dash', width=2),
                hovertemplate="Risk: %{x:.2f}%<br>CML: %{y:.2f}%<extra></extra>"))
            fig2d.add_trace(go.Scatter(
                x=[tan_risk * 100], y=[tan_return * 100], mode='markers',
                marker=dict(size=18, color='gold', symbol='star',
                            line=dict(width=1, color='#aa8800')),
                name='Tangency Portfolio',
                hovertemplate=(f"<b>Tangency</b><br>Sharpe: {tan_sharpe:.3f}<br>"
                               "Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>")))

        # Slider
        steps_2d = []
        for i in range(len(fr_pct)):
            step_data = {
                "x": [fsk_pct, ask_pct.tolist()],
                "y": [fr_pct,  ar_pct.tolist()],
            }
            if tan_risk:
                step_data["x"] += [cml_x.tolist(), [tan_risk * 100], [fsk_pct[i]]]
                step_data["y"] += [cml_y.tolist(), [tan_return * 100], [fr_pct[i]]]
            steps_2d.append(dict(
                method="update",
                args=[step_data],
                label=f"{fr_pct[i]:.1f}%"
            ))

        # Selected point trace
        fig2d.add_trace(go.Scatter(
            x=[fsk_pct[0]], y=[fr_pct[0]], mode='markers',
            marker=dict(size=14, color='white', symbol='circle',
                        line=dict(color='#00b4ff', width=2)),
            name='Selected Point',
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"))

        fig2d.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Target Return: ", "font": {"color": "#80b0d0"}},
                pad={"t": 50}, steps=steps_2d, transition={"duration": 0},
                bgcolor="#0d1525", bordercolor="#1a2d4d",
                font=dict(color="#4a6a90", size=8)
            )],
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#0d1525",
            title=dict(text="Efficient Frontier · Tangency Portfolio · Capital Market Line",
                       font=dict(color="#a0c8e8", size=13)),
            xaxis_title="Annual Risk (σ) %",
            yaxis_title="Annual Return (μ) %",
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
            height=600,
            margin=dict(l=50, r=30, t=60, b=80),
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # ── 3D Plot ────────────────────────────────────────────
        if tan_sharpe:
            st.markdown('<div class="section-heading">3D Frontier — Risk / Return / Sharpe</div>', unsafe_allow_html=True)
            frontier_sharpes = [(r - RF_ANNUAL*100) / risk for r, risk in zip(fr_pct, fsk_pct)]

            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(
                x=fsk_pct, y=fr_pct, z=frontier_sharpes,
                mode='lines+markers',
                marker=dict(size=3.5, color=frontier_sharpes, colorscale='Plasma',
                            colorbar=dict(title="Sharpe", tickfont=dict(color="#80b0d0"))),
                line=dict(width=5, color='royalblue'),
                name="Frontier",
                hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.3f}<extra></extra>"))
            fig3d.add_trace(go.Scatter3d(
                x=[tan_risk * 100], y=[tan_return * 100], z=[tan_sharpe],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond',
                            line=dict(color='#aa8800', width=1)),
                name='Tangency Portfolio'))
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(title="Risk (%)", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    yaxis=dict(title="Return (%)", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    zaxis=dict(title="Sharpe", backgroundcolor="#0d1525",
                               gridcolor="#1a2d4d", color="#80b0d0"),
                    bgcolor="#0a0e17",
                ),
                paper_bgcolor="#0a0e17",
                title=dict(text="3D Efficient Frontier — Risk / Return / Sharpe",
                           font=dict(color="#a0c8e8")),
                font=dict(family="DM Mono", color="#80b0d0"),
                legend=dict(font=dict(size=9)),
                height=560,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # ── Static Matplotlib frontier ─────────────────────────
        st.markdown('<div class="section-heading">Static Frontier Plot</div>', unsafe_allow_html=True)
        fig_s, ax_s = plt.subplots(figsize=(10, 6))
        fig_s.patch.set_facecolor("#0a0e17")
        ax_s.set_facecolor("#0d1525")

        ax_s.plot(fsk_pct, fr_pct, color="#00b4ff", linewidth=2.5, label="Efficient Frontier")
        ax_s.scatter(ask_pct, ar_pct, color="#ff5252", s=80, zorder=5, label="Assets")

        if tan_risk and tan_sharpe:
            cml_x2 = np.linspace(0, max(fsk_pct) * 1.2, 100)
            cml_y2 = RF_ANNUAL * 100 + tan_sharpe * cml_x2
            ax_s.plot(cml_x2, cml_y2, linestyle="--", color="#00e676", linewidth=1.8, label="CML")
            ax_s.scatter([tan_risk * 100], [tan_return * 100],
                         marker="*", s=350, color="gold", zorder=6, label="Tangency")

        for t in tickers:
            ax_s.annotate(t, (ar[t] * 100, are[t] * 100),
                          xytext=(5, 5), textcoords="offset points",
                          fontsize=7, color="#80c8ff")

        ax_s.set_xlabel("Annual Risk (σ) %", color="#80b0d0")
        ax_s.set_ylabel("Annual Return (μ) %", color="#80b0d0")
        ax_s.set_title("Efficient Frontier & CML", color="#a0c8e8", fontsize=12)
        ax_s.tick_params(colors="#4a6a90")
        ax_s.legend(fontsize=9, facecolor="#0d1525", edgecolor="#1a2d4d",
                    labelcolor="#80b0d0")
        ax_s.grid(alpha=0.15, color="#1a2d4d")
        for spine in ax_s.spines.values():
            spine.set_edgecolor("#1a2d4d")
        plt.tight_layout()
        st.pyplot(fig_s)


# ══════════════════════════════════════════════════════════════
# TAB 5 — ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_analysis:
    if not st.session_state.data_loaded:
        render_insight_note("Analysis turns the optimizer inputs into a story: volatility regimes, wealth paths, and covariance structure. Load data first to unlock it.")
    else:
        returns = st.session_state.returns
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
        cov     = st.session_state.cov_matrix
        ar      = st.session_state.asset_risks
        are     = st.session_state.asset_returns

        # ── Rolling correlation ────────────────────────────────
        st.markdown('<div class="section-heading">Rolling Volatility (60-day)</div>', unsafe_allow_html=True)
        render_insight_note("Use this to see regime changes. If one asset's volatility suddenly spikes, it can dominate covariance and pull the optimizer away from it.")
        roll_vol = returns.rolling(60).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        for i, t in enumerate(tickers):
            hue = int(i * 360 / len(tickers))
            fig_vol.add_trace(go.Scatter(
                x=roll_vol.index, y=roll_vol[t],
                mode='lines', name=t,
                line=dict(width=1.5, color=f"hsl({hue},65%,55%)"),
                hovertemplate=f"<b>{t}</b><br>%{{x|%Y-%m-%d}}<br>Volatility: %{{y:.2f}}%<extra></extra>"
            ))
        fig_vol.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
            xaxis_title="Date", yaxis_title="Annualised Volatility %",
            title=dict(text="60-Day Rolling Annualised Volatility", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)"),
            height=420, margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Cumulative returns ─────────────────────────────────
        st.markdown('<div class="section-heading">Cumulative Returns</div>', unsafe_allow_html=True)
        render_insight_note("This is the path each asset actually took over your sample window. It helps separate smooth compounders from names that only look good in average-return form.")
        cum_ret = (1 + returns).cumprod() - 1
        fig_cum = go.Figure()
        for i, t in enumerate(tickers):
            hue = int(i * 360 / len(tickers))
            fig_cum.add_trace(go.Scatter(
                x=cum_ret.index, y=cum_ret[t] * 100,
                mode='lines', name=t,
                line=dict(width=1.5, color=f"hsl({hue},65%,55%)"),
                hovertemplate=f"<b>{t}</b><br>%{{x|%Y-%m-%d}}<br>Cumulative Return: %{{y:.2f}}%<extra></extra>"
            ))
        fig_cum.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
            xaxis_title="Date", yaxis_title="Cumulative Return %",
            title=dict(text="Cumulative Returns Over Period", font=dict(color="#a0c8e8")),
            font=dict(family="DM Mono", color="#80b0d0"),
            legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0.4)"),
            height=420, margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Portfolio cumulative return if computed ────────────
        if st.session_state.portfolios_computed or st.session_state.weights_tan is not None:
            st.markdown('<div class="section-heading">Portfolio Cumulative Returns</div>', unsafe_allow_html=True)
            render_insight_note("This is often the most decision-relevant chart in the app. If two portfolios have similar Sharpe but one compounds more smoothly, that's usually the one investors tolerate better.")
            fig_port = go.Figure()
            port_data = {}
            if st.session_state.weights_utility is not None:
                port_data["Utility"] = (st.session_state.weights_utility, "#00b4ff")
            if st.session_state.weights_tan is not None:
                port_data["Tangency"] = (st.session_state.weights_tan, "gold")
            if st.session_state.weights_min is not None:
                port_data["Min-Risk"] = (st.session_state.weights_min, "#00e676")

            for name, (w, color) in port_data.items():
                port_cum = cached_portfolio_cumulative_returns(returns[tickers], w)
                fig_port.add_trace(go.Scatter(
                    x=port_cum.index, y=port_cum.values * 100,
                    mode='lines', name=name,
                    line=dict(width=2, color=color),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"
                ))
            fig_port.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1525",
                xaxis_title="Date", yaxis_title="Cumulative Return %",
                title=dict(text="Optimized Portfolio Cumulative Returns", font=dict(color="#a0c8e8")),
                font=dict(family="DM Mono", color="#80b0d0"),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.4)"),
                height=420, margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_port, use_container_width=True)

        # ── Covariance heatmap (Plotly) ────────────────────────
        st.markdown('<div class="section-heading">Covariance Matrix (Annualised)</div>', unsafe_allow_html=True)
        render_insight_note("Covariance is the engine room of the optimizer. Large off-diagonal blocks mean the desk sees those assets as moving together, which reduces diversification benefit.")
        cov_annual = cov * 252
        fig_cov = plotly_heatmap(
            cov_annual, "Annualised Covariance Matrix",
            colorscale="YlOrBr", zmid=None, fmt=".4f",
            height=max(420, len(tickers) * 28 + 120)
        )
        st.plotly_chart(fig_cov, use_container_width=True)
        st.caption("Hover over any cell for full ticker names and exact value.  "
                   "Diagonal = each stock's own annualised variance.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — BACKTEST
# ══════════════════════════════════════════════════════════════
with tab_backtest:
    import datetime as _dt
    import yfinance as _yf2

    st.markdown('<div class="section-heading">Portfolio Backtest — Capital Simulation</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load market data and run at least one optimisation first.</div>', unsafe_allow_html=True)
    else:
        available_portfolios = {}
        if st.session_state.weights_utility is not None:
            available_portfolios["Utility Maximiser"] = st.session_state.weights_utility
        if st.session_state.weights_tan is not None:
            available_portfolios["Tangency (Max Sharpe)"] = st.session_state.weights_tan
        if st.session_state.weights_min is not None:
            available_portfolios["Min-Risk"] = st.session_state.weights_min

        if not available_portfolios:
            st.markdown('<div class="status-box status-warn">Run at least one portfolio optimisation before backtesting.</div>', unsafe_allow_html=True)
        else:
            col_cfg, col_res = st.columns([1, 2])

            with col_cfg:
                st.markdown('<div class="card-title">Backtest Settings</div>', unsafe_allow_html=True)

                selected_portfolio = st.selectbox(
                    "Portfolio to backtest",
                    list(available_portfolios.keys()),
                    key="bt_portfolio_name"
                )
                weights_bt = available_portfolios[selected_portfolio]
                tickers_bt = st.session_state.tickers

                capital = st.number_input(
                    "Starting Capital (₹ or $)",
                    min_value=1000.0, max_value=1_000_000_000.0,
                    value=100_000.0, step=1000.0, format="%.2f",
                    key="bt_capital"
                )

                st.markdown("**Backtest period**")
                use_session_dates = st.checkbox(
                    "Use the same dates as data window", value=True, key="bt_use_session"
                )
                if use_session_dates:
                    bt_start = st.session_state.start_date
                    bt_end   = st.session_state.end_date
                    st.caption(f"▶ {bt_start}  →  {bt_end}")
                else:
                    bt_mode = st.radio(
                        "Date mode", ["Start / End", "Days back from today"],
                        horizontal=True, key="bt_date_mode"
                    )
                    if bt_mode == "Start / End":
                        bt_start = st.text_input("Backtest start", value=st.session_state.start_date, key="bt_s")
                        bt_end   = st.text_input("Backtest end",   value=st.session_state.end_date,   key="bt_e")
                    else:
                        bt_days  = st.number_input("Days back", min_value=30, max_value=7300,
                                                    value=365*3, step=30, key="bt_days")
                        bt_end   = _dt.date.today().strftime("%Y-%m-%d")
                        bt_start = (_dt.date.today() - _dt.timedelta(days=int(bt_days))).strftime("%Y-%m-%d")
                        st.caption(f"▶ {bt_start}  →  {bt_end}")

                allow_fractional = st.checkbox(
                    "Allow fractional shares", value=False, key="bt_frac",
                    help="If OFF, floor to whole shares and leave residual as cash."
                )

                # ── Entry mode ─────────────────────────────────────
                st.markdown("---")
                st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Entry Mode</div>', unsafe_allow_html=True)
                entry_mode = st.radio(
                    "entry_mode_radio",
                    [
                        "📅 Common start — all stocks bought on the same day",
                        "📈 Staggered — each stock bought on its own first listed day",
                    ],
                    label_visibility="collapsed",
                    key="bt_entry_mode"
                )
                if entry_mode.startswith("📅"):
                    st.caption(
                        "All stocks are bought on the **latest** first-available date among "
                        "the portfolio (so every stock has a price on day 1). "
                        "Returns are directly comparable across the same window."
                    )
                else:
                    st.caption(
                        "Each stock is bought on its own first trading day in the window. "
                        "Capital for that stock sits as uninvested cash until then. "
                        "Per-stock returns cover different periods — portfolio-level CAGR "
                        "is measured from the earliest buy to the latest sell."
                    )

                # ── Pre-flight data availability panel ─────────────
                st.markdown("---")
                st.markdown('<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Ticker Data Availability</div>', unsafe_allow_html=True)
                try:
                    chosen_start = _dt.date.fromisoformat(bt_start)
                    chosen_end   = _dt.date.fromisoformat(bt_end)
                except Exception:
                    chosen_start = chosen_end = None

                for t in tickers_bt:
                    first_date, last_date, colour, age = get_ticker_date_range(t)
                    warn_start = warn_end = False
                    try:
                        if first_date != "N/A" and chosen_start:
                            warn_start = chosen_start < _dt.date.fromisoformat(first_date)
                        if last_date != "N/A" and chosen_end:
                            warn_end   = chosen_end > _dt.date.fromisoformat(last_date)
                    except Exception:
                        pass
                    warn_html = ""
                    if warn_start:
                        warn_html += f'<br><span style="font-size:0.63rem;color:#ff5252">⚠ data starts {first_date} — entry shifts to first available day</span>'
                    if warn_end:
                        warn_html += f'<br><span style="font-size:0.63rem;color:#ffd54f">⚠ last data {last_date} — exit will use that date</span>'
                    st.markdown(
                        f'<div style="margin-bottom:0.45rem">'
                        f'<span class="ticker-chip">{t}</span> '
                        f'<span style="font-size:0.68rem;color:#4a90d9">from </span>'
                        f'<span style="font-size:0.68rem;color:#a0c8e8">{first_date}</span>'
                        f'<span style="font-size:0.68rem;color:#4a90d9"> → </span>'
                        f'<span style="font-size:0.68rem;color:{colour}">{last_date}</span>'
                        + warn_html + '</div>',
                        unsafe_allow_html=True
                    )

                run_bt = st.button("▶ Run Backtest", use_container_width=True, key="run_bt_btn")

            # ── Backtest engine ────────────────────────────────────
            with col_res:
                if run_bt:
                    with st.spinner("Downloading price data for backtest window…"):
                        try:
                            bt_data = _yf2.download(
                                tickers_bt, start=bt_start, end=bt_end,
                                auto_adjust=True, progress=False
                            )
                            bt_prices = bt_data["Close"]
                            if isinstance(bt_prices, pd.Series):
                                bt_prices = bt_prices.to_frame(name=tickers_bt[0])

                            present = [t for t in tickers_bt if t in bt_prices.columns]
                            bt_prices = bt_prices[present]

                            # Drop rows where ALL tickers are NaN but keep partial rows —
                            # each ticker gets its own first/last valid date below.
                            bt_prices = bt_prices.dropna(how="all")

                            if bt_prices.empty or len(bt_prices) < 2:
                                st.error("No price data returned for the backtest window.")
                            else:
                                # ── Step 1: find each ticker's own first & last valid date ──
                                ticker_first_valid = {}   # first day with real price in window
                                ticker_last_valid  = {}   # last  day with real price in window
                                valid_tickers = []

                                for t in present:
                                    col = bt_prices[t].dropna()
                                    col = col[col > 0]
                                    if len(col) < 2:
                                        continue
                                    valid_tickers.append(t)
                                    ticker_first_valid[t] = col.index[0]
                                    ticker_last_valid[t]  = col.index[-1]

                                skipped = [t for t in present if t not in valid_tickers]
                                if skipped:
                                    st.warning(f"⚠ Skipped (no valid prices in window): {skipped}")

                                if not valid_tickers:
                                    st.error("No tickers had valid prices in the backtest window.")
                                else:
                                    use_common = entry_mode.startswith("📅")

                                    # ── Step 2: decide the common entry date if needed ─────
                                    if use_common:
                                        # Common start = latest first-available date across all tickers
                                        # This guarantees every stock has a real price on day 0
                                        common_entry_ts = max(ticker_first_valid[t] for t in valid_tickers)
                                        common_exit_ts  = min(ticker_last_valid[t]  for t in valid_tickers)

                                        if common_entry_ts >= common_exit_ts:
                                            st.error(
                                                "No overlapping trading window exists across all tickers. "
                                                "Try Staggered mode or remove tickers with very different listing dates."
                                            )
                                            st.stop()

                                        # Warn if common start is much later than requested window
                                        days_lost = (common_entry_ts.date() - _dt.date.fromisoformat(bt_start)).days
                                        if days_lost > 30:
                                            latest_t = max(valid_tickers, key=lambda t: ticker_first_valid[t])
                                            st.info(
                                                f"ℹ Common start is **{common_entry_ts.strftime('%Y-%m-%d')}** "
                                                f"because **{latest_t}** only has data from that date. "
                                                f"{days_lost} days of earlier history are excluded to keep all stocks on equal footing."
                                            )

                                        # ── Resolve actual prices at common dates ──
                                        # Use .asof() which finds the last valid value AT or BEFORE
                                        # the target date — immune to timezone drift and sparse indices.
                                        def safe_price(series, target_ts):
                                            """Return the closest real price at or before target_ts."""
                                            clean = series.dropna()
                                            clean = clean[clean > 0]
                                            if clean.empty:
                                                return float("nan")
                                            p = clean.asof(target_ts)
                                            if pd.isna(p):
                                                # target_ts is before first data point; take first real price
                                                p = float(clean.iloc[0])
                                            return float(p)

                                        entry_prices = {t: safe_price(bt_prices[t], common_entry_ts) for t in valid_tickers}
                                        exit_prices  = {t: safe_price(bt_prices[t], common_exit_ts)  for t in valid_tickers}

                                        # Resolve actual index labels closest to common dates
                                        def nearest_label(index, target_ts):
                                            pos = index.get_indexer([target_ts], method="nearest")[0]
                                            return index[pos]

                                        actual_entry_ts = nearest_label(bt_prices.index, common_entry_ts)
                                        actual_exit_ts  = nearest_label(bt_prices.index, common_exit_ts)

                                        entry_dates = {t: actual_entry_ts.strftime("%Y-%m-%d") for t in valid_tickers}
                                        exit_dates  = {t: actual_exit_ts.strftime("%Y-%m-%d")  for t in valid_tickers}
                                        overall_entry = actual_entry_ts.strftime("%Y-%m-%d")
                                        overall_exit  = actual_exit_ts.strftime("%Y-%m-%d")

                                        # Re-check: drop any ticker that still has NaN price after safe_price
                                        bad = [t for t in valid_tickers if pd.isna(entry_prices[t]) or pd.isna(exit_prices[t])]
                                        if bad:
                                            st.warning(f"⚠ Dropped tickers with no price on common dates: {bad}")
                                            valid_tickers = [t for t in valid_tickers if t not in bad]
                                            entry_prices  = {t: entry_prices[t] for t in valid_tickers}
                                            exit_prices   = {t: exit_prices[t]  for t in valid_tickers}
                                            entry_dates   = {t: entry_dates[t]  for t in valid_tickers}
                                            exit_dates    = {t: exit_dates[t]   for t in valid_tickers}
                                            if not valid_tickers:
                                                st.error("No tickers had valid prices on the common entry/exit dates.")
                                                st.stop()

                                    else:
                                        # Staggered: each stock enters on its own first day
                                        # Use safe_price / nearest_label for same robustness
                                        def safe_price(series, target_ts):
                                            clean = series.dropna()
                                            clean = clean[clean > 0]
                                            if clean.empty:
                                                return float("nan")
                                            p = clean.asof(target_ts)
                                            if pd.isna(p):
                                                p = float(clean.iloc[0])
                                            return float(p)

                                        entry_dates  = {t: ticker_first_valid[t].strftime("%Y-%m-%d") for t in valid_tickers}
                                        exit_dates   = {t: ticker_last_valid[t].strftime("%Y-%m-%d")  for t in valid_tickers}
                                        entry_prices = {t: safe_price(bt_prices[t], ticker_first_valid[t]) for t in valid_tickers}
                                        exit_prices  = {t: safe_price(bt_prices[t], ticker_last_valid[t])  for t in valid_tickers}

                                        bad = [t for t in valid_tickers if pd.isna(entry_prices[t]) or pd.isna(exit_prices[t])]
                                        if bad:
                                            st.warning(f"⚠ Dropped tickers with no resolvable price: {bad}")
                                            valid_tickers = [t for t in valid_tickers if t not in bad]
                                            entry_prices  = {t: entry_prices[t] for t in valid_tickers}
                                            exit_prices   = {t: exit_prices[t]  for t in valid_tickers}
                                            entry_dates   = {t: entry_dates[t]  for t in valid_tickers}
                                            exit_dates    = {t: exit_dates[t]   for t in valid_tickers}
                                            if not valid_tickers:
                                                st.error("No tickers had valid prices.")
                                                st.stop()

                                        all_entry_ts = sorted(ticker_first_valid[t] for t in valid_tickers)
                                        all_exit_ts  = sorted(ticker_last_valid[t]  for t in valid_tickers)
                                        overall_entry = all_entry_ts[0].strftime("%Y-%m-%d")
                                        overall_exit  = all_exit_ts[-1].strftime("%Y-%m-%d")

                                    valid_weights_raw = np.array([weights_bt[tickers_bt.index(t)] for t in valid_tickers])
                                    valid_weights     = valid_weights_raw / valid_weights_raw.sum()

                                    # ── Step 3: allocate capital → shares ──────
                                    alloc = {t: capital * w for t, w in zip(valid_tickers, valid_weights)}
                                    if allow_fractional:
                                        shares   = {t: alloc[t] / entry_prices[t] for t in valid_tickers}
                                        residual = 0.0
                                    else:
                                        shares   = {t: float(int(alloc[t] / entry_prices[t])) for t in valid_tickers}
                                        residual = sum(alloc[t] - shares[t] * entry_prices[t] for t in valid_tickers)

                                    # ── Step 4: exit values & portfolio totals ──
                                    exit_vals    = {t: shares[t] * exit_prices[t] for t in valid_tickers}
                                    total_exit   = sum(exit_vals.values()) + residual
                                    total_invest = capital
                                    abs_return   = total_exit - total_invest
                                    pct_return   = abs_return / total_invest * 100

                                    n_days  = (
                                        _dt.date.fromisoformat(overall_exit) -
                                        _dt.date.fromisoformat(overall_entry)
                                    ).days
                                    n_years = n_days / 365.25
                                    cagr    = ((total_exit / total_invest) ** (1 / max(n_years, 0.01)) - 1) * 100

                                    # ── Step 5: trade blotter ──────────────────
                                    rows = []
                                    for t in valid_tickers:
                                        ep  = entry_prices[t]
                                        xp  = exit_prices[t]
                                        sh  = shares[t]
                                        inv = sh * ep
                                        val = sh * xp
                                        g   = val - inv
                                        hold_days = (
                                            _dt.date.fromisoformat(exit_dates[t]) -
                                            _dt.date.fromisoformat(entry_dates[t])
                                        ).days
                                        rows.append({
                                            "Ticker":           t,
                                            "Weight %":         round(valid_weights[valid_tickers.index(t)] * 100, 2),
                                            "Buy Date":         entry_dates[t],
                                            "Buy Price":        round(ep, 2),
                                            "Shares Bought":    round(sh, 4),
                                            "Capital Deployed": round(inv, 2),
                                            "Sell Date":        exit_dates[t],
                                            "Sell Price":       round(xp, 2),
                                            "Current Value":    round(val, 2),
                                            "Gain / Loss":      round(g, 2),
                                            "Hold (days)":      hold_days,
                                            "Return %":         round((xp - ep) / ep * 100, 2),
                                        })
                                    bt_df = pd.DataFrame(rows)

                                    # ── Step 6: daily portfolio value curve ────
                                    # For common mode: slice to [common_entry, common_exit], ffill gaps
                                    # For staggered:   each ticker contributes from its own entry onward
                                    if use_common:
                                        window = bt_prices.loc[common_entry_ts:common_exit_ts].copy()
                                        window = window[valid_tickers].ffill()
                                        daily_vals = pd.Series(0.0, index=window.index)
                                        for t in valid_tickers:
                                            daily_vals += shares[t] * window[t]
                                        daily_vals += residual
                                    else:
                                        daily_vals = pd.Series(0.0, index=bt_prices.index)
                                        for t in valid_tickers:
                                            col_raw   = bt_prices[t].copy()
                                            e_idx     = bt_prices.index.get_loc(ticker_first_valid[t])
                                            col_slice = col_raw.iloc[e_idx:].ffill()
                                            daily_vals.iloc[e_idx:e_idx + len(col_slice)] += shares[t] * col_slice.values
                                        daily_vals = daily_vals.replace(0.0, np.nan).ffill().bfill()
                                        daily_vals += residual

                                    # ── Summary metrics ────────────────────────
                                    st.markdown(f"""
                                    <div class="metric-row">
                                        {metric_card("Capital Invested", f"₹{total_invest:,.0f}", "accent")}
                                        {metric_card("Portfolio Value",  f"₹{total_exit:,.0f}",   "positive" if pct_return>0 else "")}
                                        {metric_card("Absolute Return",  f"₹{abs_return:+,.0f}",  "positive" if abs_return>0 else "")}
                                        {metric_card("Total Return",     f"{pct_return:+.2f}%",   "positive" if pct_return>0 else "")}
                                        {metric_card("CAGR",             f"{cagr:+.2f}%",          "positive" if cagr>0 else "")}
                                        {metric_card("Period",           f"{n_days}d / {n_years:.1f}y", "accent")}
                                    </div>""", unsafe_allow_html=True)

                                    st.caption(
                                        f"Earliest buy: **{overall_entry}**  ·  Latest sell: **{overall_exit}**  ·  "
                                        f"{'Fractional' if allow_fractional else 'Whole'} shares  ·  "
                                        f"Uninvested cash: ₹{residual:,.2f}"
                                    )

                                    # ── Trade blotter table ────────────────────
                                    st.markdown('<div class="section-heading">Trade Blotter — Buy & Sell Detail</div>', unsafe_allow_html=True)
                                    st.dataframe(
                                        bt_df.style
                                            .background_gradient(cmap="RdYlGn", subset=["Return %", "Gain / Loss"])
                                            .format({
                                                "Capital Deployed": "₹{:,.2f}",
                                                "Current Value":    "₹{:,.2f}",
                                                "Gain / Loss":      "₹{:+,.2f}",
                                                "Buy Price":        "{:,.2f}",
                                                "Sell Price":       "{:,.2f}",
                                                "Shares Bought":    "{:.4f}",
                                                "Return %":         "{:+.2f}%",
                                            }),
                                        use_container_width=True,
                                        height=min(400, 60 + 35 * len(bt_df))
                                    )

                                    # ── Portfolio value over time chart ────────
                                    st.markdown('<div class="section-heading">Portfolio Value Over Time</div>', unsafe_allow_html=True)
                                    fig_bt = go.Figure()
                                    fig_bt.add_trace(go.Scatter(
                                        x=daily_vals.index, y=daily_vals.values,
                                        mode='lines', name="Portfolio Value",
                                        fill='tozeroy',
                                        line=dict(color="#00b4ff", width=2),
                                        fillcolor="rgba(0,180,255,0.07)",
                                        hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.2f}<extra></extra>"
                                    ))
                                    fig_bt.add_hline(
                                        y=capital, line_dash="dash",
                                        line_color="rgba(255,213,79,0.5)",
                                        annotation_text="Invested Capital",
                                        annotation_font_color="#ffd54f"
                                    )
                                    fig_bt.update_layout(
                                        template="plotly_dark", paper_bgcolor="#0a0e17",
                                        plot_bgcolor="#0d1525",
                                        xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                                        title=dict(text=f"Backtest: {selected_portfolio}", font=dict(color="#a0c8e8")),
                                        font=dict(family="DM Mono", color="#80b0d0"),
                                        height=400, margin=dict(l=50, r=20, t=50, b=50)
                                    )
                                    st.plotly_chart(fig_bt, use_container_width=True)

                                    # ── Per-stock return bar chart ─────────────
                                    st.markdown('<div class="section-heading">Per-Stock Return %</div>', unsafe_allow_html=True)
                                    bt_df_sorted = bt_df.sort_values("Return %", ascending=False)
                                    colors_bar   = ["#00e676" if r > 0 else "#ff5252" for r in bt_df_sorted["Return %"]]
                                    fig_bar = go.Figure(go.Bar(
                                        x=bt_df_sorted["Ticker"], y=bt_df_sorted["Return %"],
                                        marker_color=colors_bar,
                                        text=[f"{r:+.1f}%" for r in bt_df_sorted["Return %"]],
                                        textposition="outside",
                                        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>"
                                    ))
                                    fig_bar.update_layout(
                                        template="plotly_dark", paper_bgcolor="#0a0e17",
                                        plot_bgcolor="#0d1525",
                                        xaxis_title="", yaxis_title="Return %",
                                        font=dict(family="DM Mono", color="#80b0d0"),
                                        height=320, margin=dict(l=50, r=20, t=30, b=60),
                                        uniformtext_minsize=7, uniformtext_mode="hide"
                                    )
                                    fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
                                    st.plotly_chart(fig_bar, use_container_width=True)

                                    # ── Persist results for PDF export ───────
                                    st.session_state.bt_results = {
                                        "portfolio_name": selected_portfolio,
                                        "capital":        capital,
                                        "entry_mode":     "Common Start" if use_common else "Staggered",
                                        "overall_entry":  overall_entry,
                                        "overall_exit":   overall_exit,
                                        "n_days":         n_days,
                                        "n_years":        n_years,
                                        "total_invest":   total_invest,
                                        "total_exit":     total_exit,
                                        "abs_return":     abs_return,
                                        "pct_return":     pct_return,
                                        "cagr":           cagr,
                                        "residual":       residual,
                                        "allow_fractional": allow_fractional,
                                        "blotter":        bt_df,
                                    }

                                    bt_csv = bt_df.to_csv(index=False).encode()
                                    st.download_button(
                                        "⬇ Download Trade Blotter CSV", data=bt_csv,
                                        file_name="quansen_backtest.csv", mime="text/csv",
                                        use_container_width=True
                                    )

                        except Exception as e:
                            st.error(f"Backtest failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# TAB 7 — EXPORT
# ══════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-heading">Export Results</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown('<div class="status-box status-info">Load data and run optimizations to export results.</div>', unsafe_allow_html=True)
    else:
        # ── PDF Report ─────────────────────────────────────────
        st.markdown('<div class="card-title">📄 Full PDF Report</div>', unsafe_allow_html=True)
        has_bt = st.session_state.bt_results is not None
        has_opt = any([
            st.session_state.weights_utility is not None,
            st.session_state.weights_tan is not None,
            st.session_state.weights_min is not None,
        ])
        if not has_opt:
            st.markdown('<div class="status-box status-warn">Run at least one optimisation to generate the PDF.</div>', unsafe_allow_html=True)
        else:
            if not has_bt:
                st.caption("ℹ Run a backtest first to include the Trade Blotter section in the PDF.")
            if st.button("📄 Generate & Download Full PDF Report", use_container_width=True, key="pdf_btn"):
                with st.spinner("Building PDF…"):
                    try:
                        pdf_bytes = generate_pdf(st.session_state)
                        st.download_button(
                            label="⬇ Click here to download  quansen_report.pdf",
                            data=pdf_bytes,
                            file_name="quansen_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="pdf_download"
                        )
                        st.success("✔ PDF ready — click the button above to download.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        st.markdown("---")
        tickers = st.session_state.tickers
        er      = optimizer_expected_returns()
        cov     = st.session_state.cov_matrix

        # ── Weights CSV ────────────────────────────────────────
        st.markdown('<div class="card-title">Portfolio Weights</div>', unsafe_allow_html=True)

        export_rows = {"Ticker": tickers}
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            export_rows["Utility_Weight"] = w
            export_rows["Utility_Pct"]    = (w * 100).round(2)
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            export_rows["Tangency_Weight"] = w
            export_rows["Tangency_Pct"]    = (w * 100).round(2)
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            export_rows["MinRisk_Weight"] = w
            export_rows["MinRisk_Pct"]    = (w * 100).round(2)

        if len(export_rows) > 1:
            weights_df = pd.DataFrame(export_rows)
            st.dataframe(weights_df, use_container_width=True)
            csv_bytes = weights_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Weights CSV",
                data=csv_bytes,
                file_name="quansen_weights.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.markdown('<div class="status-box status-warn">Run at least one optimization to export weights.</div>', unsafe_allow_html=True)

        # ── Asset summary CSV ──────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">Asset Summary</div>', unsafe_allow_html=True)

        asset_risks = st.session_state.asset_risks
        asset_returns = st.session_state.asset_returns
        summary_exp_df = pd.DataFrame({
            "Ticker"        : tickers,
            "Ann_Return_Pct": (asset_returns.values * 100).round(4),
            "Ann_Risk_Pct"  : (asset_risks.values   * 100).round(4),
            "Sharpe"        : ((asset_returns.values - RF_ANNUAL) / asset_risks.values).round(4),
        })
        st.dataframe(summary_exp_df, use_container_width=True)
        st.download_button(
            "⬇ Download Asset Summary CSV",
            data=summary_exp_df.to_csv(index=False).encode(),
            file_name="quansen_asset_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ── Frontier CSV ───────────────────────────────────────
        if st.session_state.frontier_computed:
            st.markdown("---")
            st.markdown('<div class="card-title">Efficient Frontier Points</div>', unsafe_allow_html=True)
            frontier_df = pd.DataFrame({
                "Return_Pct": [r * 100 for r in st.session_state.frontier_returns],
                "Risk_Pct":   [r * 100 for r in st.session_state.frontier_risks],
            })
            frontier_df["Sharpe"] = ((frontier_df["Return_Pct"] - RF_ANNUAL * 100)
                                      / frontier_df["Risk_Pct"]).round(4)
            st.dataframe(frontier_df.head(20), use_container_width=True)
            st.download_button(
                "⬇ Download Frontier CSV",
                data=frontier_df.to_csv(index=False).encode(),
                file_name="quansen_frontier.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── Full stats summary ─────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">Optimizer Summary Report</div>', unsafe_allow_html=True)

        lines = [
            "QUANSEN PORTFOLIO OPTIMIZER — SUMMARY REPORT",
            "=" * 60,
            f"Tickers      : {', '.join(tickers)}",
            f"Date Range   : {st.session_state.start_date} to {st.session_state.end_date}",
            f"Risk-Free Rt : {RF_ANNUAL*100:.2f}%",
            f"Min Weight   : {st.session_state.min_w*100:.1f}%",
            f"Max Weight   : {st.session_state.max_w*100:.1f}%",
            "",
        ]
        if st.session_state.weights_utility is not None:
            w = st.session_state.weights_utility
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            lines += [
                "UTILITY-MAXIMIZED PORTFOLIO",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {sharpe:.3f}",
                "",
            ]
        if st.session_state.weights_tan is not None:
            w = st.session_state.weights_tan
            lines += [
                "TANGENCY PORTFOLIO (MAX SHARPE)",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {st.session_state.tan_return*100:.2f}%",
                f"  Annual Risk    : {st.session_state.tan_risk*100:.2f}%",
                f"  Sharpe Ratio   : {st.session_state.tan_sharpe:.3f}",
                "",
            ]
        if st.session_state.weights_min is not None:
            w = st.session_state.weights_min
            ann_ret, ann_risk, sharpe = cached_portfolio_stats(w, er, cov)
            lines += [
                f"MIN-RISK PORTFOLIO (Target: {st.session_state.target_return*100:.1f}%)",
                "-" * 40,
                *[f"  {t:<20} {w_:.4f}  ({w_*100:.2f}%)" for t, w_ in zip(tickers, w)],
                f"  Annual Return  : {ann_ret*100:.2f}%",
                f"  Annual Risk    : {ann_risk*100:.2f}%",
                f"  Sharpe Ratio   : {sharpe:.3f}",
                "",
            ]
        lines.append("=" * 60)
        report_text = "\n".join(lines)
        st.code(report_text, language=None)
        st.download_button(
            "⬇ Download Summary Report (.txt)",
            data=report_text.encode(),
            file_name="quansen_report.txt",
            mime="text/plain",
            use_container_width=True
        )
