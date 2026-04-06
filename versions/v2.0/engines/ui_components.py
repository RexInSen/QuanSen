"""
============================================================
  QuanSen — UI Components, Renderers & Global Styles
  Module 3 of 4: All reusable Streamlit UI rendering
  functions and the global CSS stylesheet.
============================================================
"""

import streamlit as st
import html
import pandas as pd
import numpy as np

from engines.config import RF_ANNUAL
from engines.config_and_state import (
    SECTOR_PRESETS, APP_VERSION,
    normalize_symbol_list, parse_uploaded_symbols,
    push_bridge_symbols, request_action,
)
from engines.data_and_compute import cached_fetch_tape_quotes


# ── Global CSS Stylesheet ─────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0e17;
    color: #dce6f0;
}

/* ── Main container ── */
.main .block-container {
    background: #0a0e17;
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1c2840;
}
[data-testid="stSidebar"][aria-expanded="true"],
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    min-width: 390px !important;
    max-width: 390px !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem 2rem;
}
[data-testid="stSidebar"] .stExpander {
    border: 1px solid #183155 !important;
    border-radius: 12px !important;
    background: linear-gradient(180deg, rgba(10,18,32,0.94), rgba(13,21,37,0.88)) !important;
}
[data-testid="stSidebar"] .stExpander summary {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.05em !important;
}
@media (max-width: 1200px) {
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        min-width: 340px !important;
        max-width: 340px !important;
    }
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a0e17 0%, #0d1a35 50%, #091429 100%);
    border: 1px solid #1c3a6e;
    border-radius: 6px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner,
.hero-command-deck,
.card {
    animation: fadeLift 0.45s ease both;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e8f4ff;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-title span { color: #00b4ff; }
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a6a90;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.25);
    color: #00b4ff;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-top: 0.8rem;
}
.hero-command-deck {
    display: block;
    margin: 0.2rem 0 1rem 0;
}
.command-panel {
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,14,26,0.9));
    border: 1px solid #17314d;
    border-radius: 14px;
    padding: 1rem 1.1rem;
}
.command-kicker {
    font-size: 0.64rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a6a90;
    margin-bottom: 0.35rem;
}
.command-headline {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #e9f6ff;
    margin-bottom: 0.25rem;
}
.command-copy {
    font-size: 0.78rem;
    color: #8db4d9;
    line-height: 1.45;
}
.command-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.9rem;
}
.command-cell {
    border: 1px solid #17314d;
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    background: linear-gradient(180deg, rgba(12,25,44,0.95), rgba(9,18,31,0.92));
}
.command-label {
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a6a90;
}
.command-value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: #ecf6ff;
    margin-top: 0.2rem;
}
.command-value.is-accent { color: #7dd3fc; }
.command-value.is-good { color: #00e676; }
.command-value.is-warn { color: #ffd54f; }
.hero-action-row {
    margin-top: 0.85rem;
}
@media (max-width: 1050px) {
    .command-grid {
        grid-template-columns: 1fr;
    }
}
.workflow-rail {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.15rem 0 1rem 0;
}
.workflow-step {
    border: 1px solid #17314d;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: linear-gradient(180deg, rgba(8,17,30,0.96), rgba(11,20,35,0.9));
}
.workflow-step.is-live {
    border-color: #00b4ff;
    box-shadow: 0 0 18px rgba(0,180,255,0.12);
}
.workflow-step.is-ready {
    border-color: rgba(0,230,118,0.45);
}
.workflow-kicker {
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a6a90;
}
.workflow-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    color: #e8f4ff;
    margin-top: 0.25rem;
}
.workflow-copy {
    font-size: 0.72rem;
    color: #86a9cb;
    margin-top: 0.25rem;
    line-height: 1.4;
}
.workflow-chip {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.22rem 0.48rem;
    border-radius: 999px;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: rgba(0,180,255,0.08);
    color: #8bd8ff;
    border: 1px solid rgba(0,180,255,0.22);
}
.workflow-summary {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.25rem 0 1rem 0;
}
.workflow-metric {
    border: 1px solid #17314d;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,15,27,0.9));
}
.workflow-metric-label {
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a6a90;
}
.workflow-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    color: #edf7ff;
    margin-top: 0.25rem;
}
.workflow-metric-copy {
    font-size: 0.7rem;
    color: #82a7ca;
    margin-top: 0.2rem;
}
.workflow-section-shell {
    border: 1px solid #153050;
    border-radius: 16px;
    padding: 1rem 1.1rem 1.05rem;
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,15,26,0.9));
    margin: 0.95rem 0 1rem 0;
}
.workflow-section-topline {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: baseline;
    margin-bottom: 0.8rem;
}
.workflow-section-kicker {
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a6a90;
}
.workflow-section-copy {
    font-size: 0.74rem;
    color: #85a9cb;
}
.build-action-rail {
    display: grid;
    grid-template-columns: 1.25fr 1fr 1fr 1fr;
    gap: 0.75rem;
    margin: 0.5rem 0 1rem 0;
}
.build-action-card {
    border: 1px solid #17314d;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,15,27,0.92));
}
.build-action-card .headline {
    font-family: 'Syne', sans-serif;
    color: #edf7ff;
    font-size: 0.96rem;
}
.build-action-card .copy {
    font-size: 0.72rem;
    color: #89abcc;
    line-height: 1.45;
    margin-top: 0.25rem;
}
.spotlight-shell {
    border: 1px solid #163250;
    border-radius: 18px;
    padding: 1rem 1.1rem 1.1rem;
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,15,27,0.9));
    margin: 0.7rem 0 1rem;
}
.spotlight-shell.is-winner {
    border-color: rgba(255, 213, 79, 0.45);
    box-shadow: 0 0 22px rgba(255, 213, 79, 0.08);
}
.spotlight-topline {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
    align-items: center;
    margin-bottom: 0.8rem;
}
.spotlight-title {
    font-family: 'Syne', sans-serif;
    color: #edf7ff;
    font-size: 1.08rem;
}
.spotlight-tag {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 999px;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid rgba(0,180,255,0.24);
    color: #8bd8ff;
    background: rgba(0,180,255,0.08);
}
.spotlight-tag.is-gold {
    color: #ffe082;
    border-color: rgba(255,213,79,0.28);
    background: rgba(255,213,79,0.08);
}
.spotlight-copy {
    font-size: 0.75rem;
    color: #85a9cb;
    line-height: 1.45;
    margin-bottom: 0.8rem;
}
.spotlight-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 0.85rem;
}
.spotlight-metric {
    border: 1px solid #17314d;
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    background: linear-gradient(180deg, rgba(12,25,44,0.95), rgba(9,18,31,0.92));
}
.spotlight-metric .label {
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a6a90;
    margin-bottom: 0.3rem;
}
.spotlight-metric .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    color: #edf7ff;
}
.spotlight-metric .value.good { color: #00e676; }
.spotlight-metric .value.warn { color: #ffd54f; }
.spotlight-metric .value.accent { color: #7dd3fc; }
.compare-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.6rem 0 1rem;
}
.compare-card {
    border: 1px solid #17314d;
    border-radius: 14px;
    padding: 0.95rem 1rem;
    background: linear-gradient(180deg, rgba(10,18,33,0.98), rgba(8,15,27,0.9));
}
.compare-card.is-best {
    border-color: rgba(255,213,79,0.4);
    box-shadow: 0 0 20px rgba(255,213,79,0.07);
}
.compare-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: #edf7ff;
}
.compare-badge {
    display: inline-block;
    margin-top: 0.4rem;
    padding: 0.2rem 0.45rem;
    border-radius: 999px;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: rgba(255,213,79,0.08);
    color: #ffe082;
    border: 1px solid rgba(255,213,79,0.22);
}
.compare-row {
    margin-top: 0.7rem;
}
.compare-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5d80a2;
}
.compare-value {
    font-family: 'DM Mono', monospace;
    color: #d9ecff;
    margin-top: 0.18rem;
    font-size: 0.82rem;
}
.compare-bar {
    margin-top: 0.32rem;
    width: 100%;
    height: 7px;
    border-radius: 999px;
    background: #112033;
    overflow: hidden;
}
.compare-fill {
    height: 100%;
    border-radius: 999px;
}
.insight-note {
    border: 1px solid rgba(0,180,255,0.18);
    border-left: 4px solid #00b4ff;
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    background: rgba(0,180,255,0.05);
    color: #b4d6f3;
    font-size: 0.76rem;
    line-height: 1.5;
    margin: 0.45rem 0 0.9rem;
}
@media (max-width: 1100px) {
    .build-action-rail,
    .spotlight-metrics,
    .compare-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 1100px) {
    .workflow-rail,
    .workflow-summary {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 760px) {
    .build-action-rail,
    .spotlight-metrics,
    .compare-grid,
    .workflow-rail,
    .workflow-summary {
        grid-template-columns: 1fr;
    }
}

/* ── Cards ── */
.card {
    background: #0d1525;
    border: 1px solid #1a2d4d;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00b4ff;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1a2d4d;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1;
    min-width: 130px;
    background: #0a1428;
    border: 1px solid #1a2d4d;
    border-radius: 5px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6a90;
    margin-bottom: 0.4rem;
}
.metric-box .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1;
}
.metric-box .value.positive { color: #00e676; }
.metric-box .value.accent   { color: #00b4ff; }
.metric-box .value.gold     { color: #ffd54f; }

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #a8c8e8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-left: 3px solid #00b4ff;
    padding-left: 0.7rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Tables ── */
.quansen-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.quansen-table th {
    background: #0d1a35;
    color: #4a90d9;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.6rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid #1a2d4d;
}
.quansen-table td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #111e33;
    color: #c8ddf0;
    font-family: 'DM Mono', monospace;
}
.quansen-table tr:last-child td { border-bottom: none; }
.quansen-table tr:hover td { background: rgba(0,180,255,0.03); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0044aa) !important;
    color: #e8f4ff !important;
    border: 1px solid #0055bb !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 0.95rem !important;
    line-height: 1.15 !important;
    min-height: 2.7rem !important;
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: anywhere !important;
    text-align: center !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077ee, #0055cc) !important;
    border-color: #0077ee !important;
    box-shadow: 0 0 16px rgba(0,120,255,0.25) !important;
}
.hero-action-row .stButton > button {
    min-height: 3.1rem !important;
    font-size: 0.72rem !important;
    border-radius: 10px !important;
    padding: 0.7rem 0.9rem !important;
}
.builder-helper-link {
    display: block;
    text-decoration: none;
    border: 1px solid #18406a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    background: linear-gradient(135deg, rgba(8,24,45,0.98), rgba(9,17,31,0.92));
    color: #d9ecff !important;
    margin: 0.45rem 0 0.7rem;
}
.builder-helper-link:hover {
    border-color: #00b4ff;
    box-shadow: 0 0 16px rgba(0,180,255,0.14);
}
.builder-helper-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    color: #e8f4ff;
}
.builder-helper-copy {
    font-size: 0.73rem;
    color: #84a9cc;
    margin-top: 0.2rem;
}

/* ── Input fields ── */
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
    background: #0a1428 !important;
    color: #c8ddf0 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 2px rgba(0,102,204,0.2) !important;
}

.stTextInput [data-testid="stWidgetLabel"],
.stTextArea [data-testid="stWidgetLabel"] {
    color: #8fb8de !important;
}

div[data-baseweb="input"] input::placeholder,
textarea::placeholder {
    color: #5f7fa1 !important;
    opacity: 1 !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: #0a1428 !important;
    border: 1px solid #1a3050 !important;
    color: #c8ddf0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Slider ── */
.stSlider > div { padding: 0.2rem 0; }

/* ── Status/Alert boxes ── */
.status-box {
    border-radius: 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.82rem;
    border-left: 3px solid;
}
.status-success { background: rgba(0,230,118,0.06); border-color: #00e676; color: #80ffc0; }
.status-info    { background: rgba(0,180,255,0.06); border-color: #00b4ff; color: #80d8ff; }
.status-warn    { background: rgba(255,213,79,0.06); border-color: #ffd54f; color: #ffe082; }
.status-error   { background: rgba(255,82,82,0.06);  border-color: #ff5252; color: #ff8a80; }

/* ── Ticker chip ── */
.ticker-chip {
    display: inline-block;
    background: rgba(0,102,204,0.15);
    border: 1px solid rgba(0,102,204,0.4);
    color: #80c8ff;
    border-radius: 3px;
    padding: 2px 9px;
    font-size: 0.75rem;
    margin: 2px 3px;
    font-family: 'DM Mono', monospace;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #4a90d9 !important;
    background: #0d1525 !important;
    border: 1px solid #1a2d4d !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #1a2d4d !important; margin: 1.5rem 0 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: #00b4ff !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1525;
    border-bottom: 1px solid #1a2d4d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6a90 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.2rem !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00b4ff !important;
    border-bottom-color: #00b4ff !important;
}

/* ── DataFrame ── */
.dataframe { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }

/* ── Weight bar fill ── */
.weight-bar-bg {
    background: #111e33;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.weight-bar-fill {
    height: 6px;
    border-radius: 2px;
    background: linear-gradient(90deg, #0066cc, #00b4ff);
}

/* ── Logo pulse animation ── */
@keyframes logoPulse {
    0%,100% { box-shadow: 0 0 18px rgba(0,180,255,0.35); }
    50%      { box-shadow: 0 0 34px rgba(0,180,255,0.70), 0 0 60px rgba(0,180,255,0.20); }
}

/* ── Tab fade-in transition ── */
@keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeLift {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stTabsContent"] > div[role="tabpanel"] {
    animation: tabFadeIn 0.32s ease forwards;
}

/* ── Ticker tape container ── */
.qs-tape-wrap {
    overflow: hidden;
    background: linear-gradient(90deg,#050810 0%,#0a0e17 8%,#0a0e17 92%,#050810 100%);
    border-top: 1px solid #1a2d4d;
    border-bottom: 1px solid #1a2d4d;
    padding: 5px 0;
    margin-bottom: 0.6rem;
    white-space: nowrap;
}
.qs-tape-inner {
    display: inline-flex;
    gap: 0;
    animation: tapescroll linear infinite;
}
@keyframes tapescroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.qs-tick {
    display: inline-block;
    padding: 0 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    border-right: 1px solid #1a2d4d;
}
.qs-tick-sym  { color: #a0c8e8; font-weight: 600; }
.qs-tick-price{ color: #e8f4ff; margin-left: 0.4rem; }
.qs-tick-up   { color: #00e676; margin-left: 0.3rem; }
.qs-tick-dn   { color: #ff5252; margin-left: 0.3rem; }
.qs-tick-neu  { color: #80b0d0; margin-left: 0.3rem; }
.qs-tape-ts   {
    font-size: 0.6rem; color: #2a4060;
    text-align: right; padding-right: 0.5rem;
    font-family: 'DM Mono', monospace;
}

/* ══════════════════════════════════════════════════════════════
   MOBILE-RESPONSIVE OVERRIDES
   These media queries ONLY fire on small screens.
   Desktop layout is completely unaffected.
   ══════════════════════════════════════════════════════════════ */

/* ── Tablet & small laptop (≤768px) ── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 1rem !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        min-width: 280px !important;
        max-width: 85vw !important;
    }
    .hero-banner {
        padding: 1.4rem 1.2rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.8rem;
    }
    .hero-subtitle {
        font-size: 0.7rem;
        letter-spacing: 0.08em;
    }
    .command-grid {
        grid-template-columns: 1fr;
    }
    .build-action-rail {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.65rem !important;
        padding: 0.5rem 0.6rem !important;
        letter-spacing: 0.04em !important;
    }
    .quansen-table {
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .metric-box .value {
        font-size: 1.3rem;
    }
}

/* ── Phone (≤480px) ── */
@media (max-width: 480px) {
    .main .block-container {
        padding: 0.6rem 0.5rem !important;
    }
    /* Hero banner — full visibility on small phones */
    .hero-banner {
        padding: 1rem 0.8rem 0.9rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
    .hero-banner::before {
        display: none;  /* hide decorative glow on mobile */
    }
    .hero-title {
        font-size: 1.4rem;
        line-height: 1.15;
        word-break: break-word;
    }
    .hero-subtitle {
        font-size: 0.62rem;
        letter-spacing: 0.06em;
    }
    .hero-badge {
        font-size: 0.58rem;
        padding: 2px 7px;
        margin-top: 0.5rem;
    }

    /* Command deck — stack everything */
    .hero-command-deck {
        margin: 0.1rem 0 0.6rem 0;
    }
    .command-panel {
        padding: 0.7rem 0.75rem;
        border-radius: 10px;
    }
    .command-headline {
        font-size: 0.9rem;
    }
    .command-copy {
        font-size: 0.7rem;
    }
    .command-grid {
        grid-template-columns: 1fr;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .command-cell {
        padding: 0.55rem 0.7rem;
        border-radius: 8px;
    }
    .command-label {
        font-size: 0.52rem;
    }
    .command-value {
        font-size: 0.88rem;
    }

    /* Workflow rail & summary — single column */
    .workflow-rail,
    .workflow-summary {
        grid-template-columns: 1fr !important;
        gap: 0.5rem;
    }
    .workflow-step,
    .workflow-metric {
        padding: 0.7rem 0.8rem;
    }
    .workflow-title {
        font-size: 0.82rem;
    }
    .workflow-copy {
        font-size: 0.66rem;
    }
    .workflow-metric-value {
        font-size: 1rem;
    }

    /* Build action rail — single column */
    .build-action-rail {
        grid-template-columns: 1fr !important;
        gap: 0.5rem;
    }

    /* Spotlights — stack metrics */
    .spotlight-shell {
        padding: 0.75rem 0.8rem;
        border-radius: 12px;
    }
    .spotlight-title {
        font-size: 0.92rem;
    }
    .spotlight-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.5rem;
    }
    .spotlight-metric {
        padding: 0.6rem 0.7rem;
    }
    .spotlight-metric .value {
        font-size: 0.9rem;
    }

    /* Compare grid — single column */
    .compare-grid {
        grid-template-columns: 1fr !important;
        gap: 0.6rem;
    }

    /* Metric boxes — wrap into 2 columns */
    .metric-row {
        gap: 0.5rem;
    }
    .metric-box {
        min-width: 100px;
        padding: 0.7rem 0.8rem;
    }
    .metric-box .value {
        font-size: 1.15rem;
    }
    .metric-box .label {
        font-size: 0.58rem;
    }

    /* Session strip pills — wrap tighter */
    .session-strip-wrap {
        gap: 0.4rem;
    }

    /* Tables — horizontal scroll */
    .quansen-table {
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        font-size: 0.72rem;
    }
    .quansen-table th {
        padding: 0.45rem 0.6rem;
        font-size: 0.58rem;
        white-space: nowrap;
    }
    .quansen-table td {
        padding: 0.4rem 0.6rem;
        white-space: nowrap;
    }

    /* Tabs — smaller text, allow horizontal scroll */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        flex-wrap: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.58rem !important;
        padding: 0.45rem 0.5rem !important;
        letter-spacing: 0.02em !important;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* Buttons — touch-friendly */
    .stButton > button {
        min-height: 2.8rem !important;
        font-size: 0.7rem !important;
        padding: 0.5rem 0.7rem !important;
    }
    .hero-action-row .stButton > button {
        min-height: 2.8rem !important;
        font-size: 0.65rem !important;
        padding: 0.55rem 0.6rem !important;
        border-radius: 8px !important;
    }

    /* Cards — tighter padding */
    .card {
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 0.65rem;
        margin-bottom: 0.7rem;
    }

    /* Section headings */
    .section-heading {
        font-size: 0.85rem;
        margin: 1.2rem 0 0.7rem 0;
    }

    /* Insight note */
    .insight-note {
        font-size: 0.7rem;
        padding: 0.65rem 0.75rem;
    }

    /* Ticker tape — smaller text */
    .qs-tick {
        padding: 0 0.8rem;
        font-size: 0.62rem;
    }

    /* Status boxes */
    .status-box {
        font-size: 0.74rem;
        padding: 0.6rem 0.8rem;
    }

    /* Sidebar expanders */
    [data-testid="stSidebar"] .block-container {
        padding: 1rem 0.8rem 1.5rem;
    }
    [data-testid="stSidebar"] .stExpander {
        border-radius: 8px !important;
    }

    /* Builder helper link */
    .builder-helper-link {
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
    }
    .builder-helper-title {
        font-size: 0.78rem;
    }
    .builder-helper-copy {
        font-size: 0.66rem;
    }

    /* Workflow section shell */
    .workflow-section-shell {
        padding: 0.75rem 0.8rem;
        border-radius: 12px;
    }
    .workflow-section-topline {
        flex-direction: column;
        gap: 0.3rem;
    }
}
</style>
"""



# ── Status & Info Panels ──────────────────────────────────────
def render_perf_status_panel():
    data_loaded = bool(st.session_state.get("data_loaded"))
    momentum_enabled = bool(st.session_state.get("momentum_enabled"))
    momentum_ready = st.session_state.get("momentum_final_er") is not None
    utility_ready = st.session_state.get("weights_utility") is not None
    tangency_ready = st.session_state.get("weights_tan") is not None
    frontier_ready = bool(st.session_state.get("frontier_computed"))
    tape_count = len(st.session_state.get("tape_indices", []))

    def _badge(ok, on_text, off_text):
        color = "#00e676" if ok else "#ffb300"
        label = on_text if ok else off_text
        return f'<span style="color:{color};font-weight:600">{label}</span>'

    st.markdown(
        '<div style="font-size:0.68rem;color:#4a6a90;letter-spacing:0.1em;'
        'text-transform:uppercase;margin-bottom:0.45rem">Performance Status</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="status-box status-info" style="padding:0.75rem 0.9rem">
            <div style="display:grid;grid-template-columns:1fr auto;row-gap:0.32rem;column-gap:0.8rem;
                        font-size:0.74rem;align-items:center">
                <div>Data cache</div><div>{_badge(data_loaded, "Warm", "Idle")}</div>
                <div>Momentum cache</div><div>{_badge(momentum_ready, "Ready", "Not built")}</div>
                <div>Momentum in optimizer</div><div>{_badge(momentum_enabled and momentum_ready, "Active", "Off")}</div>
                <div>Utility / Tangency</div><div>{_badge(utility_ready and tangency_ready, "Ready", "Pending")}</div>
                <div>Frontier</div><div>{_badge(frontier_ready, "Ready", "Pending")}</div>
                <div>Live tape</div><div>{tape_count} symbols</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Top ticker tape is client-side and refreshes without rerunning Streamlit.")


def render_session_strip():
    ticker_count = len(st.session_state.get("tickers", []))
    tape_count = len(st.session_state.get("tape_indices", []))
    data_loaded = bool(st.session_state.get("data_loaded"))
    momentum_live = bool(
        st.session_state.get("momentum_enabled") and
        st.session_state.get("momentum_final_er") is not None
    )
    start_date = st.session_state.get("start_date", "N/A")
    end_date = st.session_state.get("end_date", "N/A")

    def _pill(label, value, color):
        return (
            f'<div style="padding:0.55rem 0.8rem;border:1px solid #17314d;border-radius:10px;'
            f'background:linear-gradient(180deg,rgba(7,18,33,0.96),rgba(10,22,40,0.88));'
            f'min-width:120px">'
            f'<div style="font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:#4a6a90">{label}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:{color};margin-top:3px">{value}</div>'
            f'</div>'
        )

    items = [
        _pill("Portfolio", f"{ticker_count} tickers", "#d9ecff"),
        _pill("Date Range", f"{start_date} → {end_date}", "#9bd3ff"),
        _pill("Momentum", "Active" if momentum_live else "Idle", "#00e676" if momentum_live else "#ffd54f"),
        _pill("Data", "Loaded" if data_loaded else "Awaiting load", "#00e676" if data_loaded else "#ffb300"),
        _pill("Live Tape", f"{tape_count} symbols", "#7dd3fc"),
    ]

    st.markdown(
        '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin:0.35rem 0 0.85rem 0">'
        + "".join(items) +
        '</div>',
        unsafe_allow_html=True
    )

def show_flash_notice():
    flash = st.session_state.pop("flash_notice", None)
    if not flash:
        return
    level, message = flash
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)



# ── Hero & Workflow ───────────────────────────────────────────
def render_market_weather_panel(meta=None):
    """
    Quant-mode market weather panel using the shared regime engine output.
    """
    meta = meta or st.session_state.get("regime_meta") or st.session_state.get("noob_regime_meta") or {}
    if not meta:
        return

    probs = meta.get("regime_probabilities", {})
    dominant = meta.get("dominant_regime", "unknown")
    confidence = float(meta.get("confidence", 0.0))
    fast_dominant = meta.get("fast_dominant_regime", dominant)
    fast_confidence = float(meta.get("fast_confidence", confidence))
    basket = meta.get("basket", {})
    forecast = meta.get("transition_forecast", {}) or {}
    forecast_next = str(forecast.get("expected_next_regime", fast_dominant)).title()
    forecast_shift = float(forecast.get("bearish_shift_prob_10d", 0.0) or 0.0)
    forecast_stay = float(forecast.get("stay_prob_10d", 0.0) or 0.0)
    forecast_copy = str(forecast.get("summary", "Current regime is likely to persist"))
    alpha = float(st.session_state.get("shrinkage_alpha") or meta.get("alpha", 0.55))
    beta = float(st.session_state.get("momentum_beta") or meta.get("beta", 0.55))

    palette = {
        "bull": ("#22c55e", "Bull"),
        "sideways": ("#ffd166", "Sideways"),
        "bear": ("#ff4d6d", "Bear"),
        "crisis": ("#9b5de5", "Crisis"),
    }
    dominant_color, dominant_label = palette.get(dominant, ("#7dd89a", dominant.capitalize()))

    bars = []
    for regime in ["bull", "sideways", "bear", "crisis"]:
        value = float(probs.get(regime, 0.0))
        color, label = palette[regime]
        bars.append(
            (
                f'<div style="min-width:120px;flex:1;">'
                f'<div style="font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:#6c8bb0;">{label}</div>'
                f'<div style="height:7px;background:#142033;border-radius:999px;overflow:hidden;margin:4px 0;">'
                f'<div style="width:{value * 100:.1f}%;height:100%;background:{color};border-radius:999px;"></div>'
                f'</div>'
                f'<div style="font-size:0.84rem;font-weight:700;color:{color};">{value * 100:.0f}%</div>'
                f'</div>'
            )
        )

    st.markdown(
        f"""
        <div class="workflow-section-shell" style="margin-top:0.9rem;">
            <div class="workflow-section-topline" style="align-items:flex-start;gap:1rem;">
                <div style="flex:1;min-width:260px;">
                    <div class="workflow-section-kicker">Market Weather</div>
                    <div class="section-heading" style="margin:0.22rem 0 0 0;color:{dominant_color};">
                        {dominant_label} Bias
                    </div>
                    <div class="workflow-section-copy" style="margin-top:0.35rem;">
                        The regime engine is blending index trend, volatility, drawdown, basket breadth, participation, and correlation stress.
                    </div>
                </div>
                <div style="display:flex;gap:0.7rem;flex-wrap:wrap;">
                    <div class="metric-shell" style="min-width:120px;">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence * 100:.0f}%</div>
                    </div>
                    <div class="metric-shell" style="min-width:120px;">
                        <div class="metric-label">Alpha</div>
                        <div class="metric-value">{alpha:.0%}</div>
                    </div>
                    <div class="metric-shell" style="min-width:120px;">
                        <div class="metric-label">Momentum Weight</div>
                        <div class="metric-value">{1 - beta:.0%}</div>
                    </div>
                    <div class="metric-shell" style="min-width:150px;">
                        <div class="metric-label">Short-Term Pulse</div>
                        <div class="metric-value" style="font-size:1rem;">{fast_dominant.title()}</div>
                        <div class="metric-copy">{fast_confidence * 100:.0f}% confidence</div>
                    </div>
                    <div class="metric-shell" style="min-width:170px;">
                        <div class="metric-label">Next 10D</div>
                        <div class="metric-value" style="font-size:0.98rem;">{forecast_next}</div>
                        <div class="metric-copy">{forecast_stay * 100:.0f}% stay chance</div>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.85rem;">
                {''.join(bars)}
            </div>
            <div style="display:flex;gap:0.9rem;flex-wrap:wrap;margin-top:0.9rem;font-size:0.82rem;color:#9bb0cd;">
                <span>Above 200D: <strong style="color:#e6f0ff;">{basket.get('above_200_pct', 50):.0f}%</strong></span>
                <span>Participation: <strong style="color:#e6f0ff;">{basket.get('participation_pct', 50):.0f}%</strong></span>
                <span>63D Correlation: <strong style="color:#e6f0ff;">{basket.get('avg_corr_63', 0.35):.2f}</strong></span>
                <span>Downside Share: <strong style="color:#e6f0ff;">{basket.get('downside_pct', 25):.0f}%</strong></span>
                <span>Bear Shift Risk: <strong style="color:#e6f0ff;">{forecast_shift * 100:.0f}%</strong></span>
            </div>
            <div style="margin-top:0.65rem;font-size:0.8rem;color:#9bb0cd;">{forecast_copy}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_regime_change_panel(current_snapshot=None, previous_snapshot=None):
    """Show what changed versus the last successful regime/momentum run."""
    current_snapshot = current_snapshot or st.session_state.get("regime_run_snapshot")
    previous_snapshot = previous_snapshot or st.session_state.get("regime_prev_snapshot")
    if not current_snapshot or not previous_snapshot:
        return

    def _fmt_delta(curr, prev, pct=False, signed=False):
        if curr is None or prev is None:
            return "new"
        diff = curr - prev
        if pct:
            return f"{diff:+.0%}"
        if signed:
            return f"{diff:+.0f}d"
        return f"{diff:+.2f}"

    regime_curr = current_snapshot.get("dominant_regime", "unknown")
    regime_prev = previous_snapshot.get("dominant_regime", "unknown")
    alpha_curr = current_snapshot.get("alpha")
    alpha_prev = previous_snapshot.get("alpha")
    beta_curr = current_snapshot.get("beta")
    beta_prev = previous_snapshot.get("beta")
    lb_curr = current_snapshot.get("lookback")
    lb_prev = previous_snapshot.get("lookback")
    conf_curr = current_snapshot.get("confidence")
    conf_prev = previous_snapshot.get("confidence")
    fast_curr = current_snapshot.get("fast_dominant_regime", regime_curr)
    fast_prev = previous_snapshot.get("fast_dominant_regime", regime_prev)
    horizon_curr = current_snapshot.get("primary_horizon")
    horizon_prev = previous_snapshot.get("primary_horizon")

    st.markdown(
        f"""
        <div class="status-box status-info" style="margin-top:0.7rem;">
            <b>What changed vs last run</b><br>
            Regime: <b>{regime_prev}</b> → <b>{regime_curr}</b> &nbsp;|&nbsp;
            Pulse: <b>{fast_prev}</b> → <b>{fast_curr}</b> &nbsp;|&nbsp;
            Confidence: <b>{conf_curr:.0%}</b> ({_fmt_delta(conf_curr, conf_prev, pct=True)}) &nbsp;|&nbsp;
            Alpha: <b>{alpha_curr:.0%}</b> ({_fmt_delta(alpha_curr, alpha_prev, pct=True)}) &nbsp;|&nbsp;
            Beta: <b>{beta_curr:.0%}</b> ({_fmt_delta(beta_curr, beta_prev, pct=True)}) &nbsp;|&nbsp;
            Lookback: <b>{lb_curr if lb_curr is not None else '—'}d</b> ({_fmt_delta(lb_curr, lb_prev, signed=True) if lb_curr is not None and lb_prev is not None else 'new'}) &nbsp;|&nbsp;
            Focus window: <b>{horizon_curr if horizon_curr is not None else '—'}d</b> ({_fmt_delta(horizon_curr, horizon_prev, signed=True) if horizon_curr is not None and horizon_prev is not None else 'new'})
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hero_action_hub():
    ss = st.session_state
    data_loaded = bool(ss.get("data_loaded"))
    momentum_enabled = bool(ss.get("momentum_enabled"))
    momentum_ready = ss.get("momentum_final_er") is not None
    frontier_ready = bool(ss.get("frontier_computed"))
    portfolios_ready = bool(ss.get("portfolios_computed"))
    ticker_count = len(ss.get("tickers", []))

    overlap_window = "Awaiting load"
    if data_loaded and ss.get("returns") is not None and len(ss.returns.index) > 0:
        overlap_window = (
            f"{ss.returns.index[0].strftime('%Y-%m-%d')} -> "
            f"{ss.returns.index[-1].strftime('%Y-%m-%d')}"
        )

    model_label = "Momentum blend" if momentum_enabled and momentum_ready else "Shrinkage core"
    next_action = (
        "Build momentum and launch the full optimizer stack."
        if data_loaded and momentum_enabled and not momentum_ready else
        "Data is loaded. Run optimizations whenever you're ready."
        if data_loaded else
        "Load the basket first to lock in the optimizer window."
    )

    st.markdown(
        f"""
        <div class="hero-command-deck">
            <div class="command-panel">
                <div class="command-kicker">Command Deck</div>
                <div class="command-headline">Fast path for the current session</div>
                <div class="command-copy">{next_action}</div>
                <div class="command-grid">
                    <div class="command-cell">
                        <div class="command-label">Active basket</div>
                        <div class="command-value">{ticker_count} symbols</div>
                    </div>
                    <div class="command-cell">
                        <div class="command-label">Expected-return engine</div>
                        <div class="command-value {'is-good' if momentum_enabled and momentum_ready else 'is-accent'}">{model_label}</div>
                    </div>
                    <div class="command-cell">
                        <div class="command-label">Optimizer window</div>
                        <div class="command-value is-accent">{overlap_window}</div>
                    </div>
                    <div class="command-cell">
                        <div class="command-label">Result state</div>
                        <div class="command-value {'is-good' if portfolios_ready else 'is-warn'}">{'Portfolios ready' if portfolios_ready else 'Awaiting run'}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.05, 1.05, 0.95])
    with c1:
        if st.button("Load / Refresh Data", key="hero_load_data", use_container_width=True, disabled=ticker_count < 2):
            request_action("load_data")
            st.rerun()
    with c2:
        if st.button(
            "Compute Momentum" if not momentum_ready else "Refresh Momentum",
            key="hero_compute_momentum",
            use_container_width=True,
            disabled=not (momentum_enabled and data_loaded),
        ):
            request_action("compute_momentum")
            st.rerun()
    with c3:
        if st.button("Run ALL", key="hero_run_all", use_container_width=True, disabled=not data_loaded):
            request_action("run_all")
            st.rerun()
    with c4:
        if st.button("Frontier Only", key="hero_frontier_only", use_container_width=True, disabled=not data_loaded):
            request_action("frontier")
            st.rerun()

def render_build_workflow_overview():
    ss = st.session_state
    basket_count = len(ss.get("tickers", []))
    data_ready = bool(ss.get("data_loaded"))
    momentum_enabled = bool(ss.get("momentum_enabled"))
    momentum_ready = ss.get("momentum_final_er") is not None
    portfolio_ready = bool(ss.get("weights_utility") is not None or ss.get("weights_tan") is not None or ss.get("weights_min") is not None)
    frontier_ready = bool(ss.get("frontier_computed"))
    overlap_window = "Awaiting data load"
    if data_ready and ss.get("returns") is not None and len(ss.returns.index) > 0:
        overlap_window = (
            f"{ss.returns.index[0].strftime('%Y-%m-%d')} -> "
            f"{ss.returns.index[-1].strftime('%Y-%m-%d')}"
        )

    steps = [
        {
            "title": "Assemble Universe",
            "copy": "Search, type, or import a basket until the investable set feels right.",
            "chip": f"{basket_count} symbols" if basket_count else "Start here",
            "live": basket_count < 2,
            "ready": basket_count >= 2,
        },
        {
            "title": "Load Market Window",
            "copy": "Lock the strict common-overlap sample before the optimizer touches anything.",
            "chip": "Ready" if data_ready else "Pending load",
            "live": basket_count >= 2 and not data_ready,
            "ready": data_ready,
        },
        {
            "title": "Choose Return Model",
            "copy": "Stay on shrinkage or tilt the engine with momentum signals.",
            "chip": "Momentum ready" if momentum_ready else ("Momentum on" if momentum_enabled else "Shrinkage"),
            "live": data_ready and momentum_enabled and not momentum_ready,
            "ready": data_ready and (momentum_ready or not momentum_enabled),
        },
        {
            "title": "Run the Desk",
            "copy": "Compute utility, tangency, frontier, and compare what deserves capital.",
            "chip": "Results ready" if portfolio_ready or frontier_ready else "Awaiting run",
            "live": data_ready and not (portfolio_ready or frontier_ready),
            "ready": portfolio_ready or frontier_ready,
        },
    ]

    rail_html = []
    for idx, step in enumerate(steps, start=1):
        classes = ["workflow-step"]
        if step["live"]:
            classes.append("is-live")
        if step["ready"]:
            classes.append("is-ready")
        rail_html.append(
            (
                f'<div class="{" ".join(classes)}">'
                f'<div class="workflow-kicker">Step {idx}</div>'
                f'<div class="workflow-title">{html.escape(step["title"])}</div>'
                f'<div class="workflow-copy">{html.escape(step["copy"])}</div>'
                f'<div class="workflow-chip">{html.escape(step["chip"])}</div>'
                f'</div>'
            )
        )

    momentum_state = "Active" if momentum_enabled and momentum_ready else ("Enabled" if momentum_enabled else "Off")
    summary_html = f"""
        <div class="workflow-summary">
            <div class="workflow-metric">
                <div class="workflow-metric-label">Basket</div>
                <div class="workflow-metric-value">{basket_count}</div>
                <div class="workflow-metric-copy">symbols selected for the current build</div>
            </div>
            <div class="workflow-metric">
                <div class="workflow-metric-label">Optimizer Window</div>
                <div class="workflow-metric-value" style="font-size:0.98rem">{overlap_window}</div>
                <div class="workflow-metric-copy">strict common overlap used by the optimizer</div>
            </div>
            <div class="workflow-metric">
                <div class="workflow-metric-label">Return Model</div>
                <div class="workflow-metric-value">{momentum_state}</div>
                <div class="workflow-metric-copy">{"momentum-blended ERs are driving the optimizer" if momentum_enabled and momentum_ready else "shrinkage-adjusted returns are active"}</div>
            </div>
            <div class="workflow-metric">
                <div class="workflow-metric-label">Desk State</div>
                <div class="workflow-metric-value">{'Live' if portfolio_ready or frontier_ready else 'Staged'}</div>
                <div class="workflow-metric-copy">{'results are ready to inspect across tabs' if portfolio_ready or frontier_ready else 'build and load data to unlock the full desk'}</div>
            </div>
        </div>
    """

    rail_markup = '<div class="workflow-rail">' + "".join(rail_html) + '</div>'
    st.markdown(rail_markup, unsafe_allow_html=True)
    st.markdown(summary_html, unsafe_allow_html=True)


# ── Universe Helper ───────────────────────────────────────────
def render_universe_helper(helper_mode):
    st.markdown(
        """
        <div class="hero-banner" style="padding:1.25rem 1.5rem;margin-bottom:1rem">
            <div class="hero-subtitle">Universe Builder Helper</div>
            <div class="hero-title" style="font-size:2rem;margin-top:0.35rem">Basket <span>Desk</span></div>
            <div class="hero-badge">Send symbols back to the main app</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if helper_mode == "upload":
        st.markdown("### Upload Basket File")
        st.caption("Upload CSV, TXT, or Excel here in a separate window, then send the cleaned symbol list back to the main app.")
        upload = st.file_uploader(
            "Choose a basket file",
            type=["csv", "txt", "xlsx", "xls"],
            key="helper_universe_upload",
        )
        symbols = parse_uploaded_symbols(upload)
        source = "file helper"
    else:
        st.markdown("### Sector Basket Helper")
        st.caption("Choose a ready-made sector basket here and send it back to the main app in one shot.")
        sector_name = st.selectbox(
            "Sector preset",
            options=sorted(SECTOR_PRESETS.keys()),
            key="helper_sector_pick",
        )
        symbols = SECTOR_PRESETS.get(sector_name, [])
        source = f"sector helper · {sector_name}"

    if symbols:
        preview = ", ".join(symbols[:20])
        st.success(f"{len(symbols)} symbols ready.")
        st.code(preview + (f" ... +{len(symbols) - 20} more" if len(symbols) > 20 else ""))
        h1, h2 = st.columns(2)
        with h1:
            if st.button("Send As Replace", use_container_width=True, key="helper_send_replace_btn"):
                if push_bridge_symbols(symbols, "replace", source):
                    st.success("Sent to main app. Return to the main tab and refresh or click any action.")
        with h2:
            if st.button("Send As Append", use_container_width=True, key="helper_send_append_btn"):
                if push_bridge_symbols(symbols, "append", source):
                    st.success("Sent to main app. Return to the main tab and refresh or click any action.")
    else:
        st.info("No symbols ready yet.")

    st.link_button("Open Main App", "?", use_container_width=True)
    st.stop()


# ── Alerts ────────────────────────────────────────────────────
def evaluate_alerts(alerts):
    enabled = [a for a in alerts if a.get("enabled", True) and a.get("symbol")]
    symbols = sorted({a["symbol"] for a in enabled})
    if not symbols:
        return [], {}

    quotes = cached_fetch_tape_quotes(tuple(symbols))
    quote_map = {q["symbol"]: q for q in quotes}
    hits = []
    for alert in enabled:
        symbol = alert["symbol"]
        if symbol not in quote_map:
            continue
        price = quote_map[symbol]["price"]
        threshold = float(alert.get("threshold", 0.0))
        condition = alert.get("condition", "above")
        triggered = price >= threshold if condition == "above" else price <= threshold
        if triggered:
            hits.append({
                "symbol": symbol,
                "price": price,
                "threshold": threshold,
                "condition": condition,
                "note": alert.get("note", ""),
            })
    return hits, quote_map



# ── Live Ticker Tape ──────────────────────────────────────────
def render_ticker_tape(chosen_names: list, all_indices: dict):
    """Server-side cached ticker tape for reliable quote display."""
    if not chosen_names:
        return
    sym_to_name = {v: k for k, v in all_indices.items()}
    symbols = [all_indices[n] for n in chosen_names if n in all_indices]
    if not symbols:
        return
    items = cached_fetch_tape_quotes(tuple(symbols))
    if not items:
        st.markdown(
            '<div class="status-box status-warn">Live tape quotes are temporarily unavailable. '
            'Use the refresh control in the sidebar to try again.</div>',
            unsafe_allow_html=True
        )
        return

    def _fmt_price(p):
        if p >= 1000:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:.3f}"
        return f"{p:.6f}"

    duration = max(18, len(items) * 4)
    tape_items = []
    for d in items:
        name = sym_to_name.get(d["symbol"], d["symbol"])
        chg = d["chg"]
        arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "●")
        cls = "qs-tick-up" if chg > 0 else ("qs-tick-dn" if chg < 0 else "qs-tick-neu")
        tape_items.append(
            f'<span class="qs-tick">'
            f'<span class="qs-tick-sym">{name}</span>'
            f'<span class="qs-tick-price">{_fmt_price(d["price"])}</span>'
            f'<span class="{cls}">{arrow} {abs(chg):.2f}%</span>'
            f'</span>'
        )
    doubled = "".join(tape_items + tape_items)

    st.markdown(
        f"""
        <div class="qs-tape-shell">
          <div class="qs-tape-head">
            <div class="qs-tape-title">Live Market Strip</div>
            <div class="qs-tape-meta">
              <span class="qs-tape-badge">{len(items)} symbols</span>
              <span class="qs-tape-badge">server cached</span>
              <span class="qs-tape-badge">refresh from sidebar</span>
            </div>
          </div>
          <div class="qs-tape-wrap">
            <div class="qs-tape-inner" style="animation-duration:{duration}s">
              {doubled}
            </div>
          </div>
          <div class="qs-tape-ts">Quotes cached for 60s to keep the app responsive.</div>
        </div>
        """,
        unsafe_allow_html=True
    )



# ── Metric Cards & Weights Table ──────────────────────────────
def metric_card(label, value, style=""):
    return f"""
    <div class="metric-box">
        <div class="label">{label}</div>
        <div class="value {style}">{value}</div>
    </div>"""


def weight_bar(pct: float, color="#00b4ff"):
    return f"""
    <div class="weight-bar-bg">
        <div class="weight-bar-fill" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,#0066cc,{color});"></div>
    </div>"""


def make_weights_table(tickers, weights, label_col="Weight"):
    rows = ""
    for t, w in zip(tickers, weights):
        pct = w * 100
        rows += f"""<tr>
            <td><span class="ticker-chip">{t}</span></td>
            <td style="font-family:'DM Mono'">{w:.4f}</td>
            <td style="font-family:'DM Mono';color:#00e676">{pct:.2f}%</td>
            <td style="min-width:140px">{weight_bar(pct)}</td>
        </tr>"""
    return f"""
    <table class="quansen-table">
        <thead><tr>
            <th>Ticker</th>
            <th>{label_col}</th>
            <th>Allocation %</th>
            <th>Visual</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def portfolio_metrics_row(ann_ret, ann_risk, sharpe):
    ret_class  = "positive" if ann_ret > 0 else ""
    shp_class  = "accent" if sharpe > 1 else ("positive" if sharpe > 0 else "")
    st.markdown(f"""
    <div class="metric-row">
        {metric_card("Annual Return",  f"{ann_ret*100:.2f}%",   ret_class)}
        {metric_card("Annual Risk",    f"{ann_risk*100:.2f}%",  "")}
        {metric_card("Sharpe Ratio",   f"{sharpe:.3f}",          shp_class)}
        {metric_card("Risk-Free Rate", f"{RF_ANNUAL*100:.1f}%", "accent")}
    </div>""", unsafe_allow_html=True)


def render_insight_note(text: str):
    st.markdown(f'<div class="insight-note">{text}</div>', unsafe_allow_html=True)



# ── Portfolio Spotlights ──────────────────────────────────────
def _spotlight_metric(label, value, tone=""):
    return (
        '<div class="spotlight-metric">'
        f'<div class="label">{label}</div>'
        f'<div class="value {tone}">{value}</div>'
        '</div>'
    )


def render_portfolio_spotlight(title, subtitle, weights, ann_ret, ann_risk, sharpe, pie_fig, badge=None, badge_gold=False):
    tone_ret = "good" if ann_ret > RF_ANNUAL else "accent"
    tone_sharpe = "good" if sharpe > 1 else ("accent" if sharpe > 0 else "warn")
    tag_class = "spotlight-tag is-gold" if badge_gold else "spotlight-tag"
    tag_html = f'<div class="{tag_class}">{badge}</div>' if badge else ""
    top_weights = sorted(zip(st.session_state.tickers, weights), key=lambda x: x[1], reverse=True)[:3]
    thesis = " · ".join([f"{t} {w*100:.1f}%" for t, w in top_weights])
    st.markdown(
        (
            '<div class="spotlight-shell{}">'.format(" is-winner" if badge_gold else "") +
            '<div class="spotlight-topline"><div class="spotlight-title">{}</div>{}</div>'.format(title, tag_html) +
            f'<div class="spotlight-copy">{subtitle}<br><span style="color:#5f7fa1">Top weights:</span> {thesis}</div>' +
            '<div class="spotlight-metrics">' +
            _spotlight_metric("Annual Return", f"{ann_ret*100:.2f}%", tone_ret) +
            _spotlight_metric("Annual Risk", f"{ann_risk*100:.2f}%", "") +
            _spotlight_metric("Sharpe Ratio", f"{sharpe:.3f}", tone_sharpe) +
            _spotlight_metric("Risk-Free Rate", f"{RF_ANNUAL*100:.1f}%", "accent") +
            '</div></div>'
        ),
        unsafe_allow_html=True
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(make_weights_table(st.session_state.tickers, weights, "Weight"), unsafe_allow_html=True)
    with right:
        st.plotly_chart(pie_fig, use_container_width=True)


def render_comparison_spotlights(comp_df: pd.DataFrame):
    if comp_df.empty:
        return
    max_return = max(comp_df["Return %"].max(), 1e-9)
    max_risk = max(comp_df["Risk %"].max(), 1e-9)
    max_sharpe = max(comp_df["Sharpe"].max(), 1e-9)
    best_sharpe = comp_df["Sharpe"].idxmax()
    best_return = comp_df["Return %"].idxmax()
    best_risk = comp_df["Risk %"].idxmin()

    cards = []
    for name, row in comp_df.iterrows():
        badges = []
        if name == best_sharpe:
            badges.append("Best Sharpe")
        if name == best_return:
            badges.append("Highest Return")
        if name == best_risk:
            badges.append("Lowest Risk")
        badge_html = f'<div class="compare-badge">{html.escape(" · ".join(badges))}</div>' if badges else ""
        return_width = (row["Return %"] / max_return) * 100 if max_return > 0 else 0.0
        risk_width = (row["Risk %"] / max_risk) * 100 if max_risk > 0 else 0.0
        sharpe_width = (max(row["Sharpe"], 0) / max_sharpe) * 100 if max_sharpe > 0 else 0.0
        cards.append(
            (
                f'<div class="compare-card{" is-best" if badges else ""}">'
                f'<div class="compare-name">{html.escape(str(name))}</div>'
                f'{badge_html}'
                f'<div class="compare-row"><div class="compare-label">Return</div>'
                f'<div class="compare-value">{row["Return %"]:.2f}%</div>'
                f'<div class="compare-bar"><div class="compare-fill" style="width:{return_width:.1f}%;background:linear-gradient(90deg,#008cff,#00e5ff)"></div></div></div>'
                f'<div class="compare-row"><div class="compare-label">Risk</div>'
                f'<div class="compare-value">{row["Risk %"]:.2f}%</div>'
                f'<div class="compare-bar"><div class="compare-fill" style="width:{risk_width:.1f}%;background:linear-gradient(90deg,#ff7043,#ffca28)"></div></div></div>'
                f'<div class="compare-row"><div class="compare-label">Sharpe</div>'
                f'<div class="compare-value">{row["Sharpe"]:.3f}</div>'
                f'<div class="compare-bar"><div class="compare-fill" style="width:{sharpe_width:.1f}%;background:linear-gradient(90deg,#00c853,#69f0ae)"></div></div></div>'
                f'</div>'
            )
        )
    st.markdown('<div class="compare-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════
#  NOOB MODE — CSS ADDITIONS
#  Append this block into GLOBAL_CSS (or inject via st.markdown)
#  All classes are prefixed .nb- to avoid collisions with Neat CSS
# ══════════════════════════════════════════════════════════════

NOOB_CSS = """
<style>
/* ════════════════════════════════════════════════════════════
   NOOB MODE — FULL GREEN THEME
   All classes prefixed .nb- to avoid collision with Neat CSS
   Palette: #031a0e bg · #1a5c2a border · #3db85a muted · #4afa7a accent · #c8ffd4 text
   ════════════════════════════════════════════════════════════ */

/* ── Global body override in noob mode ── */
.nb-body { font-family: 'Syne', sans-serif; }

/* ── Mode toggle buttons ── */
.nb-toggle-wrap {
    display: flex;
    gap: 0;
    background: #041209;
    border: 1px solid #1a5c2a;
    border-radius: 999px;
    padding: 3px;
    margin: 0.6rem 0 1rem 0;
}
.nb-toggle-btn {
    flex: 1;
    padding: 0.38rem 0.6rem;
    border-radius: 999px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #3db85a;
    transition: all 0.22s;
    text-align: center;
}
.nb-toggle-btn.active-noob {
    background: linear-gradient(135deg, #0d6b2a, #12882f);
    color: #c8ffd4;
    box-shadow: 0 0 14px rgba(26,173,68,0.45);
}
.nb-toggle-btn.active-neat {
    background: linear-gradient(135deg, #0066cc, #00b4ff);
    color: #e8f4ff;
    box-shadow: 0 0 14px rgba(0,150,255,0.3);
}

/* ── Noob hero banner ── */
.nb-hero {
    background: linear-gradient(135deg, #031a0e 0%, #042b12 60%, #031508 100%);
    border: 1px solid #1a5c2a;
    border-radius: 18px;
    padding: 2rem 2.4rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: fadeLift 0.45s ease both;
    box-shadow: 0 4px 32px rgba(0,200,80,0.10);
}
.nb-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(74,250,122,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.nb-hero-emoji {
    font-size: 3rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    display: block;
}
.nb-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #c8ffd4;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.nb-hero-title span { color: #4afa7a; }
.nb-hero-sub {
    font-size: 1rem;
    color: #3db85a;
    margin: 0 0 1.2rem 0;
    line-height: 1.5;
}
.nb-hero-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
}
.nb-chip {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.nb-chip-green { background: rgba(74,250,122,0.12); border: 1px solid rgba(74,250,122,0.3);  color: #4afa7a; }
.nb-chip-amber { background: rgba(255,213,79,0.12);  border: 1px solid rgba(255,213,79,0.28); color: #ffd54f; }
.nb-chip-blue  { background: rgba(0,200,100,0.12);   border: 1px solid rgba(0,200,100,0.28);  color: #69f0ae; }
.nb-chip-pink  { background: rgba(0,230,118,0.10);   border: 1px solid rgba(0,230,118,0.25);  color: #00e676; }

/* ── Noob section heading ── */
.nb-section {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #c8ffd4;
    border-left: 4px solid #22c55e;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Piggy bank simulator card ── */
.nb-piggy-shell {
    background: linear-gradient(180deg, #031a0e, #042214);
    border: 1px solid #1a5c2a;
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    margin: 0.6rem 0 1rem 0;
    animation: fadeLift 0.4s ease both;
    box-shadow: 0 2px 20px rgba(0,200,80,0.08);
}
.nb-piggy-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3db85a;
    margin-bottom: 0.35rem;
}
.nb-piggy-number {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #4afa7a;
    line-height: 1;
}
.nb-piggy-gain {
    font-size: 1rem;
    color: #3db85a;
    margin-top: 0.35rem;
}
.nb-piggy-gain.up   { color: #4afa7a; }
.nb-piggy-gain.down { color: #ff5252; }
.nb-piggy-bar-bg {
    background: rgba(74,250,122,0.07);
    border-radius: 999px;
    height: 10px;
    margin: 1rem 0 0.4rem 0;
    overflow: hidden;
    border: 1px solid rgba(74,250,122,0.15);
}
.nb-piggy-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0d6b2a, #22c55e, #4afa7a);
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── Ride smoothness badge ── */
.nb-ride-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.nb-ride-smooth { background: rgba(74,250,122,0.10); border:1px solid rgba(74,250,122,0.30); color:#4afa7a; }
.nb-ride-bumpy  { background: rgba(255,213,79,0.10);  border:1px solid rgba(255,213,79,0.30); color:#ffd54f; }
.nb-ride-wild   { background: rgba(255,82,82,0.08);   border:1px solid rgba(255,82,82,0.28);  color:#ff5252; }

/* ── Fight club card ── */
.nb-fight-shell {
    background: linear-gradient(180deg, #031a0e, #042214);
    border: 1px solid #1a5c2a;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin: 0.6rem 0 1rem 0;
    animation: fadeLift 0.4s ease both;
    box-shadow: 0 2px 20px rgba(0,200,80,0.08);
}
.nb-vs-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0.8rem;
    align-items: center;
    margin: 0.8rem 0;
}
.nb-fight-card {
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.nb-fight-card.winner {
    background: linear-gradient(180deg, rgba(10,60,25,0.9), rgba(5,35,15,0.85));
    border: 1px solid #22c55e;
    box-shadow: 0 0 14px rgba(34,197,94,0.15);
}
.nb-fight-card.loser {
    background: linear-gradient(180deg, rgba(40,10,10,0.85), rgba(25,8,8,0.8));
    border: 1px solid rgba(255,82,82,0.22);
}
.nb-fight-card.draw {
    background: linear-gradient(180deg, rgba(5,25,12,0.85), rgba(4,18,9,0.8));
    border: 1px solid #1a5c2a;
}
.nb-fight-ticker {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #c8ffd4;
}
.nb-fight-return {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.3rem;
}
.nb-fight-return.up   { color: #4afa7a; }
.nb-fight-return.down { color: #ff5252; }
.nb-fight-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3db85a;
    margin-top: 0.25rem;
}
.nb-vs-badge {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1a5c2a;
    text-align: center;
}
.nb-crown { font-size: 1.5rem; display: block; }

/* ── Time machine card ── */
.nb-timemachine-shell {
    background: linear-gradient(180deg, #031a0e, #042214);
    border: 1px solid #1a5c2a;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin: 0.6rem 0 1rem 0;
    animation: fadeLift 0.4s ease both;
    box-shadow: 0 2px 20px rgba(0,200,80,0.08);
}
.nb-tm-result-shell {
    background: rgba(10,40,18,0.6);
    border: 1px solid #1a5c2a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.nb-tm-headline {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #4afa7a;
    line-height: 1;
}
.nb-tm-sub {
    font-size: 0.82rem;
    color: #3db85a;
    margin-top: 0.3rem;
}

/* ── Monthly scoreboard ── */
.nb-month-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 0.4rem;
    margin-top: 0.6rem;
}
.nb-month-cell {
    border-radius: 10px;
    padding: 0.5rem 0.4rem;
    text-align: center;
    font-size: 0.72rem;
}
.nb-month-cell .mo  { font-size: 0.6rem; color: #3db85a; letter-spacing: 0.08em; text-transform: uppercase; }
.nb-month-cell .ret { font-family: 'DM Mono', monospace; font-weight: 600; margin-top: 2px; }
.nb-month-win  { background: rgba(74,250,122,0.07); border:1px solid rgba(74,250,122,0.18); }
.nb-month-win .ret { color: #4afa7a; }
.nb-month-loss { background: rgba(255,82,82,0.06);  border:1px solid rgba(255,82,82,0.18); }
.nb-month-loss .ret { color: #ff5252; }

/* ── Recommended pick card ── */
.nb-pick-shell {
    background: linear-gradient(135deg, #031a0e, #052e14);
    border: 2px solid #22c55e;
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    margin: 0.6rem 0 1.2rem 0;
    animation: fadeLift 0.4s ease both;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 28px rgba(34,197,94,0.12);
}
.nb-pick-shell::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(74,250,122,0.09) 0%, transparent 70%);
    pointer-events: none;
}
.nb-pick-crown {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 0.4rem;
}
.nb-pick-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #4afa7a;
    margin-bottom: 0.2rem;
}
.nb-pick-copy {
    font-size: 0.82rem;
    color: #3db85a;
    line-height: 1.5;
    max-width: 520px;
}
.nb-pick-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
    margin-top: 1.1rem;
}
.nb-pick-stat {
    background: rgba(74,250,122,0.06);
    border: 1px solid #1a5c2a;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.nb-pick-stat-label { font-size:0.6rem; letter-spacing:0.12em; text-transform:uppercase; color:#2e7d52; margin-bottom:0.2rem; }
.nb-pick-stat-value { font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800; color:#4afa7a; }

/* ── Slot machine weights — green palette ── */
.nb-slot-shell {
    background: linear-gradient(180deg, #031a0e, #042214);
    border: 1px solid #1a5c2a;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin: 0.6rem 0 1rem 0;
    animation: fadeLift 0.4s ease both;
    box-shadow: 0 2px 20px rgba(0,200,80,0.08);
}
.nb-slot-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(26,92,42,0.35);
}
.nb-slot-row:last-child { border-bottom: none; }
.nb-slot-ticker {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #7dd89a;
    min-width: 100px;
}
.nb-slot-bar-bg {
    flex: 1;
    background: rgba(26,92,42,0.25);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.nb-slot-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0d6b2a, #22c55e, #4afa7a);
    transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.nb-slot-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #4afa7a;
    min-width: 48px;
    text-align: right;
}

/* ── Plain-English risk gauge ── */
.nb-risk-gauge {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    margin: 0.6rem 0;
}
.nb-risk-row {
    display: grid;
    grid-template-columns: 130px 1fr auto;
    align-items: center;
    gap: 0.7rem;
}
.nb-risk-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7dd89a;
}
.nb-risk-bar-bg {
    background: rgba(26,92,42,0.2);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.nb-risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.7s ease;
}
.nb-risk-label {
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
    white-space: nowrap;
}
.nb-risk-smooth { color:#4afa7a; background:rgba(74,250,122,0.08); border:1px solid rgba(74,250,122,0.22); }
.nb-risk-bumpy  { color:#ffd54f; background:rgba(255,213,79,0.08); border:1px solid rgba(255,213,79,0.22); }
.nb-risk-wild   { color:#ff5252; background:rgba(255,82,82,0.06);  border:1px solid rgba(255,82,82,0.20); }

/* ── Fun step card ── */
.nb-step-card {
    border: 1px solid #1a5c2a;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    background: linear-gradient(180deg, #031a0e, #042214);
    margin: 0.5rem 0 0.8rem 0;
}
.nb-step-emoji { font-size: 2rem; display: block; margin-bottom: 0.4rem; }
.nb-step-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; color:#c8ffd4; margin-bottom:0.2rem; }
.nb-step-copy  { font-size:0.78rem; color:#3db85a; line-height:1.45; }
.nb-step-chip  {
    display:inline-block; margin-top:0.5rem;
    padding:0.22rem 0.55rem; border-radius:999px;
    font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
    background:rgba(74,250,122,0.08); color:#4afa7a; border:1px solid rgba(74,250,122,0.22);
}

/* ── Return emoji badge ── */
.nb-ret-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    padding: 0.5rem 1rem;
    border-radius: 14px;
}
.nb-ret-rocket { background:rgba(74,250,122,0.10); border:1px solid rgba(74,250,122,0.30); color:#4afa7a; }
.nb-ret-ok     { background:rgba(0,230,118,0.08);  border:1px solid rgba(0,230,118,0.25);  color:#00e676; }
.nb-ret-meh    { background:rgba(255,213,79,0.10); border:1px solid rgba(255,213,79,0.28); color:#ffd54f; }
.nb-ret-oops   { background:rgba(255,82,82,0.08);  border:1px solid rgba(255,82,82,0.25);  color:#ff5252; }

/* ── Noob info box — green tinted ── */
.nb-info-box {
    border: 1px solid #1a5c2a;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    background: rgba(10,40,18,0.5);
    color: #7dd89a;
    font-size: 0.82rem;
    line-height: 1.55;
    margin: 0.5rem 0 0.9rem;
}
.nb-info-box strong { color: #c8ffd4; }

/* ── Dashboard stat cards ── */
.nb-dash-card {
    background: linear-gradient(135deg, #031a0e, #042214);
    border: 1px solid #1a5c2a;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.7rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 14px rgba(0,200,80,0.06);
}
.nb-dash-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #0d6b2a, #4afa7a, #0d6b2a);
}
.nb-dash-card-label {
    font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
    color:#3db85a; margin-bottom:0.25rem;
}
.nb-dash-card-value {
    font-size:1.6rem; font-weight:800; color:#4afa7a;
    font-family:'DM Mono',monospace; line-height:1.1;
}
.nb-dash-card-sub { font-size:0.7rem; color:#2e7d52; margin-top:0.2rem; }
.nb-dash-card-icon { position:absolute; top:0.8rem; right:1rem; font-size:1.8rem; opacity:0.15; }

/* ── Stock chip ── */
.nb-stock-chip {
    display:inline-flex; align-items:center; gap:4px;
    background:rgba(0,200,80,0.10); border:1px solid #1a7a38;
    border-radius:20px; padding:3px 10px 3px 8px;
    font-size:0.72rem; color:#7dd89a; margin:2px 3px;
    font-family:'DM Mono',monospace;
}

/* ── Green section heading ── */
.nb-green-heading {
    font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
    color:#22c55e; font-weight:700;
    margin:1.2rem 0 0.6rem;
    padding-bottom:0.3rem; border-bottom:1px solid #1a5c2a;
}

/* ── Status dot ── */
.nb-green-status-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#22c55e; box-shadow:0 0 8px #22c55e;
    margin-right:6px; vertical-align:middle;
}
.nb-green-status-dot.red { background:#ff4d6d; box-shadow:0 0 8px #ff4d6d; }
.nb-green-status-dot.yellow { background:#ffd166; box-shadow:0 0 8px #ffd166; }

/* ── Responsive noob ── */
@media (max-width: 760px) {
    .nb-vs-grid { grid-template-columns: 1fr; }
    .nb-pick-stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .nb-risk-row { grid-template-columns: 90px 1fr auto; }
    .nb-hero-title { font-size: 1.8rem; }
}
@media (max-width: 480px) {
    .nb-piggy-number { font-size: 2.2rem; }
    .nb-fight-return { font-size: 1.5rem; }
    .nb-pick-stats { grid-template-columns: 1fr; }
    .nb-month-grid { grid-template-columns: repeat(4, 1fr); }
}
</style>
"""


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — SIDEBAR TOGGLE
# ══════════════════════════════════════════════════════════════

def render_mode_toggle():
    """
    Drop this right after the logo block in the sidebar.
    Sets st.session_state.app_mode to either 'noob' or 'neat'.
    Returns the current mode string so main.py can branch on it.
    """
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "neat"  # Neat is default

    st.markdown(NOOB_CSS, unsafe_allow_html=True)

    col_noob, col_neat = st.columns(2)
    with col_noob:
        noob_active = "active-noob" if st.session_state.app_mode == "noob" else ""
        if st.button("🎮 Noob", key="mode_btn_noob", use_container_width=True, disabled=st.session_state.app_mode == "noob"):
            st.session_state.app_mode = "noob"
            st.rerun()
    with col_neat:
        neat_active = "active-neat" if st.session_state.app_mode == "neat" else ""
        if st.button("📐 Neat", key="mode_btn_neat", use_container_width=True, disabled=st.session_state.app_mode == "neat"):
            st.session_state.app_mode = "neat"
            st.rerun()

    # Visual indicator of current mode
    if st.session_state.app_mode == "noob":
        st.markdown(
            '<div style="text-align:center;font-size:0.62rem;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#22c55e;padding:0.2rem 0 0.5rem">🌿 Easy mode active</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align:center;font-size:0.62rem;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#1a4a6a;padding:0.2rem 0 0.5rem">📐 Quant mode active</div>',
            unsafe_allow_html=True
        )

    return st.session_state.app_mode


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — HERO BANNER
# ══════════════════════════════════════════════════════════════

def render_noob_hero():
    """
    Replaces the standard QuanSen hero banner in Noob mode.
    Shows a friendly, money-focused welcome with zero jargon.
    """
    ss = st.session_state
    ticker_count = len(ss.get("tickers", []))
    data_loaded = bool(ss.get("data_loaded"))
    portfolios_ready = bool(
        ss.get("weights_utility") is not None or
        ss.get("weights_tan") is not None
    )

    # Dynamic status line so the hero feels alive
    if portfolios_ready:
        status_line = "Your picks have been crunched. Check <b>🎰 The Mix</b> tab to see the results! 💰"
    elif data_loaded:
        status_line = "Data's in! Hit <b>🚀 Run Magic!</b> in the sidebar to generate your portfolios."
    elif ticker_count >= 2:
        status_line = f"You've got {ticker_count} stocks lined up. Hit <b>📡 Load Data</b> in the sidebar to get the party started!"
    else:
        status_line = "Add stocks using the sidebar on the left — or hit <b>🎲 Load 20</b> for an instant portfolio. 😄"

    st.markdown(
        f"""
        <div class="nb-hero">
            <span class="nb-hero-emoji">🐷💸</span>
            <div class="nb-hero-title">Make Your <span>Money Work</span></div>
            <div class="nb-hero-sub">{status_line}</div>
            <div class="nb-hero-chips">
                <span class="nb-chip nb-chip-green">No spreadsheets needed</span>
                <span class="nb-chip nb-chip-amber">Real stock data</span>
                <span class="nb-chip nb-chip-blue">Real math under the hood</span>
                <span class="nb-chip nb-chip-pink">100% jargon-free</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — HOW IT WORKS (Step cards replacing workflow rail)
# ══════════════════════════════════════════════════════════════

def render_noob_how_it_works():
    """
    Replaces render_build_workflow_overview() in Noob mode.
    Four big friendly step cards instead of the quant workflow rail.
    """
    ss = st.session_state
    basket_count = len(ss.get("tickers", []))
    data_ready = bool(ss.get("data_loaded"))
    portfolios_ready = bool(
        ss.get("weights_utility") is not None or
        ss.get("weights_tan") is not None
    )

    steps = [
        {
            "emoji": "🛒",
            "title": "Pick Your Stocks",
            "copy": "Search for companies in the Dashboard tab or use the sidebar. Hit 🎲 Load 20 for an instant Indian stock basket.",
            "chip": f"{basket_count} stocks chosen" if basket_count else "Start here →",
            "done": basket_count >= 2,
        },
        {
            "emoji": "📡",
            "title": "Grab the Numbers",
            "copy": "Hit Load Market Data and we'll pull years of real price history from the market. Takes about 10 seconds.",
            "chip": "Done! ✓" if data_ready else "Waiting for you",
            "done": data_ready,
        },
        {
            "emoji": "🧮",
            "title": "Let the Magic Happen",
            "copy": "Hit 🚀 Run Magic! and our optimizer figures out the best mix of your stocks to grow your money with the least drama.",
            "chip": "Results ready! ✓" if portfolios_ready else "Almost there",
            "done": portfolios_ready,
        },
        {
            "emoji": "🎯",
            "title": "See Your Results",
            "copy": "Check the other tabs! You'll see how much your money could have grown, which stocks do the heavy lifting, and how bumpy the ride gets.",
            "chip": "Explore the tabs →",
            "done": portfolios_ready,
        },
    ]

    cols = st.columns(4)
    for col, step in zip(cols, steps):
        with col:
            border_color = "#22c55e" if step["done"] else "#1a5c2a"
            st.markdown(
                f"""
                <div class="nb-step-card" style="border-color:{border_color}">
                    <span class="nb-step-emoji">{step['emoji']}</span>
                    <div class="nb-step-title">{step['title']}</div>
                    <div class="nb-step-copy">{step['copy']}</div>
                    <div class="nb-step-chip">{step['chip']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — SHARED EXPLAINERS
# ══════════════════════════════════════════════════════════════

def _noob_ride_profile(ann_risk: float):
    """Map annualised risk to a plain-English ride label."""
    if ann_risk < 0.10:
        return "😌", "Smooth ride", "This mix is built for calmer compounding and fewer stomach-drop moments."
    if ann_risk < 0.20:
        return "✈️", "Some turbulence", "Expect normal ups and downs, but not full roller-coaster behavior."
    return "🎢", "Wild ride", "This mix can move around a lot. Bigger upside usually comes with bigger nerves."


def _noob_regime_profile(meta: dict):
    """Plain-English market regime summary for noob mode."""
    dominant = (meta or {}).get("dominant_regime", "unknown")
    profiles = {
        "bull": ("🟢", "Tailwind", "#22c55e", "The market looks supportive right now, so the engine is more willing to reward momentum and stock-specific strength."),
        "sideways": ("🟡", "Chop Zone", "#ffd166", "The market looks mixed, so the engine stays balanced and avoids over-committing to any one style."),
        "bear": ("🔴", "Storm Watch", "#ff4d6d", "The market is showing defensive signals, so the engine leans more on safer assumptions and reduces stock-specific trust."),
        "crisis": ("🟣", "Crash Alert", "#9b5de5", "Stress signals are elevated, so the engine shifts into preservation mode and becomes much stricter."),
    }
    return profiles.get(dominant, ("⚪", "Unknown", "#7dd89a", "The engine does not have enough signal strength yet, so it stays neutral."))


def render_noob_market_weather_card():
    """Explain the current detected market regime in plain English."""
    meta = st.session_state.get("noob_regime_meta") or {}
    if not meta:
        return

    emoji, label, color, copy = _noob_regime_profile(meta)
    probs = meta.get("regime_probabilities", {})
    basket = meta.get("basket", {})
    confidence = float(meta.get("confidence", 0.0))
    fast_dominant = meta.get("fast_dominant_regime", meta.get("dominant_regime", "unknown"))
    forecast = meta.get("transition_forecast", {}) or {}
    forecast_next = str(forecast.get("expected_next_regime", fast_dominant)).title()
    forecast_shift = float(forecast.get("bearish_shift_prob_10d", 0.0) or 0.0)
    forecast_copy = str(forecast.get("summary", "The short-term mood is still settling."))
    alpha = float(st.session_state.get("shrinkage_alpha") or meta.get("alpha", 0.55))
    beta = float(st.session_state.get("momentum_beta") or meta.get("beta", 0.55))
    momentum_meta = st.session_state.get("momentum_meta") or {}
    primary_horizon = momentum_meta.get("primary_horizon")

    bars = []
    for regime in ["bull", "sideways", "bear", "crisis"]:
        prob = float(probs.get(regime, 0.0))
        bar_color = {
            "bull": "#22c55e",
            "sideways": "#ffd166",
            "bear": "#ff4d6d",
            "crisis": "#9b5de5",
        }[regime]
        bars.append(
            f'<div style="min-width:110px;flex:1;">'
            f'<div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em">{regime}</div>'
            f'<div style="height:7px;background:#eef5ef;border-radius:999px;overflow:hidden;margin:4px 0 5px;">'
            f'<div style="height:100%;width:{prob * 100:.1f}%;background:{bar_color};border-radius:999px;"></div>'
            f'</div>'
            f'<div style="font-size:0.82rem;font-weight:700;color:{bar_color}">{prob * 100:.0f}%</div>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div style="border:1px solid #d7eadc;border-radius:18px;padding:1rem 1.1rem;
                    background:linear-gradient(135deg,#f9fff9,#f2fbf5);margin:0.8rem 0 1rem;">
            <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:220px;">
                    <div style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;color:#5e7d6d;">
                        Market Weather
                    </div>
                    <div style="font-size:1.35rem;font-weight:800;color:{color};margin-top:0.2rem;">
                        {emoji} {label}
                    </div>
                    <div style="font-size:0.86rem;color:#466355;margin-top:0.35rem;line-height:1.45;">
                        {copy}
                    </div>
                </div>
                <div style="display:flex;gap:0.7rem;flex-wrap:wrap;align-items:stretch;">
                    <div style="min-width:120px;padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d7eadc;">
                        <div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Confidence</div>
                        <div style="font-size:1.1rem;font-weight:800;color:#123524;">{confidence * 100:.0f}%</div>
                    </div>
                    <div style="min-width:120px;padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d7eadc;">
                        <div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Stock Trust</div>
                        <div style="font-size:1.1rem;font-weight:800;color:#123524;">{alpha:.0%}</div>
                    </div>
                    <div style="min-width:120px;padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d7eadc;">
                        <div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Momentum Weight</div>
                        <div style="font-size:1.1rem;font-weight:800;color:#123524;">{1 - beta:.0%}</div>
                    </div>
                    <div style="min-width:120px;padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d7eadc;">
                        <div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Short-Term Pulse</div>
                        <div style="font-size:1.1rem;font-weight:800;color:#123524;">{str(fast_dominant).title()}</div>
                    </div>
                    <div style="min-width:120px;padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d7eadc;">
                        <div style="font-size:0.62rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Next 10D</div>
                        <div style="font-size:1.05rem;font-weight:800;color:#123524;">{forecast_next}</div>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-top:0.9rem;">
                {''.join(bars)}
            </div>
            <div style="margin-top:0.85rem;font-size:0.8rem;color:#527364;">
                Basket health:
                {basket.get('above_200_pct', 50):.0f}% above long trend,
                {basket.get('participation_pct', 50):.0f}% still participating,
                average pair correlation {basket.get('avg_corr_63', 0.35):.2f}
                {f", tactical focus {int(primary_horizon)}d." if primary_horizon is not None else "."}
                Bear-shift risk {forecast_shift * 100:.0f}%.
            </div>
            <div style="margin-top:0.35rem;font-size:0.78rem;color:#527364;">{forecast_copy}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_noob_portfolio_digest(tickers, weights, ann_ret, ann_risk):
    """Give the user a quick plain-English summary of what this mix implies."""
    if tickers is None or weights is None:
        return

    pairs = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
    top_weight = float(pairs[0][1]) if pairs else 0.0
    top3_share = float(sum(w for _, w in pairs[:3]))
    active_slots = sum(1 for _, w in pairs if w >= 0.05)
    _, ride_label, ride_copy = _noob_ride_profile(float(ann_risk))

    if top3_share < 0.45:
        mix_shape = "well spread out"
    elif top3_share < 0.65:
        mix_shape = "moderately concentrated"
    else:
        mix_shape = "quite concentrated"

    base_amount = 100000.0
    one_year = base_amount * (1.0 + float(ann_ret))
    five_year = base_amount * ((1.0 + float(ann_ret)) ** 5) if ann_ret > -0.95 else np.nan

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                    gap:0.75rem;margin:0.85rem 0 0.95rem;">
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#fff;border:1px solid #ddebe1;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Mix Shape</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{mix_shape.title()}</div>
                <div style="font-size:0.78rem;color:#577465;margin-top:0.28rem;">Top 3 stocks hold {top3_share * 100:.0f}% of the portfolio.</div>
            </div>
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#fff;border:1px solid #ddebe1;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Largest Position</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{pairs[0][0] if pairs else '—'}</div>
                <div style="font-size:0.78rem;color:#577465;margin-top:0.28rem;">It carries about {top_weight * 100:.0f}% of your money.</div>
            </div>
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#fff;border:1px solid #ddebe1;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Ride Profile</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{ride_label}</div>
                <div style="font-size:0.78rem;color:#577465;margin-top:0.28rem;">{ride_copy}</div>
            </div>
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#fff;border:1px solid #ddebe1;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">₹1L Illustration</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">₹{one_year:,.0f} in 1y</div>
                <div style="font-size:0.78rem;color:#577465;margin-top:0.28rem;">~₹{five_year:,.0f} in 5y if the same average return repeated.</div>
            </div>
        </div>
        <div class="nb-info-box">
            This portfolio uses {active_slots} meaningful slots. More slots usually means better diversification,
            but only if those stocks do not all move together.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_noob_stock_summary(tickers, asset_returns_series, asset_risks_series):
    """Quick plain-English stock leaderboard for the noob stock tab."""
    if asset_returns_series is None or asset_risks_series is None or len(tickers) == 0:
        return

    rows = []
    for ticker in tickers:
        rows.append({
            "ticker": ticker,
            "ret": float(asset_returns_series.get(ticker, 0.0)),
            "risk": float(asset_risks_series.get(ticker, 0.0)),
        })
    if not rows:
        return

    best = max(rows, key=lambda x: x["ret"])
    smoothest = min(rows, key=lambda x: x["risk"])
    wildest = max(rows, key=lambda x: x["risk"])

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                    gap:0.75rem;margin:0.4rem 0 1rem;">
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#f9fff9;border:1px solid #d7eadc;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Top Grower</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{best['ticker']}</div>
                <div style="font-size:0.82rem;color:#577465;margin-top:0.28rem;">About {best['ret'] * 100:+.1f}% a year in the selected history.</div>
            </div>
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#f9fff9;border:1px solid #d7eadc;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Smoothest Ride</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{smoothest['ticker']}</div>
                <div style="font-size:0.82rem;color:#577465;margin-top:0.28rem;">This one bounced around the least in the sample window.</div>
            </div>
            <div style="padding:0.9rem 1rem;border-radius:16px;background:#f9fff9;border:1px solid #d7eadc;">
                <div style="font-size:0.64rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Wild Card</div>
                <div style="font-size:1.02rem;font-weight:800;color:#123524;margin-top:0.22rem;">{wildest['ticker']}</div>
                <div style="font-size:0.82rem;color:#577465;margin-top:0.28rem;">This stock had the bumpiest ride, so position sizing matters more.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — PIGGY BANK SIMULATOR
#  Main landing feature: "I invested ₹X, it became ₹Y"
# ══════════════════════════════════════════════════════════════

def render_noob_piggy_bank(weights, cumulative_returns_df, key_prefix: str = "nb"):
    """
    Takes the best portfolio weights and a cumulative returns DataFrame
    (index = dates, one column per portfolio type or a single 'portfolio' column).
    Shows a big friendly piggy-bank growth card with an animated bar.

    cumulative_returns_df should be indexed by date with values as multipliers
    (e.g. 1.0 = start, 1.35 = 35% growth). If it's a multi-col df, we use the
    first column (caller should pass the best/tangency portfolio column).

    key_prefix: unique string to namespace widget keys — avoids duplicate-key
    errors when this function is called more than once per page render.
    """
    st.markdown('<div class="nb-section">💰 Your Money Story</div>', unsafe_allow_html=True)

    # ── Input controls ────────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        invest_amount = st.number_input(
            "💵 How much would you invest? (₹)",
            min_value=1000,
            max_value=10_000_000,
            value=100_000,
            step=5_000,
            key=f"{key_prefix}_invest_amount",
            help="This is a simulation — no real money involved!"
        )
    with c2:
        # Let them pick a label that maps to an actual multiplier
        friendly_options = {
            "From day one (full history)": "full",
            "Last 3 years": "3y",
            "Last 1 year": "1y",
        }
        period_label = st.selectbox(
            "⏱ Over what period?",
            options=list(friendly_options.keys()),
            key=f"{key_prefix}_invest_period"
        )
        period_key = friendly_options[period_label]

    # ── Compute growth ────────────────────────────────────────
    if cumulative_returns_df is None or cumulative_returns_df.empty:
        st.markdown(
            '<div class="nb-info-box">Hit <strong>🚀 Run Magic!</strong> first and come back here '
            'to see your money grow! 🌱</div>',
            unsafe_allow_html=True
        )
        return

    # Work with the first (or only) column as the portfolio curve
    series = cumulative_returns_df.iloc[:, 0] if hasattr(cumulative_returns_df, 'iloc') else cumulative_returns_df

    # Slice based on period
    try:
        if period_key == "3y":
            cutoff = series.index[-1] - pd.DateOffset(years=3)
            sliced = series[series.index >= cutoff]
        elif period_key == "1y":
            cutoff = series.index[-1] - pd.DateOffset(years=1)
            sliced = series[series.index >= cutoff]
        else:
            sliced = series

        if sliced.empty:
            sliced = series  # fallback to full history

        # Growth multiplier: end / start
        multiplier = float(sliced.iloc[-1]) / float(sliced.iloc[0])
        final_value = invest_amount * multiplier
        gain = final_value - invest_amount
        gain_pct = (multiplier - 1) * 100
        bar_pct = min(max(multiplier - 1, 0) * 40, 100)  # visual scale
        years = max((sliced.index[-1] - sliced.index[0]).days / 365.25, 0.01)
        cagr_pct = ((multiplier ** (1 / years)) - 1) * 100 if multiplier > 0 else -100.0

    except Exception:
        st.warning("Could not compute growth. Please run the optimizer first.")
        return

    # ── Emoji badge based on return ───────────────────────────
    if gain_pct >= 50:
        emoji, badge_class, verdict = "🚀", "nb-ret-rocket", "Rocket fuel!"
    elif gain_pct >= 15:
        emoji, badge_class, verdict = "📈", "nb-ret-ok", "Pretty good!"
    elif gain_pct >= 0:
        emoji, badge_class, verdict = "😊", "nb-ret-meh", "Slow and steady"
    else:
        emoji, badge_class, verdict = "😬", "nb-ret-oops", "Rough patch"

    gain_sign = "+" if gain >= 0 else ""
    gain_color_class = "up" if gain >= 0 else "down"
    bar_color = "#00e676" if gain >= 0 else "#ff5252"

    st.markdown(
        f"""
        <div class="nb-piggy-shell">
            <div class="nb-piggy-label">If you invested</div>
            <div class="nb-piggy-number">₹{invest_amount:,.0f}</div>
            <div class="nb-piggy-label" style="margin-top:1rem">It would now be worth</div>
            <div class="nb-piggy-number">₹{final_value:,.0f}</div>
            <div class="nb-piggy-gain {gain_color_class}">
                {gain_sign}₹{gain:,.0f} &nbsp;·&nbsp; {gain_sign}{gain_pct:.1f}%
            </div>
            <div style="font-size:0.78rem;color:#4a7060;margin-top:0.3rem">
                That works out to roughly <strong>{cagr_pct:.1f}% a year</strong> over {years:.1f} years.
            </div>
            <div class="nb-piggy-bar-bg">
                <div class="nb-piggy-bar-fill"
                     style="width:{bar_pct:.1f}%;background:linear-gradient(90deg,
                     {'#00c853,#00e676,#69f0ae' if gain >= 0 else '#d32f2f,#ff5252,#ff867c'});">
                </div>
            </div>
            <div style="margin-top:0.6rem">
                <span class="nb-ret-badge {badge_class}">{emoji} {verdict}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Monthly scoreboard ────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
        'color:#2e7d52;margin:0.8rem 0 0.3rem">Monthly Report Card</div>',
        unsafe_allow_html=True
    )

    # Resample the sliced series to monthly returns
    try:
        monthly = sliced.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            monthly = sliced.resample("M").last().pct_change().dropna()

        # Show last 24 months max
        monthly_display = monthly.tail(24)
        if not monthly.empty:
            best_month = float(monthly.max() * 100)
            worst_month = float(monthly.min() * 100)
            win_rate = float((monthly > 0).mean() * 100)
            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                            gap:0.65rem;margin:0.65rem 0 0.85rem;">
                    <div style="padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d9ebde;">
                        <div style="font-size:0.6rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Winning Months</div>
                        <div style="font-size:1rem;font-weight:800;color:#123524;margin-top:0.2rem;">{win_rate:.0f}%</div>
                    </div>
                    <div style="padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d9ebde;">
                        <div style="font-size:0.6rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Best Month</div>
                        <div style="font-size:1rem;font-weight:800;color:#0b8f4d;margin-top:0.2rem;">+{best_month:.1f}%</div>
                    </div>
                    <div style="padding:0.7rem 0.85rem;border-radius:14px;background:#fff;border:1px solid #d9ebde;">
                        <div style="font-size:0.6rem;color:#6f8c7d;text-transform:uppercase;letter-spacing:0.08em;">Worst Month</div>
                        <div style="font-size:1rem;font-weight:800;color:#c0392b;margin-top:0.2rem;">{worst_month:.1f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        cells = ""
        for dt, ret in monthly_display.items():
            cls = "nb-month-win" if ret >= 0 else "nb-month-loss"
            sign = "+" if ret >= 0 else ""
            mo_label = dt.strftime("%b '%y")
            cells += (
                f'<div class="nb-month-cell {cls}">'
                f'<div class="mo">{mo_label}</div>'
                f'<div class="ret">{sign}{ret*100:.1f}%</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="nb-month-grid">{cells}</div>',
            unsafe_allow_html=True
        )
    except Exception:
        pass  # Monthly scoreboard is a bonus — fail silently


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — RECOMMENDED PORTFOLIO ("Best Balanced Pick")
#  Wraps the tangency portfolio in plain-English packaging
# ══════════════════════════════════════════════════════════════

def render_noob_recommended_pick(tickers, weights, ann_ret, ann_risk, sharpe):
    """
    Shows the tangency (best Sharpe) portfolio as a plain-English
    "recommended pick" card — no Sharpe ratio label in sight.
    Accepts the same data render_portfolio_spotlight uses.
    """
    st.markdown('<div class="nb-section">🏆 Our Best Pick For You</div>', unsafe_allow_html=True)

    # ── Volatility plain-English label ───────────────────────
    ride_emoji, ride_label, ride_copy = _noob_ride_profile(float(ann_risk))
    ride_class = (
        "nb-ride-smooth" if ann_risk < 0.10 else
        ("nb-ride-bumpy" if ann_risk < 0.20 else "nb-ride-wild")
    )

    # ── Return plain-English framing ─────────────────────────
    ret_pct = ann_ret * 100
    if ret_pct >= 25:
        ret_copy = f"Could grow by about {ret_pct:.0f}% a year on average. That's seriously impressive."
    elif ret_pct >= 10:
        ret_copy = f"Around {ret_pct:.0f}% growth per year on average. Solid!"
    elif ret_pct >= 0:
        ret_copy = f"About {ret_pct:.0f}% per year. Modest, but better than cash under the mattress."
    else:
        ret_copy = f"Historical returns were negative ({ret_pct:.1f}%). Consider tweaking your stock picks."

    # ── Top 3 holdings ────────────────────────────────────────
    top3 = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)[:3]
    top3_html = " &nbsp;·&nbsp; ".join(
        [f'<span class="ticker-chip">{t}</span> {w*100:.0f}%' for t, w in top3]
    )

    # ── Quality score (remapped Sharpe, 1-10 scale) ───────────
    # Sharpe 0→score 1, Sharpe 3→score 10; clamp to [1,10]
    quality_score = max(1, min(10, round(1 + (sharpe / 3) * 9)))
    concentration = sum(sorted(weights, reverse=True)[:3]) * 100
    rupee_year = 100000 * (1 + ann_ret)

    st.markdown(
        f"""
        <div class="nb-pick-shell">
            <span class="nb-pick-crown">🏆</span>
            <div class="nb-pick-title">Best Balanced Pick</div>
            <div class="nb-pick-copy">
                The optimizer tested thousands of combinations and found this one hits the
                sweetest spot between growth and stability.<br>
                <strong>Top holdings:</strong> {top3_html}
            </div>
            <div class="nb-pick-stats">
                <div class="nb-pick-stat">
                    <div class="nb-pick-stat-label">Yearly growth</div>
                    <div class="nb-pick-stat-value">{ret_pct:.1f}%</div>
                </div>
                <div class="nb-pick-stat">
                    <div class="nb-pick-stat-label">Quality score</div>
                    <div class="nb-pick-stat-value">{quality_score}/10</div>
                </div>
                <div class="nb-pick-stat">
                    <div class="nb-pick-stat-label">Ride type</div>
                    <div class="nb-pick-stat-value">{ride_emoji}</div>
                </div>
            </div>
            <div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.85rem;">
                <div style="padding:0.55rem 0.75rem;border-radius:12px;background:rgba(255,255,255,0.14);">
                    Top 3 concentration: <strong>{concentration:.0f}%</strong>
                </div>
                <div style="padding:0.55rem 0.75rem;border-radius:12px;background:rgba(255,255,255,0.14);">
                    ₹1L illustration: <strong>₹{rupee_year:,.0f}</strong> in 1 year
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f'<span class="nb-ride-badge {ride_class}">{ride_emoji} {ride_label}</span>'
        f'<div style="font-size:0.78rem;color:#4a7060;margin:0.3rem 0 0.8rem">{ride_copy}<br>'
        f'<span style="color:#3a6050">{ret_copy}</span></div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — SLOT MACHINE WEIGHTS
#  Portfolio weights shown as a colourful bar chart without numbers
# ══════════════════════════════════════════════════════════════

def render_noob_slot_weights(tickers, weights, title="Your Portfolio Mix 🎰"):
    """
    Renders portfolio allocation as a slot-machine style bar list.
    Each stock gets a candy-coloured bar proportional to its weight.
    The frame calls it "the mix" — never "weights" or "allocation".
    """
    st.markdown(f'<div class="nb-section">{title}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="nb-info-box">The wider the bar, the more of your money goes into that stock. '
        'The optimizer chose this mix to give you the best balance of growth and stability.</div>',
        unsafe_allow_html=True
    )

    # Sort by weight descending so big winners are at the top
    pairs = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)

    # Colour gradient cycling through purple shades for visual fun
    bar_colors = [
        "linear-gradient(90deg,#7c4dff,#b388ff)",
        "linear-gradient(90deg,#536dfe,#82b1ff)",
        "linear-gradient(90deg,#00b0ff,#80d8ff)",
        "linear-gradient(90deg,#00bfa5,#a7ffeb)",
        "linear-gradient(90deg,#64dd17,#ccff90)",
        "linear-gradient(90deg,#ffab00,#ffe57f)",
        "linear-gradient(90deg,#ff6d00,#ffd180)",
        "linear-gradient(90deg,#e040fb,#ea80fc)",
    ]

    rows_html = ""
    for i, (ticker, weight) in enumerate(pairs):
        pct = weight * 100
        color = bar_colors[i % len(bar_colors)]
        # Only show stocks with meaningful allocation (>0.5%)
        if pct < 0.5:
            continue
        rows_html += (
            f'<div class="nb-slot-row">'
            f'<span class="nb-slot-ticker">{ticker}</span>'
            f'<div class="nb-slot-bar-bg">'
            f'<div class="nb-slot-bar-fill" style="width:{min(pct*2.5, 100):.1f}%;background:{color}"></div>'
            f'</div>'
            f'<span class="nb-slot-pct">{pct:.0f}%</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="nb-slot-shell">{rows_html}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — STOCK FIGHT CLUB
#  Head-to-head return comparison between two stocks
# ══════════════════════════════════════════════════════════════

def render_noob_fight_club(tickers, asset_returns_series, asset_risks_series):
    """
    Lets the user pick two stocks and see a VS card.
    asset_returns_series: pd.Series indexed by ticker, values = annualised returns (decimal)
    asset_risks_series:   pd.Series indexed by ticker, values = annualised risk (decimal)
    """
    st.markdown('<div class="nb-section">🥊 Stock Fight Club</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nb-info-box">Pick any two stocks from your basket and we\'ll show you '
        'which one would have made you more money. <strong>Winner gets the crown. 👑</strong></div>',
        unsafe_allow_html=True
    )

    if len(tickers) < 2:
        st.warning("Add at least 2 stocks to your basket to use Fight Club.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fighter_a = st.selectbox("🔴 Choose challenger A", options=tickers, key="nb_fight_a")
    with c2:
        # Default to second ticker so they don't start on the same pick
        default_idx = 1 if len(tickers) > 1 else 0
        fighter_b = st.selectbox("🔵 Choose challenger B", options=tickers,
                                  index=default_idx, key="nb_fight_b")

    if fighter_a == fighter_b:
        st.info("Pick two different stocks for a real fight!")
        return

    # ── Pull the numbers ──────────────────────────────────────
    ret_a = float(asset_returns_series.get(fighter_a, 0)) * 100
    ret_b = float(asset_returns_series.get(fighter_b, 0)) * 100
    risk_a = float(asset_risks_series.get(fighter_a, 0)) * 100
    risk_b = float(asset_risks_series.get(fighter_b, 0)) * 100

    # Determine winner
    if ret_a > ret_b:
        winner, loser = fighter_a, fighter_b
        a_cls, b_cls = "winner", "loser"
        crown_a, crown_b = "👑", ""
    elif ret_b > ret_a:
        winner, loser = fighter_b, fighter_a
        a_cls, b_cls = "loser", "winner"
        crown_a, crown_b = "", "👑"
    else:
        a_cls, b_cls = "draw", "draw"
        crown_a, crown_b = "🤝", "🤝"

    # Return display class
    ret_a_cls = "up" if ret_a >= 0 else "down"
    ret_b_cls = "up" if ret_b >= 0 else "down"

    # Risk plain-English
    def _ride_label(risk_pct):
        if risk_pct < 10:
            return "😌 Smooth"
        elif risk_pct < 20:
            return "✈️ Bumpy"
        else:
            return "🎢 Wild"

    st.markdown(
        f"""
        <div class="nb-fight-shell">
            <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;
                        color:#4a3a80;margin-bottom:0.6rem">Annual returns · historical average</div>
            <div class="nb-vs-grid">
                <div class="nb-fight-card {a_cls}">
                    <span class="nb-crown">{crown_a}</span>
                    <div class="nb-fight-ticker">{fighter_a}</div>
                    <div class="nb-fight-return {ret_a_cls}">{ret_a:+.1f}%</div>
                    <div class="nb-fight-label">per year</div>
                    <div class="nb-fight-label" style="margin-top:0.4rem">{_ride_label(risk_a)}</div>
                </div>
                <div class="nb-vs-badge">VS</div>
                <div class="nb-fight-card {b_cls}">
                    <span class="nb-crown">{crown_b}</span>
                    <div class="nb-fight-ticker">{fighter_b}</div>
                    <div class="nb-fight-return {ret_b_cls}">{ret_b:+.1f}%</div>
                    <div class="nb-fight-label">per year</div>
                    <div class="nb-fight-label" style="margin-top:0.4rem">{_ride_label(risk_b)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Plain-English verdict ─────────────────────────────────
    if a_cls == "winner":
        diff = abs(ret_a - ret_b)
        st.markdown(
            f'<div class="nb-info-box"><strong>{fighter_a}</strong> beats <strong>{fighter_b}</strong> by '
            f'{diff:.1f}% per year on average. That adds up fast over many years! 📈</div>',
            unsafe_allow_html=True
        )
    elif b_cls == "winner":
        diff = abs(ret_b - ret_a)
        st.markdown(
            f'<div class="nb-info-box"><strong>{fighter_b}</strong> beats <strong>{fighter_a}</strong> by '
            f'{diff:.1f}% per year on average. Compound that over 10 years — huge difference! 📈</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="nb-info-box">These two are neck and neck! In this case, the smoother ride '
            'is probably the better choice long-term.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — TIME MACHINE
#  "What if I invested on my birthday / a famous date?"
# ══════════════════════════════════════════════════════════════

def render_noob_time_machine(tickers, price_data_df):
    """
    Lets the user pick a start date and an amount, then shows
    what the tangency-weighted portfolio would be worth today.

    price_data_df: adjusted close price DataFrame, indexed by date, columns = tickers.
    """
    st.markdown('<div class="nb-section">⏰ The Time Machine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nb-info-box">Ever wonder: <strong>"What if I had invested on my birthday?"</strong> '
        'Now you can find out. Pick a date and an amount and we\'ll show you what happened.</div>',
        unsafe_allow_html=True
    )

    if price_data_df is None or price_data_df.empty:
        st.markdown(
            '<div class="nb-info-box">Load your data first and come back here to time travel! ⏳</div>',
            unsafe_allow_html=True
        )
        return

    # Date range from available data
    min_date = price_data_df.index[0].date()
    max_date = price_data_df.index[-1].date()

    c1, c2 = st.columns(2)
    with c1:
        invest_date = st.date_input(
            "📅 Pick your investment date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="nb_tm_date",
            help="Try your birthday, a holiday, or a famous market crash date!"
        )
    with c2:
        invest_amount_tm = st.number_input(
            "💵 How much? (₹)",
            min_value=1_000,
            max_value=10_000_000,
            value=50_000,
            step=5_000,
            key="nb_tm_amount"
        )

    # ── Famous date suggestions ───────────────────────────────
    st.caption("💡 Tip: Try March 2020 (COVID crash), Jan 2021 (post-crash boom), or any date that means something to you!")

    # ── Compute the return ────────────────────────────────────
    try:
        invest_dt = pd.Timestamp(invest_date)

        # Get price on or after the chosen date
        future_prices = price_data_df[price_data_df.index >= invest_dt]
        if future_prices.empty:
            st.warning("No data available from that date. Try a more recent date.")
            return

        start_prices = future_prices.iloc[0]
        end_prices   = price_data_df.iloc[-1]
        end_date_str = price_data_df.index[-1].strftime("%d %b %Y")
        actual_start_str = future_prices.index[0].strftime("%d %b %Y")

        # Equal-weight fallback (noob mode doesn't burden user with choosing a strategy)
        # But if weights available from session state, use them
        ss = st.session_state
        if ss.get("weights_tan") is not None and len(ss.weights_tan) == len(tickers):
            w = ss.weights_tan
        else:
            w = np.ones(len(tickers)) / len(tickers)

        # Portfolio return = weighted sum of individual returns
        available = [t for t in tickers if t in start_prices.index and t in end_prices.index]
        if not available:
            st.warning("Not enough overlapping tickers for the chosen date range.")
            return

        ticker_returns = []
        for t in available:
            s = float(start_prices.get(t, np.nan))
            e = float(end_prices.get(t, np.nan))
            if np.isnan(s) or np.isnan(e) or s <= 0:
                ticker_returns.append(0.0)
            else:
                ticker_returns.append((e - s) / s)

        # Reweight to only available tickers
        w_avail = np.array([w[tickers.index(t)] for t in available])
        w_avail = w_avail / w_avail.sum()  # renormalise
        portfolio_return = float(np.dot(w_avail, ticker_returns))

        final_value = invest_amount_tm * (1 + portfolio_return)
        gain = final_value - invest_amount_tm
        gain_pct = portfolio_return * 100
        years = max((price_data_df.index[-1] - future_prices.index[0]).days / 365.25, 0.01)

        # Emoji verdict
        if gain_pct >= 100:
            verdict_emoji, verdict_text = "🚀🚀🚀", "You would have DOUBLED your money!"
        elif gain_pct >= 50:
            verdict_emoji, verdict_text = "🚀", "Excellent result!"
        elif gain_pct >= 20:
            verdict_emoji, verdict_text = "📈", "Pretty solid!"
        elif gain_pct >= 0:
            verdict_emoji, verdict_text = "😊", "Modest but positive."
        else:
            verdict_emoji, verdict_text = "😬", "Rough patch — markets recover though!"

        gain_sign = "+" if gain >= 0 else ""
        gain_color = "#00e676" if gain >= 0 else "#ff5252"

        st.markdown(
            f"""
            <div class="nb-timemachine-shell">
                <div style="font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;
                            color:#3db85a;margin-bottom:0.4rem">
                    Invested on {actual_start_str} · Worth on {end_date_str} · {years:.1f} years
                </div>
                <div class="nb-tm-result-shell">
                    <div style="font-size:0.8rem;color:#3db85a;margin-bottom:0.5rem">
                        ₹{invest_amount_tm:,.0f} invested became...
                    </div>
                    <div class="nb-tm-headline">₹{final_value:,.0f}</div>
                    <div class="nb-tm-sub" style="color:{gain_color}">
                        {gain_sign}₹{gain:,.0f} &nbsp;·&nbsp; {gain_sign}{gain_pct:.1f}%
                    </div>
                    <div style="margin-top:0.8rem;font-size:1.4rem">{verdict_emoji}</div>
                    <div style="font-size:0.88rem;color:#3db85a;margin-top:0.2rem">{verdict_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.warning(f"Time machine glitch: {e}. Try a different date!")


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — PLAIN-ENGLISH RISK GAUGE
#  Per-stock volatility shown as ride-smoothness labels
# ══════════════════════════════════════════════════════════════

def render_noob_risk_gauge(tickers, asset_risks_series, asset_returns_series):
    """
    Shows each stock's volatility as a plain-English ride label
    instead of the σ number. Bar length = relative roughness.
    asset_risks_series: pd.Series, index=ticker, values=annualised volatility (decimal)
    """
    st.markdown('<div class="nb-section">🎢 How Bumpy Is Each Stock?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nb-info-box">'
        'Every stock has its own "ride type". Smooth stocks barely move day to day. '
        'Wild ones can make big jumps — up <em>and</em> down. '
        '<strong>Neither is better</strong> — it depends on your stomach! 😄'
        '</div>',
        unsafe_allow_html=True
    )

    if asset_risks_series is None or len(asset_risks_series) == 0:
        st.info("Load data and run the optimizer to see the risk breakdown.")
        return

    # Sort by risk descending — wildest at top
    pairs = sorted(
        [(t, float(asset_risks_series.get(t, 0)), float(asset_returns_series.get(t, 0)))
         for t in tickers],
        key=lambda x: x[1], reverse=True
    )

    max_risk = max(p[1] for p in pairs) or 1.0
    rows_html = ""
    for ticker, risk, ret in pairs:
        bar_pct = (risk / max_risk) * 100
        risk_pct = risk * 100
        ret_pct  = ret * 100

        if risk_pct < 10:
            label, css = "😌 Smooth ride", "nb-risk-smooth"
            bar_color = "linear-gradient(90deg,#00c853,#00e676)"
        elif risk_pct < 20:
            label, css = "✈️ Some turbulence", "nb-risk-bumpy"
            bar_color = "linear-gradient(90deg,#ffab00,#ffd54f)"
        else:
            label, css = "🎢 Hold on tight", "nb-risk-wild"
            bar_color = "linear-gradient(90deg,#e53935,#ff5252)"

        ret_sign = "+" if ret_pct >= 0 else ""
        ret_color = "#00e676" if ret_pct >= 0 else "#ff5252"
        rows_html += (
            f'<div class="nb-risk-row">'
            f'<span class="nb-risk-name">{ticker} '
            f'<span style="color:{ret_color};font-size:0.68rem">{ret_sign}{ret_pct:.0f}%/yr</span></span>'
            f'<div class="nb-risk-bar-bg">'
            f'<div class="nb-risk-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>'
            f'</div>'
            f'<span class="nb-risk-label {css}">{label}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="nb-slot-shell" style="border-color:rgba(255,180,0,0.18)">'
        f'<div class="nb-risk-gauge">{rows_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  NOOB MODE — TAB RENDERERS
#  One function per tab. main.py calls these when app_mode=="noob"
# ══════════════════════════════════════════════════════════════

def render_noob_tab_overview():
    """
    Noob replacement for the Overview / hero tab.
    Shows the hero banner, how-it-works steps, and the recommended pick
    if portfolios are already computed.
    """
    render_noob_hero()
    render_noob_how_it_works()

    ss = st.session_state
    if ss.get("weights_tan") is not None and ss.get("tan_return") is not None:
        render_noob_recommended_pick(
            tickers=ss.tickers,
            weights=ss.weights_tan,
            ann_ret=ss.tan_return,
            ann_risk=ss.tan_risk,
            sharpe=ss.tan_sharpe,
        )
    else:
        st.markdown(
            '<div class="nb-info-box" style="margin-top:1rem">'
            '🏆 <strong>Your recommended pick will appear here</strong> once you load data '
            'and hit <strong>🚀 Run Magic!</strong> in the sidebar!'
            '</div>',
            unsafe_allow_html=True
        )


def render_noob_tab_my_money(cumulative_returns_df):
    """
    Noob replacement for the Backtest / Returns tab.
    Piggy bank simulator + time machine.

    cumulative_returns_df: DataFrame of cumulative portfolio returns
    (index=dates, columns=portfolio types, values = multipliers starting at 1.0).
    Pass None if not computed yet.
    """
    st.markdown(NOOB_CSS, unsafe_allow_html=True)

    ss = st.session_state
    render_noob_market_weather_card()

    # ── Piggy bank ────────────────────────────────────────────
    render_noob_piggy_bank(
        weights=ss.get("weights_tan"),
        cumulative_returns_df=cumulative_returns_df,
        key_prefix="nb_money_tab",
    )

    _bt = ss.get("bt_results") or {}
    _deploy = _bt.get("deployable") or {}
    if _deploy:
        _deploy_weights = _deploy.get("weights") or {}
        _pairs = sorted(_deploy_weights.items(), key=lambda x: x[1], reverse=True)
        _top = [pair for pair in _pairs if pair[1] > 1e-4][:5]
        _cash = float(_deploy.get("cash_weight", 0.0) or 0.0) * 100
        _posture = str(_deploy.get("posture", "Balanced"))
        _conf = float(_deploy.get("confidence", 0.0) or 0.0)
        _trust = float(_deploy.get("trust_score", 0.0) or 0.0)
        _when = str(_deploy.get("date", "today"))
        st.markdown('<div class="nb-section">What The Backtest Would Invest Today</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="nb-piggy-shell">
                <div class="nb-piggy-label">Latest deploy-now plan</div>
                <div class="nb-hero-title" style="font-size:2rem;margin-bottom:0.2rem">{html.escape(_posture)}</div>
                <div class="nb-hero-sub" style="margin-bottom:0.7rem">
                    Based on the latest walk-forward rebalance on <strong>{html.escape(_when)}</strong>, the engine would deploy money like this right now.
                </div>
                <div class="nb-hero-chips">
                    <span class="nb-chip nb-chip-green">Confidence {_conf:.0f}/100</span>
                    <span class="nb-chip nb-chip-green">Trust {_trust:.0f}/100</span>
                    <span class="nb-chip nb-chip-amber">Cash {_cash:.1f}%</span>
                    <span class="nb-chip nb-chip-blue">Strategic {html.escape(str(_deploy.get('strategic_regime', 'unknown')).title())}</span>
                    <span class="nb-chip nb-chip-pink">Tactical {html.escape(str(_deploy.get('tactical_regime', 'unknown')).title())}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if _top:
            render_noob_slot_weights(
                [ticker for ticker, _ in _top],
                [weight for _, weight in _top],
                title="Today's Suggested Mix 🧭",
            )
        _cmp = _deploy.get("comparison")
        if isinstance(_cmp, pd.DataFrame) and not _cmp.empty:
            st.markdown("**Biggest Changes vs your original mix**")
            _show = _cmp[["Deploy Now %", "Delta vs Original pp"]].copy().sort_values("Delta vs Original pp", ascending=False)
            st.dataframe(
                _show.head(5).style.format({
                    "Deploy Now %": "{:.2f}%",
                    "Delta vs Original pp": "{:+.2f} pp",
                }),
                use_container_width=True,
                height=min(260, 60 + 35 * len(_show.head(5))),
            )
        _bands = _deploy.get("confidence_bands")
        if isinstance(_bands, pd.DataFrame) and not _bands.empty:
            st.markdown("**Comfort zone for the top allocations**")
            st.dataframe(
                _bands.sort_values("Deploy Now %", ascending=False).head(5).style.format({
                    "Lower %": "{:.2f}%",
                    "Deploy Now %": "{:.2f}%",
                    "Upper %": "{:.2f}%",
                }),
                use_container_width=True,
                height=min(260, 60 + 35 * len(_bands.head(5))),
            )
        _why = _deploy.get("explanations")
        if isinstance(_why, pd.DataFrame) and not _why.empty:
            st.markdown("**Why some stocks were increased or cut**")
            st.dataframe(
                _why.head(4),
                use_container_width=True,
                height=min(240, 60 + 35 * len(_why.head(4))),
            )
        _cal = _deploy.get("calibration")
        if isinstance(_cal, pd.DataFrame) and not _cal.empty:
            st.markdown("**How this plan behaved in the latest mini-tests**")
            st.dataframe(
                _cal[["Date", "Posture", "Deploy Return %", "Benchmark Return %", "Winner"]].head(4).style.format({
                    "Deploy Return %": "{:+.2f}%",
                    "Benchmark Return %": "{:+.2f}%",
                }),
                use_container_width=True,
                height=min(240, 60 + 35 * len(_cal.head(4))),
            )

    # ── Time machine ─────────────────────────────────────────
    # Use explicit None checks — `or` on a DataFrame calls bool() which pandas forbids
    _pd = ss.get("price_data")
    _pp = ss.get("prices")
    _pr = ss.get("raw_prices")
    price_df = _pd if _pd is not None else (_pp if _pp is not None else _pr)
    render_noob_time_machine(
        tickers=ss.get("tickers", []),
        price_data_df=price_df,
    )


def render_noob_tab_my_stocks():
    """
    Noob replacement for the Asset Analysis tab.
    Shows fight club + risk gauge side by side.
    """
    st.markdown(NOOB_CSS, unsafe_allow_html=True)

    ss = st.session_state
    tickers = ss.get("tickers", [])
    asset_risks = ss.get("asset_risks")
    asset_returns = ss.get("asset_returns")

    if not ss.get("data_loaded") or asset_risks is None:
        st.markdown(
            '<div class="nb-info-box">Load your data first — hit <strong>📡 Load Data</strong> '
            'in the sidebar or the Dashboard tab to unlock the stock breakdown!</div>',
            unsafe_allow_html=True
        )
        return

    render_noob_market_weather_card()
    render_noob_stock_summary(tickers, asset_returns, asset_risks)
    render_noob_fight_club(tickers, asset_returns, asset_risks)
    st.markdown("---")
    render_noob_risk_gauge(tickers, asset_risks, asset_returns)


def render_noob_tab_the_mix():
    """
    Noob replacement for the Optimizer / Weights tab.
    Shows slot-machine weight bars for the best portfolio.
    """
    st.markdown(NOOB_CSS, unsafe_allow_html=True)

    ss = st.session_state
    tickers = ss.get("tickers", [])

    # Prefer tangency (best Sharpe = "Best Pick"), fallback to utility, then min-risk
    if ss.get("weights_tan") is not None:
        weights = ss.weights_tan
        label = "Best Balanced Mix 🏆"
    elif ss.get("weights_utility") is not None:
        weights = ss.weights_utility
        label = "Max Growth Mix 🚀"
    elif ss.get("weights_min") is not None:
        weights = ss.weights_min
        label = "Smoothest Ride Mix 😌"
    else:
        st.markdown(
            '<div class="nb-info-box">Hit <strong>🚀 Run Magic!</strong> in the sidebar to see your portfolio mix!</div>',
            unsafe_allow_html=True
        )
        return

    render_noob_market_weather_card()
    if ss.get("weights_tan") is not None and ss.get("tan_return") is not None:
        render_noob_recommended_pick(
            tickers=ss.tickers,
            weights=ss.weights_tan,
            ann_ret=ss.tan_return,
            ann_risk=ss.tan_risk,
            sharpe=ss.tan_sharpe,
        )
        render_noob_portfolio_digest(
            tickers=ss.tickers,
            weights=ss.weights_tan,
            ann_ret=ss.tan_return,
            ann_risk=ss.tan_risk,
        )

    render_noob_slot_weights(tickers, weights, title=label)

    # ── Friendly explainer card ───────────────────────────────
    st.markdown(
        '<div class="nb-info-box" style="margin-top:1rem;border-left-color:#4afa7a;">'
        '🤔 <strong>Why these amounts?</strong> The optimizer looked at how each stock has '
        'behaved over years of real data — how much it grew, how much it bounced around, '
        'and how it moved relative to the other stocks. It then figured out the exact mix '
        'that gives you the best growth for the amount of drama you\'re signing up for.'
        '</div>',
        unsafe_allow_html=True
    )
