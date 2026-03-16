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
    push_bridge_symbols,
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
            st.session_state.request_load_data = True
            st.rerun()
    with c2:
        if st.button(
            "Compute Momentum" if not momentum_ready else "Refresh Momentum",
            key="hero_compute_momentum",
            use_container_width=True,
            disabled=not (momentum_enabled and data_loaded),
        ):
            st.session_state.request_compute_momentum = True
            st.rerun()
    with c3:
        if st.button("Run ALL", key="hero_run_all", use_container_width=True, disabled=not data_loaded):
            st.session_state.request_run_all = True
            st.rerun()
    with c4:
        if st.button("Frontier Only", key="hero_frontier_only", use_container_width=True, disabled=not data_loaded):
            st.session_state.request_frontier = True
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
            if st.button("Send As Replace", use_container_width=True):
                if push_bridge_symbols(symbols, "replace", source):
                    st.success("Sent to main app. Return to the main tab and refresh or click any action.")
        with h2:
            if st.button("Send As Append", use_container_width=True):
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


