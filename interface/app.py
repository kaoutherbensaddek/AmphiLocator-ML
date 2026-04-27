"""
AmphiLocator — ENSIA Amphitheatre GPS Detection System
======================================================
Run:   streamlit run app.py
Deps:  pip install streamlit pandas numpy plotly
"""

import math, hashlib, time, socket
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="AmphiLocator · ENSIA",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #09090f !important;
    color: #c9d1d9;
}

#MainMenu, header, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1320px !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── background ── */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 70% 50% at 15% 10%, rgba(88,28,220,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 55% 60% at 88% 85%, rgba(6,148,162,0.09) 0%, transparent 55%);
}

/* ── HEADER ── */
.app-header {
    padding: 2rem 0 2.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 2.5rem;
    position: relative; z-index: 10;
    display: flex; align-items: flex-end; justify-content: space-between;
}
.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; letter-spacing: -0.8px;
    color: #f0f6fc; line-height: 1;
}
.app-title span {
    background: linear-gradient(120deg, #818cf8, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: rgba(255,255,255,0.28);
    letter-spacing: 2px; text-transform: uppercase; margin-top: 6px;
}
.header-badges { display: flex; gap: 8px; align-items: center; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.5px;
    padding: 5px 14px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.4);
}
.badge.live {
    border-color: rgba(34,197,94,0.35);
    background: rgba(34,197,94,0.07);
    color: #4ade80;
}
.live-indicator {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80; display: inline-block;
    margin-right: 6px; vertical-align: middle;
    box-shadow: 0 0 8px rgba(74,222,128,0.6);
    animation: blink 2.5s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; } 50% { opacity: 0.35; }
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 4px !important; gap: 3px !important;
    margin-bottom: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: rgba(255,255,255,0.38) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important; font-weight: 500 !important;
    padding: 7px 20px !important;
    transition: all 0.18s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.7) !important;
    background: rgba(255,255,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(129,140,248,0.18) !important;
    color: #c7d2fe !important; font-weight: 600 !important;
    border: 1px solid rgba(129,140,248,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── SECTION LABEL ── */
.slabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 3px;
    color: rgba(129,140,248,0.6);
    margin-bottom: 1rem; padding-bottom: 8px;
    border-bottom: 1px solid rgba(129,140,248,0.12);
}

/* ── STATS ROW ── */
.stats-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 2.4rem;
}
.stat-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
}
.stat-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    border-radius: 14px 14px 0 0;
}
.stat-card.s-purple::after { background: linear-gradient(90deg, #818cf8, transparent); }
.stat-card.s-green::after  { background: linear-gradient(90deg, #34d399, transparent); }
.stat-card.s-cyan::after   { background: linear-gradient(90deg, #67e8f9, transparent); }
.stat-card.s-amber::after  { background: linear-gradient(90deg, #fbbf24, transparent); }
.stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700; line-height: 1; margin-bottom: 5px;
}
.stat-card.s-purple .stat-val { color: #a5b4fc; }
.stat-card.s-green  .stat-val { color: #6ee7b7; }
.stat-card.s-cyan   .stat-val { color: #a5f3fc; }
.stat-card.s-amber  .stat-val { color: #fde68a; }
.stat-key {
    font-size: 0.7rem; color: rgba(255,255,255,0.32);
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 500;
}

/* ── INPUT CARD ── */
.input-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.6rem;
    height: 100%;
}

/* ── RESULT CARD ── */
.result-card {
    border-radius: 18px; padding: 2.4rem 2rem;
    text-align: center; position: relative; overflow: hidden;
}
.result-inside {
    background: linear-gradient(160deg, rgba(4,120,87,0.22), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.35);
    box-shadow: 0 0 50px rgba(16,185,129,0.1),
                inset 0 1px 0 rgba(255,255,255,0.04);
}
.result-outside {
    background: linear-gradient(160deg, rgba(146,64,14,0.22), rgba(180,83,9,0.08));
    border: 1px solid rgba(245,158,11,0.35);
    box-shadow: 0 0 50px rgba(245,158,11,0.1),
                inset 0 1px 0 rgba(255,255,255,0.04);
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 10px;
}
.result-inside  .result-label { color: rgba(52,211,153,0.7); }
.result-outside .result-label { color: rgba(251,191,36,0.7); }
.result-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem; font-weight: 700; letter-spacing: -1.5px;
    color: #f0f6fc; line-height: 1; margin-bottom: 10px;
}
.floor-tag {
    display: inline-block; padding: 3px 16px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    font-weight: 600; margin-bottom: 16px; letter-spacing: 0.5px;
}
.floor-tag.f1 {
    background: rgba(99,102,241,0.18); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
}
.floor-tag.f2 {
    background: rgba(168,85,247,0.18); color: #d8b4fe;
    border: 1px solid rgba(168,85,247,0.3);
}
.result-meta {
    font-size: 0.78rem; color: rgba(255,255,255,0.38);
    line-height: 2; margin-top: 6px;
}
.result-meta b { color: rgba(255,255,255,0.65); font-weight: 500; }
.result-coords {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: rgba(255,255,255,0.25); margin-top: 8px;
}

/* ── CONFIDENCE BARS ── */
.conf-block { margin-top: 1.4rem; }
.conf-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.conf-label {
    font-size: 0.75rem; color: rgba(255,255,255,0.5);
    width: 68px; flex-shrink: 0; font-family: 'JetBrains Mono', monospace;
}
.conf-bar-bg {
    flex: 1; height: 5px; border-radius: 999px;
    background: rgba(255,255,255,0.06);
}
.conf-bar-fill { height: 100%; border-radius: 999px; }
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: rgba(255,255,255,0.3);
    width: 36px; text-align: right; flex-shrink: 0;
}

/* ── QUICK TEST BUTTONS ── */
.stButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.6) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    padding: 0.45rem 0.8rem !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: rgba(129,140,248,0.12) !important;
    border-color: rgba(129,140,248,0.3) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px) !important;
}

/* Primary classify button */
div[data-testid="stButton"]:has(button[kind="primary"]) > button,
.classify-btn > button {
    background: linear-gradient(135deg, #4f46e5, #6d28d9) !important;
    color: #fff !important;
    border: 1px solid rgba(129,140,248,0.3) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 24px rgba(79,70,229,0.35) !important;
    letter-spacing: 0.2px !important;
}

/* ── INPUTS ── */
.stNumberInput input, .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: rgba(129,140,248,0.45) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    color: rgba(255,255,255,0.45) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(79,70,229,0.07);
    border: 1px solid rgba(79,70,229,0.2);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.79rem; color: rgba(255,255,255,0.48);
    line-height: 1.75; margin-top: 1.4rem;
}
.info-box .info-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem; font-weight: 600;
    color: #a5b4fc; margin-bottom: 6px;
}
.info-box code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #67e8f9;
    background: rgba(103,232,249,0.07);
    padding: 1px 6px; border-radius: 4px;
    word-break: break-all;
}

/* ── QUICK TEST GRID ── */
.qt-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2.5px;
    color: rgba(129,140,248,0.5);
    margin-top: 1.8rem; margin-bottom: 0.8rem;
}

/* ── LOG ENTRIES ── */
.log-wrap {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 0.5rem;
    max-height: 420px; overflow-y: auto;
}
.log-row {
    display: flex; align-items: center; gap: 12px;
    padding: 9px 12px; border-radius: 8px;
    transition: background 0.12s;
    font-family: 'JetBrains Mono', monospace; font-size: 0.71rem;
}
.log-row:hover { background: rgba(255,255,255,0.03); }
.log-ts   { color: rgba(165,180,252,0.5); flex-shrink: 0; width: 52px; }
.log-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.log-name { font-weight: 600; }
.log-coord{ color: rgba(255,255,255,0.22); font-size: 0.67rem; }
.log-d    { margin-left: auto; color: rgba(255,255,255,0.28); flex-shrink: 0; }
.log-src  { color: rgba(255,255,255,0.22); flex-shrink: 0; }

/* ── ABOUT TEXT ── */
.about-body {
    font-size: 0.85rem; color: rgba(255,255,255,0.5);
    line-height: 2;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 1.6rem;
}
.about-body h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem; font-weight: 600;
    color: rgba(255,255,255,0.75); margin: 1.1rem 0 0.3rem;
}
.about-body h4:first-child { margin-top: 0; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; overflow: hidden;
}

/* ── PLOTLY CHARTS ── */
[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }

/* ── EMPTY STATE ── */
.empty-state {
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 16px; padding: 3.5rem 2rem;
    text-align: center; color: rgba(255,255,255,0.2);
}
.empty-state .es-icon {
    font-size: 1.8rem; margin-bottom: 10px;
    color: rgba(255,255,255,0.1);
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
}
.empty-state .es-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; color: rgba(255,255,255,0.22);
}
.empty-state .es-sub {
    font-size: 0.78rem; margin-top: 6px; color: rgba(255,255,255,0.15);
}

/* ── AMPHI REFERENCE LIST ── */
.ref-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.8rem;
}
.ref-item:last-child { border-bottom: none; }
.ref-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ref-name { color: rgba(255,255,255,0.65); flex: 1; }
.ref-floor {
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    padding: 2px 10px; border-radius: 5px;
}
.rf1 { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }
.rf2 { background: rgba(168,85,247,0.15); color: #d8b4fe; border: 1px solid rgba(168,85,247,0.25); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CENTROIDS = {
    "Amphi 1": (36.688350, 2.866760),
    "Amphi 2": (36.688318, 2.866539),
    "Amphi 3": (36.688308, 2.866222),
    "Amphi 4": (36.688303, 2.866123),
    "Amphi 5": (36.688358, 2.866695),
    "Amphi 6": (36.688370, 2.866409),
    "Amphi 7": (36.688380, 2.866311),
    "Amphi 8": (36.688323, 2.866134),
}
FLOOR  = {**{f"Amphi {i}": 1 for i in range(1, 5)},
          **{f"Amphi {i}": 2 for i in range(5, 9)}}
COLORS = {
    "Amphi 1": "#6366f1", "Amphi 2": "#22c55e",
    "Amphi 3": "#a855f7", "Amphi 4": "#ef4444",
    "Amphi 5": "#06b6d4", "Amphi 6": "#f59e0b",
    "Amphi 7": "#ec4899", "Amphi 8": "#14b8a6",
    "Outside": "#475569",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def classify(lat: float, lon: float, accuracy: float = 25.0) -> dict:
    dists   = {n: haversine(lat, lon, c[0], c[1]) for n, c in CENTROIDS.items()}
    nearest = min(dists, key=dists.get)
    nd      = dists[nearest]
    inv     = {k: 1.0 / max(v, 0.5) for k, v in dists.items()}
    s       = sum(inv.values())
    probs   = {k: v / s for k, v in inv.items()}
    on_campus = (36.6876 <= lat <= 36.6892 and 2.8650 <= lon <= 2.8685)
    label   = "Outside" if (nd > max(accuracy * 2.0, 50) or not on_campus) else nearest
    return dict(label=label, nearest=nearest, dist=nd,
                dists=dists, probs=probs, floor=FLOOR.get(label))


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "localhost"


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in dict(history=[], result=None,
                 sid=hashlib.md5(str(time.time()).encode()).hexdigest()[:8]).items():
    if k not in st.session_state:
        st.session_state[k] = v

# Auto-classify from URL params (phone submission)
params = st.query_params
if "lat" in params and "lon" in params and "submitted" not in st.session_state:
    try:
        r = classify(float(params["lat"]), float(params["lon"]),
                     float(params.get("acc", 25)))
        r.update(dict(lat=float(params["lat"]), lon=float(params["lon"]),
                      accuracy=float(params.get("acc", 25)),
                      ts=datetime.now().strftime("%H:%M:%S"), source="phone"))
        st.session_state.result  = r
        st.session_state.history.append(r)
        st.session_state.submitted = True
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
n_scans = len(st.session_state.history)
ip      = get_ip()

st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">Amphi<span>Locator</span></div>
        <div class="app-subtitle">ENSIA &nbsp;·&nbsp; GPS Classifier &nbsp;·&nbsp; v2.0</div>
    </div>
    <div class="header-badges">
        <span class="badge live">
            <span class="live-indicator"></span>Online
        </span>
        <span class="badge">8 amphitheatres</span>
        <span class="badge">{n_scans} scans &nbsp;·&nbsp; session {st.session_state.sid}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_detect, tab_log, tab_about = st.tabs(["Detect", "Scan Log", "About"])


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 1 — DETECT
# ───────────────────────────────────────────────────────────────────────────────
with tab_detect:

    history = st.session_state.history
    total   = len(history)
    inside  = sum(1 for r in history if r["label"] != "Outside")
    avg_d   = sum(r["dist"] for r in history) / total if total else 0

    # Stats row
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card s-purple">
            <div class="stat-val">{total}</div>
            <div class="stat-key">Total Scans</div>
        </div>
        <div class="stat-card s-green">
            <div class="stat-val">{inside}</div>
            <div class="stat-key">Inside Amphis</div>
        </div>
        <div class="stat-card s-cyan">
            <div class="stat-val">{total - inside}</div>
            <div class="stat-key">Outside</div>
        </div>
        <div class="stat-card s-amber">
            <div class="stat-val">{avg_d:.0f}m</div>
            <div class="stat-key">Avg Distance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    # ── LEFT — input ──────────────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="slabel">GPS Coordinates</div>', unsafe_allow_html=True)

        # ── smart paste field ─────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.38);
                    margin-bottom:8px;line-height:1.6;">
            Paste coordinates in any format — Google Maps, phone browser, or raw numbers.
        </div>
        """, unsafe_allow_html=True)

        raw_input = st.text_input(
            "coords_paste",
            placeholder="36.688350, 2.866760   or   36.688350 2.866760",
            label_visibility="collapsed",
            key="raw_coords",
        )

        import re as _re
        _parse_err = ""
        _lat_default, _lon_default = 36.6883500, 2.8664000
        if raw_input.strip():
            nums = _re.findall(r"[-+]?\d+\.?\d*", raw_input)
            if len(nums) >= 2:
                try:
                    _lat_default = float(nums[0])
                    _lon_default = float(nums[1])
                except ValueError:
                    _parse_err = "Could not parse — try: 36.688350, 2.866760"
            else:
                _parse_err = "Need two numbers (lat, lon)"

        if _parse_err:
            st.markdown(f'<div style="font-size:0.74rem;color:#f87171;margin-top:2px;'
                        f'font-family:JetBrains Mono,monospace;">{_parse_err}</div>',
                        unsafe_allow_html=True)
        elif raw_input.strip():
            st.markdown(f'<div style="font-size:0.72rem;color:rgba(74,222,128,0.75);margin-top:2px;'
                        f'font-family:JetBrains Mono,monospace;">Parsed: {_lat_default:.7f}, {_lon_default:.7f}</div>',
                        unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude",  value=_lat_default, format="%.7f",
                              min_value=36.60, max_value=36.76, key="lat_fine")
        lon = c2.number_input("Longitude", value=_lon_default, format="%.7f",
                              min_value=2.80,  max_value=2.90,  key="lon_fine")
        acc = st.number_input("Accuracy (metres)", value=22.0,
                              min_value=1.0, max_value=500.0)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.button("Classify Location", use_container_width=True, type="primary"):
            r = classify(lat, lon, acc)
            r.update(dict(lat=lat, lon=lon, accuracy=acc,
                          ts=datetime.now().strftime("%H:%M:%S"), source="manual"))
            st.session_state.result = r
            st.session_state.history.append(r)
            st.rerun()

        st.markdown(f"""
        <div class="info-box">
            <div class="info-title">Phone submission</div>
            Students can submit directly from their phone browser:<br>
            <code>http://{ip}:8501/?lat=LAT&amp;lon=LON&amp;acc=ACC</code><br>
            The app will auto-classify as soon as the URL is opened.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="qt-label">Quick test — jump to centroid</div>',
                    unsafe_allow_html=True)
        row1, row2 = st.columns(4), st.columns(4)
        for i, (name, (clat, clon)) in enumerate(CENTROIDS.items()):
            col = row1[i] if i < 4 else row2[i - 4]
            if col.button(name.replace("Amphi ", "A"), use_container_width=True,
                          key=f"qt_{name}"):
                r = classify(clat, clon, 15.0)
                r.update(dict(lat=clat, lon=clon, accuracy=15.0,
                              ts=datetime.now().strftime("%H:%M:%S"), source="test"))
                st.session_state.result = r
                st.session_state.history.append(r)
                st.rerun()

    # ── RIGHT — result ────────────────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="slabel">Classification Result</div>', unsafe_allow_html=True)
        r = st.session_state.result

        if r is None:
            st.markdown("""
            <div class="empty-state">
                <div class="es-icon">--</div>
                <div class="es-title">No result yet</div>
                <div class="es-sub">Enter coordinates and press Classify</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_inside = r["label"] != "Outside"
            cls       = "result-inside" if is_inside else "result-outside"
            lbl_text  = "CLASSIFIED INSIDE" if is_inside else "CLASSIFIED OUTSIDE"
            floor_html = ""
            if r["floor"]:
                fc = "f1" if r["floor"] == 1 else "f2"
                floor_html = f'<div class="floor-tag {fc}">Floor {r["floor"]}</div>'
            src_map = {"manual": "manual", "test": "test", "phone": "phone"}

            st.markdown(f"""
            <div class="result-card {cls}">
                <div class="result-label">{lbl_text}</div>
                <div class="result-name">{r['label']}</div>
                {floor_html}
                <div class="result-meta">
                    <b>{r['dist']:.1f} m</b> to centroid
                    &nbsp;&middot;&nbsp;
                    GPS &plusmn;<b>{r.get('accuracy',25):.0f} m</b>
                    &nbsp;&middot;&nbsp;
                    {src_map.get(r.get('source',''), 'unknown')}
                </div>
                <div class="result-coords">
                    {r.get('lat',0):.7f}, {r.get('lon',0):.7f}
                    &nbsp;&middot;&nbsp; {r.get('ts','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown('<div class="slabel" style="margin-top:1.8rem;">Class Probabilities</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="conf-block">', unsafe_allow_html=True)
            for name, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
                c = COLORS.get(name, "#475569")
                w = f"{prob * 100:.1f}%"
                st.markdown(f"""
                <div class="conf-item">
                    <span class="conf-label">{name}</span>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill"
                             style="width:{w};background:{c};opacity:0.85;"></div>
                    </div>
                    <span class="conf-pct">{w}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 2 — SCAN LOG
# ───────────────────────────────────────────────────────────────────────────────
with tab_log:
    history = st.session_state.history

    if not history:
        st.markdown("""
        <div class="empty-state" style="margin-top:1rem;">
            <div class="es-icon">--</div>
            <div class="es-title">No scans yet</div>
            <div class="es-sub">Classify a location on the Detect tab to see history here</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_log, col_charts = st.columns([1, 1.1], gap="large")

        with col_log:
            st.markdown(f'<div class="slabel">History &middot; {len(history)} entries</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="log-wrap">', unsafe_allow_html=True)
            for rec in reversed(history[-50:]):
                c    = COLORS.get(rec["label"], "#475569")
                src  = {"manual": "kbd", "test": "test", "phone": "phone"}.get(
                           rec.get("source",""), "—")
                st.markdown(f"""
                <div class="log-row">
                    <span class="log-ts">{rec.get('ts','—')}</span>
                    <span class="log-dot" style="background:{c};"></span>
                    <span class="log-name" style="color:{c};">{rec['label']}</span>
                    <span class="log-coord">
                        {rec.get('lat',0):.5f}, {rec.get('lon',0):.5f}
                    </span>
                    <span class="log-src">{src}</span>
                    <span class="log-d">{rec['dist']:.0f} m</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.session_state.result  = None
                st.rerun()

        with col_charts:
            df_h  = pd.DataFrame(history)
            cnts  = df_h["label"].value_counts().reset_index()
            cnts.columns = ["label", "count"]

            st.markdown('<div class="slabel">Distribution</div>', unsafe_allow_html=True)
            fig_b = go.Figure(go.Bar(
                x=cnts["label"], y=cnts["count"],
                marker_color=[COLORS.get(l, "#475569") for l in cnts["label"]],
                marker_opacity=0.82,
                text=cnts["count"], textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.45)", size=11),
                marker_line_width=0,
            ))
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.015)",
                font=dict(color="rgba(255,255,255,0.4)", family="Inter", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           showline=False, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           showline=False, title=""),
                height=240, margin=dict(t=10, b=0, l=0, r=10),
                bargap=0.35,
            )
            st.plotly_chart(fig_b, use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown('<div class="slabel" style="margin-top:1.2rem;">Distance over time</div>',
                        unsafe_allow_html=True)
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=list(range(1, len(df_h) + 1)), y=df_h["dist"],
                mode="lines",
                line=dict(color="rgba(129,140,248,0.3)", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(129,140,248,0.04)",
                showlegend=False,
            ))
            fig_t.add_trace(go.Scatter(
                x=list(range(1, len(df_h) + 1)), y=df_h["dist"],
                mode="markers",
                marker=dict(color=[COLORS.get(l, "#475569") for l in df_h["label"]],
                            size=8, line=dict(width=1.5, color="rgba(255,255,255,0.15)")),
                text=df_h["label"],
                hovertemplate="<b>%{text}</b><br>%{y:.1f} m<extra></extra>",
                showlegend=False,
            ))
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.015)",
                font=dict(color="rgba(255,255,255,0.4)", family="Inter", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           showline=False,
                           title=dict(text="Scan #",
                                      font=dict(size=10))),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           showline=False,
                           title=dict(text="Distance (m)",
                                      font=dict(size=10))),
                height=220, margin=dict(t=10, b=0, l=0, r=10),
            )
            st.plotly_chart(fig_t, use_container_width=True,
                            config={"displayModeBar": False})


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT
# ───────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns([1.5, 1], gap="large")

    with col_a:
        st.markdown('<div class="slabel">Project</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-body">
            <h4>AmphiLocator v2.0</h4>
            ENSIA Machine Learning Project &middot; Spring 2025-2026
            <h4>Objective</h4>
            Classify which of ENSIA's 8 amphitheatres a student is sitting in,
            using only the GPS coordinates read from their phone.
            This is a 9-class problem: Amphi 1 through Amphi 8, plus Outside.

            <h4>Classification method</h4>
            Nearest-centroid classifier on latitude and longitude,
            with soft-max confidence scores computed over inverse distances.
            Outside detection uses a threshold proportional to the device-reported
            GPS accuracy radius.

            <h4>Data</h4>
            gps_data_v2.csv &mdash; 8,224 clean readings &mdash; 9 classes
            collected across all 8 amphitheatres and outdoor campus areas.

            <h4>Limitation</h4>
            GPS cannot distinguish Floor 1 from Floor 2.
            Rooms on the same building column
            (Amphi 1 & 5, 2 & 6, 3 & 7, 4 & 8)
            share the same lat/lon footprint.
            A future version should incorporate Wi-Fi fingerprinting or
            barometric pressure to resolve floor ambiguity.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Centroid Table</div>', unsafe_allow_html=True)
        rows = [{"Amphi": n, "Floor": FLOOR[n],
                 "Lat": f"{lat:.6f}", "Lon": f"{lon:.6f}"}
                for n, (lat, lon) in CENTROIDS.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="slabel">Amphitheatre Reference</div>', unsafe_allow_html=True)
        for name, (clat, clon) in CENTROIDS.items():
            fl = FLOOR[name]
            fc = "rf1" if fl == 1 else "rf2"
            c  = COLORS[name]
            is_cur = (st.session_state.result and
                      st.session_state.result["label"] == name)
            hl = "background:rgba(255,255,255,0.04);" if is_cur else ""
            st.markdown(f"""
            <div class="ref-item" style="{hl}">
                <span class="ref-dot" style="background:{c};"></span>
                <span class="ref-name">{name}</span>
                <span class="ref-floor {fc}">Floor {fl}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Data Distribution (v2)</div>', unsafe_allow_html=True)
        sample_data = {
            "Amphi 2": 2564, "Amphi 5": 1226, "Amphi 8": 999,
            "Outside":  876, "Amphi 6":  774, "Amphi 4": 741,
            "Amphi 1":  604, "Amphi 3":  318, "Amphi 7": 122,
        }
        fig_pie = go.Figure(go.Pie(
            labels=list(sample_data.keys()),
            values=list(sample_data.values()),
            marker_colors=[COLORS.get(l, "#475569") for l in sample_data],
            marker=dict(line=dict(color="#09090f", width=2)),
            hole=0.65,
            textinfo="label+percent",
            textfont=dict(size=9.5, color="rgba(255,255,255,0.6)"),
            showlegend=False,
            pull=[0.03] * len(sample_data),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(t=0, b=0, l=0, r=0),
            annotations=[dict(
                text="8,224", x=0.5, y=0.55,
                font=dict(size=18, color="rgba(255,255,255,0.7)",
                          family="Space Grotesk"),
                showarrow=False,
            ), dict(
                text="readings", x=0.5, y=0.4,
                font=dict(size=10, color="rgba(255,255,255,0.25)",
                          family="Inter"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False})