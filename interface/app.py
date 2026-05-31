import io, math, csv, hashlib, time, socket, warnings, re as _re
from datetime import datetime
from pathlib import Path
from collections import Counter

import qrcode
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

#  CONSTANTS
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
FLOOR = {**{f"Amphi {i}": 1 for i in range(1, 5)},
         **{f"Amphi {i}": 2 for i in range(5, 9)}}
COLORS = {
    "Amphi 1": "#6366f1", "Amphi 2": "#22c55e",
    "Amphi 3": "#a855f7", "Amphi 4": "#ef4444",
    "Amphi 5": "#06b6d4", "Amphi 6": "#f59e0b",
    "Amphi 7": "#ec4899", "Amphi 8": "#14b8a6",
    "Outside": "#475569",
}
FLOOR_PAIRS = {
    "Amphi 1": "Amphi 5", "Amphi 5": "Amphi 1",
    "Amphi 2": "Amphi 6", "Amphi 6": "Amphi 2",
    "Amphi 3": "Amphi 7", "Amphi 7": "Amphi 3",
    "Amphi 4": "Amphi 8", "Amphi 8": "Amphi 4",
}

TOKEN_WINDOW   = 20
MODELS_DIR     = Path(__file__).parent.parent / "models"
DATA_DIR       = Path(__file__).parent.parent / "data"
ATTENDANCE_CSV = DATA_DIR / "attendance_log.csv"

CSV_COLUMNS = [
    "session_date", "timestamp", "student_name", "student_id",
    "label", "floor", "latitude", "longitude", "accuracy_m",
    "dist_m", "agreement_pct", "voters_used", "low_agreement",
    "floor_pair_split", "token_window",
]

#  TOKEN SYSTEM

_SECRET = "amphilocator-ensia-2026"

def _window_token(window: int) -> str:
    raw = f"{_SECRET}-{window}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

def current_token() -> str:
    return _window_token(int(time.time()) // TOKEN_WINDOW)

def is_valid_token(token: str) -> bool:
    w = int(time.time()) // TOKEN_WINDOW
    return token in (_window_token(w), _window_token(w - 1))

def seconds_until_next_token() -> int:
    return TOKEN_WINDOW - (int(time.time()) % TOKEN_WINDOW)

#  MODEL LOADING

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    model_files = {
        "gbm": "gbm_best.pkl", "knn": "knn_model.pkl",
        "lr":  "lr_model.pkl", "dt":  "dt_tuned.pkl",
        "rf":  "rf_tuned.pkl", "outside": "outside_detector.pkl",
        "label_enc": "label_encoder.pkl",
    }
    return {k: (joblib.load(MODELS_DIR / fn) if (MODELS_DIR / fn).exists() else None)
            for k, fn in model_files.items()}

MODELS     = load_models()
VOTER_KEYS = ["gbm", "knn", "lr", "dt", "rf"]

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2-p1)/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "localhost"

def build_features(lat, lon, accuracy):
    dists  = {n: haversine(lat, lon, c[0], c[1]) for n, c in CENTROIDS.items()}
    sd     = sorted(dists.values())
    nd, d2 = sd[0], sd[1]
    nn     = min(dists, key=dists.get)
    ni     = list(CENTROIDS.keys()).index(nn)
    AM, AS, DM, DS = 22.0, 15.0, 50.0, 40.0
    LAM, LAS       = np.log1p(22.0), 0.8
    hour = datetime.now().hour
    scaled = {
        "accuracy_mean_scaled": (accuracy-AM)/AS,
        **{f"dist_Amphi_{i}_scaled": (dists[f"Amphi {i}"]-DM)/DS for i in range(1,9)},
        "dist_nearest_scaled": (nd-DM)/DS, "dist_2nd_scaled": (d2-DM)/DS,
        "dist_gap_scaled": (d2-nd)/DS,
        "log_accuracy_scaled": (np.log1p(accuracy)-LAM)/LAS,
    }
    binary = {
        "high_accuracy_flag": int(accuracy<20), "has_seat": 0,
        "accuracy_bin": int(np.digitize(accuracy,[10,25,50,100])),
        "seat_block_enc": 0,
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "seat_row_filled": 0, "seat_column_filled": 0, "seat_zone_id": 0,
    }
    ohe = {f"nearest_amphi_{i}": int(ni==i) for i in range(8)}
    return pd.DataFrame([{**scaled, **binary, **ohe}])

def decode_label(pred_encoded, label_enc):
    if label_enc:
        try: return str(label_enc.inverse_transform([pred_encoded])[0])
        except Exception: pass
    try:
        idx = int(pred_encoded)
        return f"Amphi {idx+1}" if idx < 8 else "Outside"
    except Exception:
        return str(pred_encoded)

def classify(lat, lon, accuracy=25.0):
    dists = {n: haversine(lat, lon, c[0], c[1]) for n, c in CENTROIDS.items()}
    nn    = min(dists, key=dists.get)
    nd    = dists[nn]
    inv   = {k: 1.0/max(v, 0.5) for k, v in dists.items()}
    s     = sum(inv.values())
    probs = {k: v/s for k, v in inv.items()}

    le      = MODELS.get("label_enc")
    vb      = {}; vu = 0; agr = 1.0
    active  = {k: MODELS[k] for k in VOTER_KEYS if MODELS.get(k)}

    if active:
        try:
            feats   = build_features(lat, lon, accuracy)
            outside = MODELS.get("outside")
            is_out  = bool(outside.predict(feats)[0]) if outside else False
            if is_out:
                label = "Outside"; vb = {k: "Outside" for k in active}; vu = len(active)
            else:
                votes = []; ap = {}
                for key, model in active.items():
                    try:
                        pn = decode_label(model.predict(feats)[0], le)
                        vb[key] = pn; votes.append(pn); vu += 1
                        if hasattr(model, "predict_proba"):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rp = model.predict_proba(feats)[0]
                            cls = ([str(c) for c in le.classes_] if le
                                   else [f"Amphi {i+1}" for i in range(8)]+["Outside"])
                            for c, p in zip(cls, rp):
                                ap[c] = ap.get(c, 0.0) + float(p)
                    except Exception as e:
                        vb[key] = f"err:{e}"
                if votes:
                    vc    = Counter(votes)
                    label = vc.most_common(1)[0][0]
                    agr   = vc.most_common(1)[0][1] / len(votes)
                    npm   = sum(1 for m in active.values() if hasattr(m,"predict_proba"))
                    if ap and npm > 0:
                        probs = {k: v/npm for k, v in ap.items()}
                else:
                    label = nn
        except Exception as e:
            st.warning(f"Model error - centroid fallback: {e}")
            on = (36.6876<=lat<=36.6892 and 2.8650<=lon<=2.8685)
            label = "Outside" if (nd>max(accuracy*2,50) or not on) else nn
    else:
        on    = (36.6876<=lat<=36.6892 and 2.8650<=lon<=2.8685)
        label = "Outside" if (nd>max(accuracy*2,50) or not on) else nn

    fps = False
    if vb:
        pair = FLOOR_PAIRS.get(label)
        if pair:
            pv = sum(1 for v in vb.values() if v in (label, pair))
            if pv == len(vb) and agr < 1.0: fps = True
    la = (agr < 0.6) and (label != "Outside") and vu > 1

    return dict(label=label, nearest=nn, dist=nd, dists=dists, probs=probs,
                floor=FLOOR.get(label), vote_breakdown=vb, voters_used=vu,
                agreement_pct=agr, low_agreement=la, floor_pair_split=fps)

# ─────────────────────────────────────────────────────────────────────────────
#  CSV PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def ensure_csv():
    ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def append_to_csv(row: dict):
    ensure_csv()
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)

def load_full_csv() -> pd.DataFrame:
    ensure_csv()
    try:
        df = pd.read_csv(ATTENDANCE_CSV)
        return df if not df.empty else pd.DataFrame(columns=CSV_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)

def read_csv_bytes() -> bytes:
    ensure_csv()
    with open(ATTENDANCE_CSV, "rb") as f: return f.read()

# ─────────────────────────────────────────────────────────────────────────────
#  QR CODE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=TOKEN_WINDOW, show_spinner=False)
def make_qr_png(url: str) -> bytes:
    qr = qrcode.QRCode(version=None,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=10, border=3)
    qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="#f0f6fc", back_color="#09090f")
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def session_to_csv(history, dedup=False) -> bytes:
    if not history: return b""
    rows = history.copy()
    if dedup:
        seen = {}
        for r in history: seen[r.get("student_id", r.get("source",""))] = r
        rows = list(seen.values())
    df = pd.DataFrame([{
        "timestamp": r.get("ts",""), "student_name": r.get("student_name",""),
        "student_id": r.get("student_id",""), "label": r.get("label",""),
        "floor": r.get("floor",""), "latitude": r.get("lat",""),
        "longitude": r.get("lon",""), "accuracy_m": r.get("accuracy",""),
        "dist_m": round(r.get("dist",0),2), "source": r.get("source",""),
        "agreement_pct": round(r.get("agreement_pct",1.0),3),
        "voters_used": r.get("voters_used",0),
    } for r in rows])
    return df.to_csv(index=False).encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AmphiLocator - ENSIA",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#09090f!important;color:#c9d1d9}
#MainMenu,header,footer{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem!important;max-width:1320px!important}
section[data-testid="stSidebar"]{display:none!important}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 70% 50% at 15% 10%,rgba(88,28,220,.12) 0%,transparent 55%),
             radial-gradient(ellipse 55% 60% at 88% 85%,rgba(6,148,162,.09) 0%,transparent 55%)}

.app-header{padding:2rem 0 2.4rem;border-bottom:1px solid rgba(255,255,255,.05);margin-bottom:2.5rem;position:relative;z-index:10;display:flex;align-items:flex-end;justify-content:space-between}
.app-title{font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;letter-spacing:-.8px;color:#f0f6fc;line-height:1}
.app-title span{background:linear-gradient(120deg,#818cf8,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.app-subtitle{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:rgba(255,255,255,.28);letter-spacing:2px;text-transform:uppercase;margin-top:6px}
.header-badges{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.5px;padding:5px 14px;border-radius:6px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);color:rgba(255,255,255,.4)}
.badge.live{border-color:rgba(34,197,94,.35);background:rgba(34,197,94,.07);color:#4ade80}
.live-indicator{width:6px;height:6px;border-radius:50%;background:#4ade80;display:inline-block;margin-right:6px;vertical-align:middle;box-shadow:0 0 8px rgba(74,222,128,.6);animation:blink 2.5s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}

.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.06)!important;border-radius:10px!important;padding:4px!important;gap:3px!important;margin-bottom:2rem!important}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;color:rgba(255,255,255,.38)!important;font-family:'Inter',sans-serif!important;font-size:.83rem!important;font-weight:500!important;padding:7px 20px!important;transition:all .18s!important}
.stTabs [data-baseweb="tab"]:hover{color:rgba(255,255,255,.7)!important;background:rgba(255,255,255,.04)!important}
.stTabs [aria-selected="true"]{background:rgba(129,140,248,.18)!important;color:#c7d2fe!important;font-weight:600!important;border:1px solid rgba(129,140,248,.25)!important}
.stTabs [data-baseweb="tab-highlight"]{display:none!important}

.slabel{font-family:'JetBrains Mono',monospace;font-size:.6rem;text-transform:uppercase;letter-spacing:3px;color:rgba(129,140,248,.6);margin-bottom:1rem;padding-bottom:8px;border-bottom:1px solid rgba(129,140,248,.12)}
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:2.4rem}
.stat-card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.2rem 1.4rem;position:relative;overflow:hidden}
.stat-card::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;border-radius:14px 14px 0 0}
.stat-card.s-purple::after{background:linear-gradient(90deg,#818cf8,transparent)}
.stat-card.s-green::after{background:linear-gradient(90deg,#34d399,transparent)}
.stat-card.s-cyan::after{background:linear-gradient(90deg,#67e8f9,transparent)}
.stat-card.s-amber::after{background:linear-gradient(90deg,#fbbf24,transparent)}
.stat-val{font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;line-height:1;margin-bottom:5px}
.stat-card.s-purple .stat-val{color:#a5b4fc}
.stat-card.s-green  .stat-val{color:#6ee7b7}
.stat-card.s-cyan   .stat-val{color:#a5f3fc}
.stat-card.s-amber  .stat-val{color:#fde68a}
.stat-key{font-size:.7rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:1.2px;font-weight:500}

.result-card{border-radius:18px;padding:2.4rem 2rem;text-align:center;position:relative;overflow:hidden}
.result-inside{background:linear-gradient(160deg,rgba(4,120,87,.22),rgba(5,150,105,.08));border:1px solid rgba(16,185,129,.35);box-shadow:0 0 50px rgba(16,185,129,.1),inset 0 1px 0 rgba(255,255,255,.04)}
.result-outside{background:linear-gradient(160deg,rgba(146,64,14,.22),rgba(180,83,9,.08));border:1px solid rgba(245,158,11,.35);box-shadow:0 0 50px rgba(245,158,11,.1),inset 0 1px 0 rgba(255,255,255,.04)}
.result-label{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px}
.result-inside  .result-label{color:rgba(52,211,153,.7)}
.result-outside .result-label{color:rgba(251,191,36,.7)}
.result-name{font-family:'Space Grotesk',sans-serif;font-size:3.2rem;font-weight:700;letter-spacing:-1.5px;color:#f0f6fc;line-height:1;margin-bottom:10px}
.floor-tag{display:inline-block;padding:3px 16px;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:600;margin-bottom:10px;letter-spacing:.5px}
.floor-tag.f1{background:rgba(99,102,241,.18);color:#a5b4fc;border:1px solid rgba(99,102,241,.3)}
.floor-tag.f2{background:rgba(168,85,247,.18);color:#d8b4fe;border:1px solid rgba(168,85,247,.3)}
.model-badge{display:inline-block;margin-bottom:12px;font-family:'JetBrains Mono',monospace;font-size:.62rem;padding:3px 12px;border-radius:5px;letter-spacing:.5px;background:rgba(129,140,248,.1);color:rgba(129,140,248,.7);border:1px solid rgba(129,140,248,.2)}
.result-meta{font-size:.78rem;color:rgba(255,255,255,.38);line-height:2;margin-top:6px}
.result-meta b{color:rgba(255,255,255,.65);font-weight:500}
.result-coords{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.25);margin-top:8px}
.vote-grid{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:14px}
.vote-chip{font-family:'JetBrains Mono',monospace;font-size:.62rem;padding:3px 10px;border-radius:5px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);color:rgba(255,255,255,.45)}
.vote-chip.winner{background:rgba(129,140,248,.15);color:#c7d2fe;border-color:rgba(129,140,248,.3)}
.conf-warning{margin-top:14px;border-radius:9px;padding:10px 14px;font-family:'JetBrains Mono',monospace;font-size:.68rem;line-height:1.6}
.conf-warning.warn-floor{background:rgba(168,85,247,.12);border:1px solid rgba(168,85,247,.35);color:#d8b4fe}
.conf-warning.warn-low{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.35);color:#fde68a}
.conf-block{margin-top:1.4rem}
.conf-item{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.conf-label{font-size:.75rem;color:rgba(255,255,255,.5);width:68px;flex-shrink:0;font-family:'JetBrains Mono',monospace}
.conf-bar-bg{flex:1;height:5px;border-radius:999px;background:rgba(255,255,255,.06)}
.conf-bar-fill{height:100%;border-radius:999px}
.conf-pct{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.3);width:36px;text-align:right;flex-shrink:0}

.stButton>button{background:rgba(255,255,255,.04)!important;color:rgba(255,255,255,.6)!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;font-weight:500!important;padding:.45rem .8rem!important;transition:all .15s!important;box-shadow:none!important}
.stButton>button:hover{background:rgba(129,140,248,.12)!important;border-color:rgba(129,140,248,.3)!important;color:#c7d2fe!important}
div[data-testid="stButton"]:has(button[kind="primary"])>button{background:linear-gradient(135deg,#4f46e5,#6d28d9)!important;color:#fff!important;border:1px solid rgba(129,140,248,.3)!important;border-radius:10px!important;font-family:'Inter',sans-serif!important;font-size:.88rem!important;font-weight:600!important;padding:.65rem 1.4rem!important;box-shadow:0 4px 24px rgba(79,70,229,.35)!important}
[data-testid="stDownloadButton"] button{background:rgba(34,197,94,.08)!important;color:#6ee7b7!important;border:1px solid rgba(34,197,94,.25)!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;padding:.45rem .8rem!important}
[data-testid="stDownloadButton"] button:hover{background:rgba(34,197,94,.15)!important;border-color:rgba(34,197,94,.45)!important}

.stNumberInput input,.stTextInput input{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:9px!important;color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;font-size:.85rem!important}
.stNumberInput input:focus,.stTextInput input:focus{border-color:rgba(129,140,248,.45)!important;box-shadow:0 0 0 3px rgba(129,140,248,.1)!important}
label[data-testid="stWidgetLabel"] p{font-size:.78rem!important;color:rgba(255,255,255,.45)!important;font-family:'Inter',sans-serif!important;font-weight:500!important}
[data-testid="stCheckbox"] label p{font-size:.78rem!important;color:rgba(255,255,255,.45)!important}

.info-box{background:rgba(79,70,229,.07);border:1px solid rgba(79,70,229,.2);border-radius:10px;padding:1rem 1.2rem;font-size:.79rem;color:rgba(255,255,255,.48);line-height:1.75;margin-top:1.4rem}
.info-box .info-title{font-family:'Space Grotesk',sans-serif;font-size:.8rem;font-weight:600;color:#a5b4fc;margin-bottom:6px}
.info-box code{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#67e8f9;background:rgba(103,232,249,.07);padding:1px 6px;border-radius:4px;word-break:break-all}

.qr-panel{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.4rem;text-align:center}
.qr-url{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:rgba(103,232,249,.7);word-break:break-all;margin-top:10px;padding:8px 10px;background:rgba(103,232,249,.05);border:1px solid rgba(103,232,249,.12);border-radius:7px}
.token-bar-wrap{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:10px 14px;margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.4)}
.token-bar-inner{height:3px;border-radius:999px;background:linear-gradient(90deg,#818cf8,#67e8f9);margin-top:6px}

.qt-label{font-family:'JetBrains Mono',monospace;font-size:.6rem;text-transform:uppercase;letter-spacing:2.5px;color:rgba(129,140,248,.5);margin-top:1.8rem;margin-bottom:.8rem}

.log-wrap{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:12px;padding:.5rem;max-height:420px;overflow-y:auto}
.log-row{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;transition:background .12s;font-family:'JetBrains Mono',monospace;font-size:.71rem}
.log-row:hover{background:rgba(255,255,255,.03)}
.log-ts{color:rgba(165,180,252,.5);flex-shrink:0;width:52px}
.log-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.log-name{font-weight:600;min-width:64px}
.log-stu{color:rgba(255,255,255,.45);font-size:.68rem;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-d{margin-left:auto;color:rgba(255,255,255,.28);flex-shrink:0}

.about-body{font-size:.85rem;color:rgba(255,255,255,.5);line-height:2;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:14px;padding:1.6rem}
.about-body h4{font-family:'Space Grotesk',sans-serif;font-size:.82rem;font-weight:600;color:rgba(255,255,255,.75);margin:1.1rem 0 .3rem}
.about-body h4:first-child{margin-top:0}
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.06)!important;border-radius:12px!important;overflow:hidden}
[data-testid="stPlotlyChart"]{border-radius:14px;overflow:hidden}
.empty-state{border:1px dashed rgba(255,255,255,.08);border-radius:16px;padding:3.5rem 2rem;text-align:center}
.empty-state .es-icon{font-size:1.8rem;margin-bottom:10px;color:rgba(255,255,255,.1);font-family:'Space Grotesk',sans-serif;font-weight:700}
.empty-state .es-title{font-family:'Space Grotesk',sans-serif;font-size:1rem;color:rgba(255,255,255,.22)}
.empty-state .es-sub{font-size:.78rem;margin-top:6px;color:rgba(255,255,255,.15)}
.ref-item{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.8rem}
.ref-item:last-child{border-bottom:none}
.ref-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.ref-name{color:rgba(255,255,255,.65);flex:1}
.ref-floor{font-family:'JetBrains Mono',monospace;font-size:.66rem;padding:2px 10px;border-radius:5px}
.rf1{background:rgba(99,102,241,.15);color:#a5b4fc;border:1px solid rgba(99,102,241,.25)}
.rf2{background:rgba(168,85,247,.15);color:#d8b4fe;border:1px solid rgba(168,85,247,.25)}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(history=[], result=None,
                 sid=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
                 dedup_enabled=False)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

params = st.query_params
ip     = get_local_ip()

# ═════════════════════════════════════════════════════════════════════════════
#  MODE: RESULT
# ═════════════════════════════════════════════════════════════════════════════
if params.get("mode") == "result":
    token = params.get("tok", "")
    if not is_valid_token(token):
        st.markdown("""
        <div style="max-width:460px;margin:5rem auto;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;color:rgba(239,68,68,.7);">expired</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;color:#f0f6fc;margin-bottom:8px;">QR Code Expired</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:rgba(239,68,68,.8);letter-spacing:2px;text-transform:uppercase;margin-bottom:1.2rem;">Token no longer valid</div>
          <div style="font-size:.85rem;color:rgba(255,255,255,.4);line-height:1.9;">
            The QR rotates every 20 s.<br>Ask your teacher to show the current code and scan again.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    try:
        lat_v  = float(params.get("lat", "0"))
        lon_v  = float(params.get("lon", "0"))
        acc_v  = float(params.get("acc", "25"))
        sname  = params.get("name", "").replace("+", " ").strip()
        sid_v  = params.get("sid", "").strip()
    except Exception:
        st.error("Invalid submission data."); st.stop()

    if not sname or not sid_v:
        st.error("Name or student ID missing."); st.stop()
    if lat_v == 0.0 and lon_v == 0.0:
        st.error("GPS coordinates are zero - location access was likely denied."); st.stop()

    result = classify(lat_v, lon_v, acc_v)
    now    = datetime.now()
    ts_str = now.strftime("%H:%M:%S")

    rec = {
        "session_date":     now.strftime("%Y-%m-%d"),
        "timestamp":        ts_str,
        "student_name":     sname,
        "student_id":       sid_v,
        "label":            result["label"],
        "floor":            result.get("floor", ""),
        "latitude":         round(lat_v, 7),
        "longitude":        round(lon_v, 7),
        "accuracy_m":       round(acc_v, 1),
        "dist_m":           round(result["dist"], 2),
        "agreement_pct":    round(result.get("agreement_pct", 1.0), 3),
        "voters_used":      result.get("voters_used", 0),
        "low_agreement":    int(result.get("low_agreement", False)),
        "floor_pair_split": int(result.get("floor_pair_split", False)),
        "token_window":     token[:6],
    }
    append_to_csv(rec)

    fl = result.get("floor")
    floor_txt = f"Floor {fl}" if fl else ""
    fc  = "f1" if fl == 1 else "f2"
    is_in = result["label"] != "Outside"
    card_cls = "result-inside" if is_in else "result-outside"
    lbl_txt  = "ATTENDANCE RECORDED" if is_in else "OUTSIDE - NOT COUNTED"

    st.markdown(f"""
    <div style="max-width:480px;margin:4rem auto;">
      <div class="result-card {card_cls}" style="text-align:center;">
        <div class="result-label">{lbl_txt}</div>
        <div class="result-name">{result["label"]}</div>
        {"" if not fl else f'<div class="floor-tag {fc}">{floor_txt}</div>'}
        <div class="result-meta" style="margin-top:14px;">
          <b>{sname}</b> - {sid_v}<br>
          Recorded at <b>{ts_str}</b><br>
          Distance to centroid: <b>{result["dist"]:.1f} m</b>
        </div>
        <div class="result-coords">{lat_v:.6f}, {lon_v:.6f} - +/-{acc_v:.0f} m</div>
      </div>
      <div style="text-align:center;margin-top:1.4rem;font-family:'JetBrains Mono',monospace;font-size:.7rem;color:rgba(255,255,255,.25);">
        You can close this page.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  MODE: SUBMIT — student check-in form
# ═════════════════════════════════════════════════════════════════════════════
if params.get("mode") == "submit":
    token = params.get("token", "")
    if not is_valid_token(token):
        st.markdown("""
        <div style="max-width:460px;margin:5rem auto;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;color:rgba(239,68,68,.7);">expired</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;color:#f0f6fc;margin-bottom:8px;">QR Code Expired</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:rgba(239,68,68,.8);letter-spacing:2px;text-transform:uppercase;margin-bottom:1.2rem;">Token no longer valid</div>
          <div style="font-size:.85rem;color:rgba(255,255,255,.4);line-height:1.9;">
            The QR rotates every 20 s.<br>Ask your teacher to show the current code and scan again.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Render the submit form directly in the Streamlit page (NOT inside an iframe).
    # This is the key fix: st.components.v1.html() sandboxes JS and blocks
    # window.top navigation cross-origin. Using st.markdown directly means
    # the JS runs in the top-level page and location.href works perfectly.

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.submit-wrap{{display:flex;justify-content:center;padding:2rem 1rem;}}
.submit-card{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:2.4rem 2rem;width:100%;max-width:460px;}}
.submit-card h1{{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#f0f6fc;letter-spacing:-.5px;margin-bottom:4px;}}
.submit-sub{{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:rgba(255,255,255,.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:1.8rem;}}
.submit-tok-ok{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);color:#6ee7b7;font-family:'JetBrains Mono',monospace;font-size:.68rem;padding:8px 12px;border-radius:8px;margin-bottom:1rem;}}
.gps-box{{font-family:'JetBrains Mono',monospace;font-size:.72rem;padding:10px 14px;border-radius:9px;margin-bottom:1.2rem;transition:all .3s;line-height:1.6;}}
.gps-wait{{background:rgba(129,140,248,.08);border:1px solid rgba(129,140,248,.2);color:#a5b4fc;}}
.gps-ok{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);color:#6ee7b7;}}
.gps-err{{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);color:#fca5a5;}}
.sfix-box{{display:none;background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25);border-radius:10px;padding:14px 16px;margin-bottom:1.2rem;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:rgba(253,230,138,.9);line-height:1.8;}}
.sfix-box b{{color:#fde68a;}}
.sfix-box.show{{display:block;}}
.submit-card label{{display:block;font-size:.78rem;color:rgba(255,255,255,.45);margin-bottom:6px;margin-top:14px;}}
.submit-inp{{width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:9px;color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-size:.9rem;padding:10px 14px;outline:none;transition:border-color .15s;box-sizing:border-box;}}
.submit-inp:focus{{border-color:rgba(129,140,248,.5);box-shadow:0 0 0 3px rgba(129,140,248,.1);}}
.sbtn-primary{{margin-top:20px;width:100%;background:linear-gradient(135deg,#4f46e5,#6d28d9);color:#fff;border:1px solid rgba(129,140,248,.3);border-radius:10px;font-family:'Inter',sans-serif;font-size:.95rem;font-weight:600;padding:.75rem 1.4rem;cursor:pointer;box-shadow:0 4px 24px rgba(79,70,229,.35);transition:opacity .15s;display:block;}}
.sbtn-primary:hover{{opacity:.88;}}
.sbtn-primary:disabled{{opacity:.45;cursor:not-allowed;}}
.serr{{color:#fca5a5;font-size:.8rem;margin-top:12px;font-family:'JetBrains Mono',monospace;}}
.sdivider{{border:none;border-top:1px solid rgba(255,255,255,.06);margin:20px 0;}}
.smanual-section{{display:none;}}
.smanual-section.show{{display:block;}}
.sbtn-secondary{{margin-top:10px;width:100%;background:rgba(245,158,11,.1);color:#fde68a;border:1px solid rgba(245,158,11,.3);border-radius:10px;font-family:'Inter',sans-serif;font-size:.88rem;font-weight:600;padding:.65rem 1.4rem;cursor:pointer;transition:opacity .15s;display:block;}}
.sbtn-secondary:hover{{opacity:.8;}}
.strust-note{{font-family:'JetBrains Mono',monospace;font-size:.63rem;color:rgba(255,255,255,.25);margin-top:8px;text-align:center;}}
</style>

<div class="submit-wrap">
<div class="submit-card">
  <h1>Attendance Check-in</h1>
  <div class="submit-sub">ENSIA &nbsp;·&nbsp; AmphiLocator</div>
  <div class="submit-tok-ok">Valid QR code — session is active</div>

  <div class="gps-box gps-wait" id="gps-status">Acquiring GPS location...</div>

  <div class="sfix-box" id="sfix-box">
    <b>GPS blocked on this network</b><br>
    Browsers require HTTPS to access location.<br><br>
    <b>Option A (Android Chrome):</b><br>
    Open chrome://flags → "Insecure origins treated as secure" → add this origin → relaunch.<br><br>
    <b>Option B:</b> Ask your teacher to use ngrok (HTTPS tunnel).<br><br>
    <b>Option C:</b> Use the manual check-in below.
  </div>

  <label for="sname">Full name</label>
  <input class="submit-inp" type="text" id="sname" placeholder="e.g. Amina Benali" autocomplete="name">

  <label for="ssid">Student ID</label>
  <input class="submit-inp" type="text" id="ssid" placeholder="e.g. 231234">

  <button class="sbtn-primary" id="sbtn" onclick="doSub()" disabled>Getting location...</button>
  <div class="serr" id="serr-msg"></div>

  <hr class="sdivider">
  <div class="smanual-section" id="smanual-section">
    <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:rgba(245,158,11,.8);margin-bottom:8px;">
      GPS unavailable — manual check-in
    </div>
    <button class="sbtn-secondary" onclick="doManSub()">
      I am physically present in the amphitheatre
    </button>
    <div class="strust-note">Location will be estimated — subject to teacher verification</div>
  </div>
</div>
</div>

<script>
(function() {{
  var gLat = null, gLon = null, gAcc = null;
  var TOKEN = {repr(token)};

  function navigate(params) {{
    var url = window.location.origin + '/?' + params.toString();
    window.location.href = url;
  }}

  function setGPS(lat, lon, acc) {{
    gLat = lat; gLon = lon; gAcc = acc;
    var box = document.getElementById('gps-status');
    box.className = 'gps-box gps-ok';
    box.textContent = 'GPS acquired — +/-' + Math.round(acc) + ' m accuracy';
    var btn = document.getElementById('sbtn');
    btn.disabled = false;
    btn.textContent = 'Submit Attendance';
    document.getElementById('smanual-section').classList.remove('show');
    document.getElementById('sfix-box').classList.remove('show');
  }}

  function gpsError(err) {{
    var box = document.getElementById('gps-status');
    box.className = 'gps-box gps-err';
    var blocked = (window.location.protocol === 'http:' &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1');
    if (blocked || (err.code === 1 && err.message.toLowerCase().indexOf('secure') !== -1)) {{
      box.textContent = 'GPS blocked — requires HTTPS (see options below)';
      document.getElementById('sfix-box').classList.add('show');
    }} else {{
      box.textContent = 'GPS error: ' + err.message;
    }}
    document.getElementById('smanual-section').classList.add('show');
  }}

  var insecure = (window.location.protocol === 'http:' &&
      window.location.hostname !== 'localhost' &&
      window.location.hostname !== '127.0.0.1');

  if (!navigator.geolocation) {{
    document.getElementById('gps-status').className = 'gps-box gps-err';
    document.getElementById('gps-status').textContent = 'Geolocation not supported';
    document.getElementById('smanual-section').classList.add('show');
  }} else if (insecure) {{
    document.getElementById('sfix-box').classList.add('show');
    document.getElementById('smanual-section').classList.add('show');
    document.getElementById('gps-status').className = 'gps-box gps-err';
    document.getElementById('gps-status').textContent = 'GPS blocked — requires HTTPS';
    navigator.geolocation.getCurrentPosition(
      function(p) {{ setGPS(p.coords.latitude, p.coords.longitude, p.coords.accuracy); }},
      gpsError, {{enableHighAccuracy:true,timeout:8000,maximumAge:0}}
    );
  }} else {{
    navigator.geolocation.getCurrentPosition(
      function(p) {{ setGPS(p.coords.latitude, p.coords.longitude, p.coords.accuracy); }},
      gpsError, {{enableHighAccuracy:true,timeout:20000,maximumAge:0}}
    );
  }}

  function validate() {{
    var name = document.getElementById('sname').value.trim();
    var sid  = document.getElementById('ssid').value.trim();
    var err  = document.getElementById('serr-msg');
    err.textContent = '';
    if (!name) {{ err.textContent = 'Please enter your full name.'; return null; }}
    if (!sid)  {{ err.textContent = 'Please enter your student ID.'; return null; }}
    return {{name: name, sid: sid}};
  }}

  window.doSub = function() {{
    var v = validate(); if (!v) return;
    if (gLat === null) {{
      document.getElementById('serr-msg').textContent = 'GPS not ready — use manual check-in below.';
      return;
    }}
    navigate(new URLSearchParams({{
      mode: 'result', tok: TOKEN,
      name: v.name, sid: v.sid,
      lat: gLat.toFixed(7), lon: gLon.toFixed(7), acc: gAcc.toFixed(1)
    }}));
  }};

  window.doManSub = function() {{
    var v = validate(); if (!v) return;
    navigate(new URLSearchParams({{
      mode: 'result', tok: TOKEN,
      name: v.name, sid: v.sid,
      lat: '0.0', lon: '0.0', acc: '999', manual: '1'
    }}));
  }};
}})();
</script>
""", unsafe_allow_html=True)

    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  TEACHER DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
n_scans    = len(st.session_state.history)
loaded_kys = [k for k in VOTER_KEYS if MODELS.get(k)]
active_tag = f"voting ({', '.join(loaded_kys)})" if loaded_kys else "centroid fallback"

st.markdown(f"""
<div class="app-header">
  <div>
    <div class="app-title">Amphi<span>Locator</span></div>
    <div class="app-subtitle">ENSIA &nbsp;·&nbsp; GPS Classifier &nbsp;·&nbsp; v3.2</div>
  </div>
  <div class="header-badges">
    <span class="badge live"><span class="live-indicator"></span>Online</span>
    <span class="badge">{active_tag}</span>
    <span class="badge">8 amphitheatres</span>
    <span class="badge">{n_scans} scans &nbsp;·&nbsp; session {st.session_state.sid}</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_log, tab_qr, tab_attendance, tab_about = st.tabs(
    ["Detect", "Scan Log", "QR Code", "Attendance CSV", "About"]
)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — DETECT
# ─────────────────────────────────────────────────────────────────────────────
with tab_detect:
    history = st.session_state.history
    total   = len(history)
    inside  = sum(1 for r in history if r["label"] != "Outside")
    avg_d   = sum(r["dist"] for r in history) / total if total else 0

    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card s-purple"><div class="stat-val">{total}</div><div class="stat-key">Total Scans</div></div>
      <div class="stat-card s-green"><div class="stat-val">{inside}</div><div class="stat-key">Inside Amphis</div></div>
      <div class="stat-card s-cyan"><div class="stat-val">{total-inside}</div><div class="stat-key">Outside</div></div>
      <div class="stat-card s-amber"><div class="stat-val">{avg_d:.0f}m</div><div class="stat-key">Avg Distance</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="slabel">GPS Coordinates</div>', unsafe_allow_html=True)
        raw_input = st.text_input("coords_paste",
                                  placeholder="36.688350, 2.866760",
                                  label_visibility="collapsed", key="raw_coords")
        _err = ""; _lat_d, _lon_d = 36.6883500, 2.8664000
        if raw_input.strip():
            nums = _re.findall(r"[-+]?\d+\.?\d*", raw_input)
            if len(nums) >= 2:
                try: _lat_d, _lon_d = float(nums[0]), float(nums[1])
                except ValueError: _err = "Could not parse - try: 36.688350, 2.866760"
            else: _err = "Need two numbers (lat, lon)"
        if _err:
            st.markdown(f'<div style="font-size:.74rem;color:#f87171;font-family:JetBrains Mono,monospace;">{_err}</div>', unsafe_allow_html=True)
        elif raw_input.strip():
            st.markdown(f'<div style="font-size:.72rem;color:rgba(74,222,128,.75);font-family:JetBrains Mono,monospace;">Parsed: {_lat_d:.7f}, {_lon_d:.7f}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude",  value=_lat_d, format="%.7f", min_value=36.60, max_value=36.76, key="lat_fine")
        lon = c2.number_input("Longitude", value=_lon_d, format="%.7f", min_value=2.80,  max_value=2.90,  key="lon_fine")
        acc = st.number_input("Accuracy (metres)", value=22.0, min_value=1.0, max_value=500.0)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Classify Location", use_container_width=True, type="primary"):
            r = classify(lat, lon, acc)
            r.update(dict(lat=lat, lon=lon, accuracy=acc,
                          ts=datetime.now().strftime("%H:%M:%S"),
                          source="manual", student_name="", student_id=""))
            st.session_state.result = r
            st.session_state.history.append(r)
            st.rerun()

        st.markdown('<div class="qt-label">Quick test — jump to centroid</div>', unsafe_allow_html=True)
        row1, row2 = st.columns(4), st.columns(4)
        for i, (name, (clat, clon)) in enumerate(CENTROIDS.items()):
            col = row1[i] if i < 4 else row2[i-4]
            if col.button(name.replace("Amphi ","A"), use_container_width=True, key=f"qt_{name}"):
                r = classify(clat, clon, 15.0)
                r.update(dict(lat=clat, lon=clon, accuracy=15.0,
                              ts=datetime.now().strftime("%H:%M:%S"),
                              source="test", student_name="", student_id=""))
                st.session_state.result = r
                st.session_state.history.append(r)
                st.rerun()

    with col_r:
        st.markdown('<div class="slabel">Classification Result</div>', unsafe_allow_html=True)
        r = st.session_state.result
        if r is None:
            st.markdown('<div class="empty-state"><div class="es-icon">--</div><div class="es-title">No result yet</div><div class="es-sub">Enter coordinates and press Classify</div></div>', unsafe_allow_html=True)
        else:
            is_in     = r["label"] != "Outside"
            cls       = "result-inside" if is_in else "result-outside"
            lbl_text  = "CLASSIFIED INSIDE" if is_in else "CLASSIFIED OUTSIDE"
            fl = r.get("floor")
            floor_html = f'<div class="floor-tag {"f1" if fl==1 else "f2"}">Floor {fl}</div>' if fl else ""

            vb = r.get("vote_breakdown", {})
            chips = "".join(f'<span class="vote-chip {"winner" if p==r["label"] else ""}">{k.upper()}: {p}</span>'
                            for k, p in vb.items())
            vote_html = f'<div class="vote-grid">{chips}</div>' if chips else ""

            warn_html = ""
            if r.get("floor_pair_split"):
                pair = FLOOR_PAIRS.get(r["label"],"")
                warn_html = f'<div class="conf-warning warn-floor">Floor-pair ambiguity - votes split between {r["label"]} &amp; {pair}. GPS cannot distinguish Floor 1 from Floor 2.</div>'
            elif r.get("low_agreement"):
                warn_html = f'<div class="conf-warning warn-low">Low model agreement ({int(r.get("agreement_pct",0)*100)}% consensus). Consider re-scanning.</div>'

            vu = r.get("voters_used", 0)
            badge = f"majority vote - {vu} models" if vu else "centroid fallback"

            st.markdown(
                f'<div class="result-card {cls}">'
                f'<div class="result-label">{lbl_text}</div>'
                f'<div class="result-name">{r["label"]}</div>'
                f'{floor_html}<div class="model-badge">{badge}</div>'
                f'{vote_html}{warn_html}'
                f'<div class="result-meta" style="margin-top:12px;">'
                f'<b>{r["dist"]:.1f} m</b> to centroid'
                f' &nbsp;·&nbsp; GPS +/-<b>{r.get("accuracy",25):.0f} m</b>'
                f' &nbsp;·&nbsp; {r.get("source","—")}'
                f'</div>'
                f'<div class="result-coords">{r.get("lat",0):.7f}, {r.get("lon",0):.7f} &nbsp;·&nbsp; {r.get("ts","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="slabel" style="margin-top:1.8rem;">Class Probabilities</div>', unsafe_allow_html=True)
            st.markdown('<div class="conf-block">', unsafe_allow_html=True)
            for name, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
                c = COLORS.get(name, "#475569"); w = f"{prob*100:.1f}%"
                st.markdown(
                    f'<div class="conf-item">'
                    f'<span class="conf-label">{name}</span>'
                    f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{w};background:{c};opacity:.85;"></div></div>'
                    f'<span class="conf-pct">{w}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — SCAN LOG
# ─────────────────────────────────────────────────────────────────────────────
with tab_log:
    history = st.session_state.history
    if not history:
        st.markdown('<div class="empty-state" style="margin-top:1rem;"><div class="es-icon">--</div><div class="es-title">No scans yet</div><div class="es-sub">Use Detect or wait for student submissions</div></div>', unsafe_allow_html=True)
    else:
        col_log, col_charts = st.columns([1, 1.1], gap="large")
        with col_log:
            d_col, e_col = st.columns([1.4, 1])
            with d_col:
                st.session_state.dedup_enabled = st.checkbox(
                    "Keep latest per student ID", value=st.session_state.dedup_enabled)
            disp = history.copy()
            if st.session_state.dedup_enabled:
                seen = {}
                for rec in history: seen[rec.get("student_id", rec.get("source",""))] = rec
                disp = list(seen.values())
            with e_col:
                st.download_button(
                    label=f"Export {len(disp)} rows",
                    data=session_to_csv(history, dedup=st.session_state.dedup_enabled),
                    file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True,
                )
            st.markdown(f'<div class="slabel">Session log - {len(disp)} entries</div>', unsafe_allow_html=True)
            st.markdown('<div class="log-wrap">', unsafe_allow_html=True)
            for rec in reversed(disp[-50:]):
                c    = COLORS.get(rec["label"],"#475569")
                stu  = rec.get("student_name","")
                sid2 = rec.get("student_id","")
                tag  = f"{stu} - {sid2}" if stu else rec.get("source","—")
                warn = " [!]" if rec.get("low_agreement") or rec.get("floor_pair_split") else ""
                st.markdown(
                    f'<div class="log-row">'
                    f'<span class="log-ts">{rec.get("ts","—")}</span>'
                    f'<span class="log-dot" style="background:{c};"></span>'
                    f'<span class="log-name" style="color:{c};">{rec["label"]}{warn}</span>'
                    f'<span class="log-stu">{tag}</span>'
                    f'<span class="log-d">{rec["dist"]:.0f} m</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("Clear session log", use_container_width=True):
                st.session_state.history = []; st.session_state.result = None; st.rerun()

        with col_charts:
            df_h = pd.DataFrame(disp)
            cnts = df_h["label"].value_counts().reset_index()
            cnts.columns = ["label","count"]
            st.markdown('<div class="slabel">Distribution</div>', unsafe_allow_html=True)
            fig_b = go.Figure(go.Bar(
                x=cnts["label"], y=cnts["count"],
                marker_color=[COLORS.get(l,"#475569") for l in cnts["label"]],
                marker_opacity=.82, text=cnts["count"], textposition="outside",
                textfont=dict(color="rgba(255,255,255,.45)", size=11), marker_line_width=0,
            ))
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)",
                font=dict(color="rgba(255,255,255,.4)", family="Inter", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,.04)", showline=False, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(255,255,255,.04)", showline=False, title=""),
                height=240, margin=dict(t=10,b=0,l=0,r=10), bargap=.35,
            )
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="slabel" style="margin-top:1.2rem;">Distance over time</div>', unsafe_allow_html=True)
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=list(range(1,len(df_h)+1)), y=df_h["dist"], mode="lines",
                line=dict(color="rgba(129,140,248,.3)", width=1.5),
                fill="tozeroy", fillcolor="rgba(129,140,248,.04)", showlegend=False,
            ))
            fig_t.add_trace(go.Scatter(
                x=list(range(1,len(df_h)+1)), y=df_h["dist"], mode="markers",
                marker=dict(color=[COLORS.get(l,"#475569") for l in df_h["label"]],
                            size=8, line=dict(width=1.5, color="rgba(255,255,255,.15)")),
                text=df_h["label"], hovertemplate="<b>%{text}</b><br>%{y:.1f} m<extra></extra>",
                showlegend=False,
            ))
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.015)",
                font=dict(color="rgba(255,255,255,.4)", family="Inter", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,.04)", showline=False, title=dict(text="Scan #",font=dict(size=10))),
                yaxis=dict(gridcolor="rgba(255,255,255,.04)", showline=False, title=dict(text="Distance (m)",font=dict(size=10))),
                height=220, margin=dict(t=10,b=0,l=0,r=10),
            )
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — QR CODE
# ─────────────────────────────────────────────────────────────────────────────
with tab_qr:
    st.markdown('<div class="slabel">Rotating QR Code - Anti-Cheat</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.82rem;color:rgba(255,255,255,.4);margin-bottom:1.4rem;line-height:1.8;">'
        f'This QR code rotates every <b style="color:#c7d2fe;">{TOKEN_WINDOW} seconds</b>. '
        f'A screenshot sent to an absent student will be rejected once the window expires. '
        f'Project it fullscreen — students scan, fill in name + ID, GPS is captured automatically.'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_q1, col_q2 = st.columns([1, 1.2], gap="large")

    with col_q1:
        st.markdown('<div class="slabel">Settings</div>', unsafe_allow_html=True)
        qr_host = st.text_input("Server IP / hostname", value=ip, key="qr_host",
                                help="Use your ngrok/cloudflared HTTPS hostname here so GPS works on phones.")
        qr_port = st.number_input("Port", value=8501, min_value=1, max_value=65535, step=1, key="qr_port")
        qr_https = st.checkbox("Use HTTPS (tunnel/SSL)", value=False, key="qr_https",
                               help="Check this if you are using ngrok or a reverse proxy with HTTPS.")

        # Strip any scheme the user accidentally pasted into the hostname field
        _host_clean = qr_host.strip()
        if _host_clean.startswith("https://"):
            _host_clean = _host_clean[len("https://"):]
            # Also auto-enable HTTPS checkbox if not already on
        elif _host_clean.startswith("http://"):
            _host_clean = _host_clean[len("http://"):]
        # Remove any trailing slashes or path
        _host_clean = _host_clean.rstrip("/").split("/")[0]

        scheme    = "https" if qr_https else "http"
        port_part = f":{int(qr_port)}" if not (qr_https and int(qr_port) == 443) and not (not qr_https and int(qr_port) == 80) else ""
        tok_now   = current_token()
        secs_left = seconds_until_next_token()
        pct_left  = int(secs_left / TOKEN_WINDOW * 100)
        sub_url   = f"{scheme}://{_host_clean}{port_part}/?mode=submit&token={tok_now}"

        st.markdown(
            f'<div class="token-bar-wrap">'
            f'Token expires in <b style="color:#c7d2fe;">{secs_left}s</b> &nbsp;·&nbsp; '
            f'window <code style="color:rgba(103,232,249,.7);">{tok_now}</code>'
            f'<div class="token-bar-inner" style="width:{pct_left}%;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if _host_clean != qr_host.strip().rstrip("/"):
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:.66rem;'
                f'color:rgba(251,191,36,.85);background:rgba(245,158,11,.07);'
                f'border:1px solid rgba(245,158,11,.25);border-radius:7px;padding:7px 12px;margin-top:8px;">'
                f'Scheme stripped from hostname → using: <b>{_host_clean}</b></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="qr-url" style="margin-top:10px;">{sub_url}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.button("Refresh QR now", use_container_width=True):
            st.rerun()

    with col_q2:
        st.markdown('<div class="slabel">Current QR</div>', unsafe_allow_html=True)
        png_bytes = make_qr_png(sub_url)
        st.markdown('<div class="qr-panel">', unsafe_allow_html=True)
        st.image(png_bytes, width=280)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Download QR as PNG",
            data=png_bytes,
            file_name=f"qr_{tok_now}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.components.v1.html(f"""
<div id="cd" style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
     color:rgba(255,255,255,.25);text-align:center;padding:8px 0;">
  Next token in <span id="t">{secs_left}</span>s &nbsp;·&nbsp; press "Refresh QR now" to update
</div>
<script>
var s = {secs_left};
var iv = setInterval(function() {{
  s--;
  var el = document.getElementById('t');
  if (el) el.textContent = Math.max(s, 0);
  if (s <= 0) {{
    clearInterval(iv);
    var el2 = document.getElementById('cd');
    if (el2) {{ el2.textContent = 'Token expired - press Refresh QR now'; el2.style.color = 'rgba(252,165,165,.6)'; }}
  }}
}}, 1000);
</script>
""", height=40)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — ATTENDANCE CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab_attendance:
    st.markdown('<div class="slabel">Persistent Attendance Log</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.82rem;color:rgba(255,255,255,.4);margin-bottom:1.4rem;line-height:1.8;">'
        f'All student submissions are appended to '
        f'<code style="color:#67e8f9;">{ATTENDANCE_CSV.name}</code>. '
        f'This file survives restarts and accumulates across every class session.'
        f'</div>',
        unsafe_allow_html=True,
    )
    df_full = load_full_csv()

    if df_full.empty:
        st.markdown('<div class="empty-state"><div class="es-icon">--</div><div class="es-title">No attendance records yet</div><div class="es-sub">Records appear here as students submit via the QR code</div></div>', unsafe_allow_html=True)
    else:
        total_r  = len(df_full)
        uniq_s   = df_full["student_id"].nunique() if "student_id" in df_full.columns else 0
        n_dates  = df_full["session_date"].nunique() if "session_date" in df_full.columns else 0
        in_r     = (df_full["label"] != "Outside").sum() if "label" in df_full.columns else 0

        st.markdown(f"""
        <div class="stats-row">
          <div class="stat-card s-purple"><div class="stat-val">{total_r}</div><div class="stat-key">Total Records</div></div>
          <div class="stat-card s-green"><div class="stat-val">{uniq_s}</div><div class="stat-key">Unique Students</div></div>
          <div class="stat-card s-cyan"><div class="stat-val">{n_dates}</div><div class="stat-key">Sessions</div></div>
          <div class="stat-card s-amber"><div class="stat-val">{in_r}</div><div class="stat-key">Inside Detected</div></div>
        </div>
        """, unsafe_allow_html=True)

        if "session_date" in df_full.columns:
            all_dates = sorted(df_full["session_date"].dropna().unique(), reverse=True)
            chosen    = st.selectbox("Filter by session date", ["All sessions"]+list(all_dates))
            df_view   = df_full if chosen == "All sessions" else df_full[df_full["session_date"]==chosen]
        else:
            df_view = df_full

        st.dataframe(df_view, use_container_width=True, hide_index=True)

        dl_col, clr_col = st.columns([2,1])
        with dl_col:
            st.download_button(
                label=f"Download full attendance CSV ({total_r} records)",
                data=read_csv_bytes(),
                file_name=f"attendance_full_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with clr_col:
            if st.button("Clear ALL records", use_container_width=True):
                if ATTENDANCE_CSV.exists(): ATTENDANCE_CSV.unlink()
                ensure_csv(); st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns([1.5, 1], gap="large")

    with col_a:
        st.markdown('<div class="slabel">Project</div>', unsafe_allow_html=True)
        loaded_str  = ", ".join(loaded_kys) if loaded_kys else "none"
        missing_all = [k for k in VOTER_KEYS+["outside","label_enc"] if not MODELS.get(k)]
        missing_str = ", ".join(missing_all) if missing_all else "none"

        st.markdown(
            '<div class="about-body">'
            '<h4>AmphiLocator v3.2</h4>'
            'ENSIA Machine Learning Project - Spring 2025-2026'
            '<h4>Ensemble voting</h4>'
            f'Loaded: <code>{loaded_str}</code> | Missing: <code>{missing_str}</code><br>'
            'Each model votes; majority wins. Probabilities averaged across models with predict_proba.'
            '<h4>Anti-cheat QR rotation</h4>'
            f'Token window: <b>{TOKEN_WINDOW} s</b>. The QR encodes a HMAC-SHA256 token of the current time window. '
            'Expired tokens are rejected immediately.'
            '<h4>GPS requirement (HTTPS)</h4>'
            'Browsers block navigator.geolocation on plain HTTP origins (non-localhost). '
            'To enable GPS on student phones: run ngrok to get an HTTPS URL, '
            'paste the hostname in the QR Code tab, and check "Use HTTPS".'
            '<h4>Persistence</h4>'
            f'Attendance is appended to <code>{ATTENDANCE_CSV.name}</code> on every submission.'
            '<h4>Limitation</h4>'
            'GPS cannot distinguish Floor 1 from Floor 2 (Amphi 1/5, 2/6, 3/7, 4/8 share footprint).'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Centroid Table</div>', unsafe_allow_html=True)
        rows = [{"Amphi": n, "Floor": FLOOR[n], "Lat": f"{lt:.6f}", "Lon": f"{lg:.6f}"}
                for n, (lt, lg) in CENTROIDS.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="slabel">Amphitheatre Reference</div>', unsafe_allow_html=True)
        for name, (clat, clon) in CENTROIDS.items():
            fl = FLOOR[name]; fc = "rf1" if fl==1 else "rf2"; c = COLORS[name]
            is_cur = st.session_state.result and st.session_state.result["label"]==name
            hl = "background:rgba(255,255,255,.04);" if is_cur else ""
            st.markdown(
                f'<div class="ref-item" style="{hl}">'
                f'<span class="ref-dot" style="background:{c};"></span>'
                f'<span class="ref-name">{name}</span>'
                f'<span class="ref-floor {fc}">Floor {fl}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Data Distribution (v2)</div>', unsafe_allow_html=True)
        sample = {"Amphi 2":2564,"Amphi 5":1226,"Amphi 8":999,"Outside":876,
                  "Amphi 6":774,"Amphi 4":741,"Amphi 1":604,"Amphi 3":318,"Amphi 7":122}
        fig_pie = go.Figure(go.Pie(
            labels=list(sample.keys()), values=list(sample.values()),
            marker_colors=[COLORS.get(l,"#475569") for l in sample],
            marker=dict(line=dict(color="#09090f",width=2)),
            hole=.65, textinfo="label+percent",
            textfont=dict(size=9.5, color="rgba(255,255,255,.6)"),
            showlegend=False, pull=[.03]*len(sample),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=0,b=0,l=0,r=0),
            annotations=[
                dict(text="8,224",x=.5,y=.55,font=dict(size=18,color="rgba(255,255,255,.7)",family="Space Grotesk"),showarrow=False),
                dict(text="readings",x=.5,y=.4,font=dict(size=10,color="rgba(255,255,255,.25)",family="Inter"),showarrow=False),
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})