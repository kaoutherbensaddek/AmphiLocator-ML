# AmphiLocator - Amphitheatre Localization System
**ENSIA Machine Learning Project | Spring 2025–2026**

A machine learning system that identifies which of ENSIA's 8 amphitheatres a student is in - or detects they are outside - using only GPS data. This is a 9-class classification problem (Amphi 1–8 + Outside) under realistic indoor GPS noise conditions.

---

## Problem Statement

ENSIA's amphitheatres are stacked on two floors - Amphi 1–4 (floor 1), Amphi 5–8 (floor 2) - sharing the same horizontal GPS footprint. Paired centroids (1/5, 2/6, 3/7, 4/8) are less than 15 m apart, while indoor GPS accuracy is typically 15–100 m. Raw coordinates cannot disambiguate them; the system relies on engineered spatial features and GPS quality signals.

---

## Repository Structure

```
AmphiLocator/
├── data/
│   ├── raw/                        # Original GPS collection (gps_data_v2.csv)
│   └── processed/
│       ├── feature_cols.json
│       ├── train/train.csv, train_ready.csv
│       ├── val/val.csv, val_ready.csv
│       └── test/test.csv, test_ready.csv
├── interface/app.py                # Streamlit web interface
├── models/
│   ├── gbm_best.pkl                # Primary model (HistGradientBoosting)
│   ├── gbm_metadata.json
│   ├── knn_model.pkl
│   ├── lr_model.pkl
│   ├── dt_tuned.pkl
│   ├── rf_tuned.pkl
│   ├── outside_detector.pkl        # Binary outside/inside classifier
│   └── label_encoder.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_KNN_LogReg.ipynb
│   ├── 05_DecisionTree_RF.ipynb
│   └── 06_GradientBoosting.ipynb
├── results/figures/
├── scripts/
│   ├── prepare_dataset.py
│   └── feature_engineering.py
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total clean samples | 8,224 |
| Final classes | 9 (Amphi 1–8, Outside) |
| Smallest class | Amphi 7 (~72 training samples) |
| GPS accuracy indoors | 15–100 m typical |
| Core signal | `latitude_mean`, `longitude_mean`, `accuracy_mean` |

Raw labels (32+ variants) were canonicalized to 9 classes; junk entries and off-campus coordinates were removed.

---

## Pipeline

### 1. Preprocessing
```bash
python scripts/prepare_dataset.py
```
- Remove duplicates, validate timestamps, coerce GPS columns
- Canonicalize labels → `Amphi 1`–`Amphi 8`, `Outside`
- GPS noise filtering (99th percentile of `accuracy_mean` and `gps_variance`)
- IQR outlier removal (k=2.0) on spatial and quality columns
- Stratified train/val/test split; StandardScaler fit on train only

### 2. Feature Engineering
```bash
python scripts/feature_engineering.py
```

**Spatial features**

| Feature | Description |
|---------|-------------|
| `dist_Amphi_1` … `dist_Amphi_8` | Haversine distance to each centroid |
| `dist_nearest` | Distance to closest centroid |
| `dist_2nd` | Distance to second-closest centroid |
| `dist_gap` | `dist_2nd − dist_nearest` - small = high ambiguity |

Centroids are computed from training data only to prevent leakage.

**GPS quality features**

| Feature | Description |
|---------|-------------|
| `log_accuracy` | Log-transformed accuracy (compresses right tail) |
| `accuracy_bin` | 0 (≤30 m), 1 (30–80 m), 2 (>80 m) |
| `high_accuracy_flag` | 1 if accuracy < 30 m |

**Excluded features** - `is_outside` (label leakage), `nearest_amphi_enc` (label proxy), seat features (gameable), time/day features (encode collection schedule, not location), `gps_variance` (constant after filtering).

---

## Models & Results

All models tuned on training set via cross-validation. Test set touched exactly once per model.

| Model | Val Acc | Val F1-macro | Test Acc | Test F1-macro |
|-------|---------|--------------|----------|---------------|
| KNN (k=5) | 0.9891 | 0.9888 | 0.9854 | 0.9851 |
| Logistic Regression (C=5) | 0.8942 | 0.8950 | 0.8540 | 0.8527 |
| Decision Tree (tuned) | 0.9927 | 0.9925 | 0.9964 | 0.9963 |
| Random Forest (tuned) | 0.9964 | 0.9963 | 1.0000 | 1.0000 |
| **GBM - HistGradientBoosting** | **0.9930** | **0.9905** | **0.9950** | **0.9886** |

**GBM best hyperparameters:** `learning_rate=0.1`, `max_iter=100`, `max_depth=7` (tuned via GridSearchCV, 27 candidates × 5-fold stratified CV, scoring = `f1_macro`).

**GBM cross-validation:** `f1_macro` mean = **0.9885 ± 0.0033** - consistent with val/test, no sign of overfitting.

**Per-class F1 on test (GBM)**

| Class | F1 |
|-------|----|
| Amphi 2, 4, 6, 8 | 1.0000 |
| Amphi 5 | 0.9942 |
| Amphi 1 | 0.9885 |
| Amphi 3 | 0.9877 |
| Amphi 7 | 0.9677 |
| Outside | 0.9593 |

Weakest classes are Outside and Amphi 7 - both have the fewest training samples.

---

## Outside Detection

A dedicated binary classifier (`outside_detector.pkl`) runs as a gate before amphitheatre prediction. If it predicts "outside", the pipeline returns Outside immediately.

| Method | Overall Acc | Macro F1 | Outside F1 |
|--------|-------------|----------|------------|
| Raw model | 0.9950 | 0.9886 | 0.9593 |
| Confidence threshold (τ=0.7) | 0.9960 | 0.9931 | 0.9677 |

The binary outside classifier was selected over the confidence-threshold approach and is what runs in the app. Its metrics are folded into the overall pipeline results reported above.

---

## Quickstart

```bash
git clone https://github.com/your-org/amphilocator-ml.git
cd amphilocator-ml
pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/feature_engineering.py
jupyter notebook notebooks/01_eda.ipynb
```

**Inference example:**
```python
import joblib, numpy as np

model   = joblib.load("models/gbm_best.pkl")
outside = joblib.load("models/outside_detector.pkl")
encoder = joblib.load("models/label_encoder.pkl")

is_outside = outside.predict(X)[0]
if is_outside:
    label = "Outside"
else:
    label = encoder.inverse_transform([model.predict(X)[0]])[0]
```

---

## Interface

A Streamlit web app (`interface/app.py`) for real-time GPS classification.

```bash
pip install streamlit pandas numpy plotly scikit-learn joblib
streamlit run interface/app.py   # opens at http://localhost:8501
```

**Features:**
- **Detect tab** - paste GPS coordinates, classify instantly. Shows predicted amphitheatre, floor, distance to centroid, model votes, and probability bars.
- **Ensemble voting** - GBM, KNN, LR, DT, RF all vote; majority wins.
- **Rotating QR Code** - regenerates every 20 seconds with HMAC-SHA256 token; screenshots expire.
- **HTTPS GPS support** - toggle for ngrok/Cloudflare tunnels (required for phone GPS).
- **Persistent attendance CSV** - saved to `data/attendance_log.csv`, survives restarts.
- **Manual fallback** - if GPS is blocked, students click "I am physically present".
- **Scan Log tab** - session history with distribution and distance-over-time charts.
- **About tab** - model status, centroid table, data distribution.

All `.pkl` files are loaded once at startup via `@st.cache_resource`. Missing models fall back to a haversine-centroid classifier.

---

## Classroom Usage Guide

### Setup
1. Connect the teacher's laptop to the classroom Wi-Fi (same network as students).
2. Run `streamlit run interface/app.py` and note the local URL (e.g. `http://192.168.1.35:8501`).
3. Confirm the **About tab** shows all models as loaded.

### Taking Attendance
1. In the **QR Code tab**, set the server IP/hostname, port (`8501`), and enable HTTPS if using ngrok.
2. Click **Refresh QR now** and project the code on screen.
3. Students scan → enter name and ID → allow location → click **Submit Attendance**.
4. Monitor submissions in the **Scan Log tab**.

If GPS fails, students click **"I am physically present"** - entry is marked as manual check-in.

### Interpreting Results

| Signal | Meaning |
|--------|---------|
| All vote chips agree | High confidence, reliable prediction |
| Chips disagree | GPS ambiguous - likely a floor-pair case (1/5, 2/6, 3/7, 4/8). Verify floor verbally. |
| "Outside" result | Outside detector fired - student not in any amphitheatre, or GPS too noisy |
| High single probability bar | Reliable. Flat bars = noisy GPS, treat with caution. |

Click **Clear history** in the Scan Log tab to reset a session.

---

## Key Design Decisions

- **No raw lat/lon in features** - replaced by distance-to-centroid features for better generalization to GPS drift.
- **Centroids from train only** - prevents spatial leakage into val/test.
- **Seat features excluded** - discriminative in collected data but gameable by students faking a seat number.
- **Time features excluded** - encode the 3rd-year collection schedule, not physical location. Any amphitheatre can be occupied at any hour in production.
- **Binary outside detector as gate** - cleanly separates the two decisions; avoids confidence-threshold fragility.
- **Ensemble majority vote** - more robust than a single model and makes disagreement transparent.
- **Macro F1 as primary metric** - prevents large classes (Amphi 2: 2,564 samples) from dominating evaluation over small ones (Amphi 7: 72).

---

## Known Limitations

1. **Stacked floor pairs** - GPS alone cannot reliably distinguish floors; the system predicts the column (1/5, 2/6, etc.), not the floor.
2. **Amphi 7 data scarcity** - only 72 training samples; more collection would improve its F1 (currently 0.9677).
3. **Collection bias** - data gathered almost entirely by 3rd-year students. Generalization to other years or irregular schedules is untested.
4. **Hardcoded scaling constants** - `build_features()` in `app.py` uses hardcoded mean/std. For production, save and load the actual `StandardScaler` from `03_feature_engineering.ipynb`.
5. **GPS boundary cases** - some genuine outside readings land close to amphitheatre centroids; irreducible given sensor resolution.
6. **Single-session memory** - the Scan Log lives in Streamlit session state and resets on page refresh. Export to CSV for persistent records.


## Demo 


https://github.com/user-attachments/assets/35dc15b5-278a-4d25-8d69-0eea47ac304e



