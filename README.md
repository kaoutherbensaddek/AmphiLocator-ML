# ENSIA – Amphitheatre Localization System
Machine Learning Project | Spring 2025–2026

This project is developed as part of the Machine Learning course at ENSIA (National School of Artificial Intelligence). 

## Overview
This project develops a machine learning system that identifies which amphitheatre a student is located in at ENSIA using GPS data.  
The model classifies the student into one of 7–8 amphitheatres or detects if they are outside all amphitheatres while handling GPS noise and positioning errors.

## Objectives
- Collect and preprocess GPS data
- Engineer spatial features (averaged GPS, distance to centroids, variance, multiple readings)
- Train a multi-class classification model
- Detect outside cases reliably
- Evaluate performance using accuracy, precision, recall, and F1-score

## Methodology
1. Data collection in all amphitheatres and surrounding areas
2. Data cleaning, labeling, and visualization
3. Feature engineering
4. Model training 
5. Evaluation and validation on unseen data

## Data Preparation Pipeline
Run the preprocessing pipeline with:

`python scripts/prepare_dataset.py`

This pipeline:
- removes invalid rows, duplicates, and statistical outliers
- enforces coordinate and timestamp consistency
- filters high-noise GPS points
- creates canonical target labels (`Amphi 1` to `Amphi 8`, `Outside`)
- performs stratified train/validation/test split
- standardizes numeric features for modeling

Generated files are saved in `data/processed/`:
- `gps_data_clean.csv`
- `train.csv`, `val.csv`, `test.csv`
- `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`
- `preprocessing_summary.json`

## Feature Engineering Pipeline

Run the feature engineering pipeline with:

`python scripts/feature_engineering.py`

This pipeline constructs meaningful features from raw GPS data:

### Spatial Features
- **Distance to each amphitheatre** (`dist_Amphi_1` to `dist_Amphi_8`): Euclidean distance from GPS point to each amphitheatre centroid (centroids computed from training set only to prevent leakage)
- **Nearest amphitheatre distance** (`dist_nearest`): Distance to the closest amphitheatre
- **Second nearest distance** (`dist_2nd`): Distance to the second closest amphitheatre  
- **Ambiguity gap** (`dist_gap`): `dist_2nd - dist_nearest` — small gap indicates the point lies between two amphitheatres (high ambiguity)

### GPS Quality Features
- **Log accuracy** (`log_accuracy`): Log-transformed GPS accuracy (compresses the heavy right tail)
- **Accuracy bin** (`accuracy_bin`): Discretized accuracy (0: ≤30m good, 1: 30-80m ok, 2: >80m bad)
- **High accuracy flag** (`high_accuracy_flag`): Binary indicator for accuracy < 30m

### Seat Features (when available)
- **Has seat flag** (`has_seat`): Indicates whether seat information exists
- **Seat zone ID** (`seat_zone_id`): Unique identifier combining row and column (e.g., row 3, column 5 → 305)
- **Encoded seat block** (`seat_block_enc`): Left=0, Center=1, Right=2, Unknown=3
- **Filled row/column** (`seat_row_filled`, `seat_column_filled`): Missing values replaced with -1

### Temporal Features
- **Hour of day** (cyclic encoded): `hour_sin` and `hour_cos` — captures lecture schedule patterns
- **Note**: Day of week and weekend indicators were **excluded** because they encode data collection schedule, not real-world signal

### Label Encoding
- Target labels (`Amphi 1` to `Amphi 8`, `Outside`) are encoded as integers (0-8)

### Output Files
- `train_ready.csv`, `val_ready.csv`, `test_ready.csv`: Final feature-engineered datasets ready for modeling

### Key Design Decisions
- Centroids computed from **training set only** — prevents data leakage into validation/test
- Raw latitude/longitude are **dropped** after distance calculation — distance features provide cleaner spatial representation
- `gps_variance` dropped (constant 0 after preprocessing)
- Day-of-week features dropped (collection schedule bias)