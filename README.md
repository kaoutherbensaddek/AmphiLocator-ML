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

### *Feature Engineering*
Run the feature engineering script:

```python scripts/feature_engineering.py```


- computes amphitheatre centroids from training set only (prevents data leakage)
- creates distance features (Euclidean distance to each amphitheatre centroid)
- adds GPS quality features (log transforms, accuracy bins, high-accuracy flags)
- extracts temporal features (hour of day, day of week, weekend flag, cyclic encodings)
- encodes seat information (block, row, column with proper handling of missing values)
- converts string labels to integers for modeling
- scales engineered features for distance-based models

Generated files are saved in data/processed/:

| File | Description |
|------|------------|
| `train_fe.csv`, `val_fe.csv`, `test_fe.csv` | Engineered features (original scale) |
| `train_fe_scaled.csv`, `val_fe_scaled.csv`, `test_fe_scaled.csv` | Scaled engineered features |
| `centroids.json` | GPS centroids per amphitheatre |
| `feature_cols.json` | Feature names organized by category |
| `label_map.json` | String label → integer mapping |
| `scaler_fe.pkl` | Fitted StandardScaler for inference |

Feature categories (29 total):

- Distance features (11): distances to all 8 amphitheatres + nearest, 2nd, gap
- GPS quality features (5): log accuracy, log variance, accuracy bins, high accuracy flag
- Temporal features (5): hour, day of week, weekend, sin/cos encodings
- Seat features (4): has_seat flag, block encoding, row, column
- Raw GPS (4): mean latitude, mean longitude, mean accuracy, variance