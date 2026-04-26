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
