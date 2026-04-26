#!/usr/bin/env python3
"""Data cleaning and preparation pipeline for AmphiLocator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "target_label"


LABEL_CANONICAL_MAP: Dict[str, str] = {
    "amphi 1": "Amphi 1",
    "amphi 2": "Amphi 2",
    "amphi 3": "Amphi 3",
    "amphi 4": "Amphi 4",
    "amphi 5": "Amphi 5",
    "amphi 6": "Amphi 6",
    "amphi 7": "Amphi 7",
    "amphi 8": "Amphi 8",
    "lab 1": "Amphi 1",
    "lab 2": "Amphi 2",
    "lab 3": "Amphi 3",
    "lab 4": "Amphi 4",
    "lab 5": "Amphi 5",
    "lab 6": "Amphi 6",
    "lab 7": "Amphi 7",
    "lab 8": "Amphi 8",
    "outside": "Outside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split AmphiLocator data.")
    parser.add_argument(
        "--input",
        default="data/raw/gps_data_v2.csv",
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for cleaned and split datasets.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Train split fraction.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    return parser.parse_args()


def _canonicalize_label(amphitheatre: str, is_outside: object) -> str:
    label = str(amphitheatre).strip().lower()
    if label in LABEL_CANONICAL_MAP:
        return LABEL_CANONICAL_MAP[label]

    outside_value = str(is_outside).strip().lower()
    if outside_value in {"true", "1", "yes"}:
        return "Outside"

    return "Outside"


def _remove_outliers_iqr(df: pd.DataFrame, columns: List[str], k: float = 1.5) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= series.between(lower, upper, inclusive="both")
    return df.loc[mask].copy()


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Deduplicate rows and remove duplicate IDs by keeping latest timestamp.
    df = df.drop_duplicates().copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    df = df.dropna(subset=["latitude_mean", "longitude_mean", "timestamp"])
    df = df.sort_values("timestamp")
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")

    # Ensure numeric consistency.
    for col in ["latitude_mean", "longitude_mean", "accuracy_mean", "gps_variance", "sample_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic coordinate validity.
    df = df[df["latitude_mean"].between(-90, 90)]
    df = df[df["longitude_mean"].between(-180, 180)]

    # Canonical target labels.
    df[TARGET_COLUMN] = [
        _canonicalize_label(amphitheatre, is_outside)
        for amphitheatre, is_outside in zip(df.get("amphitheatre", ""), df.get("is_outside", ""))
    ]

    # Noise filtering: remove very poor-quality GPS points.
    if "accuracy_mean" in df.columns:
        acc_cap = df["accuracy_mean"].quantile(0.99)
        df = df[(df["accuracy_mean"].isna()) | (df["accuracy_mean"] <= acc_cap)]
    if "gps_variance" in df.columns:
        var_cap = df["gps_variance"].quantile(0.99)
        df = df[(df["gps_variance"].isna()) | (df["gps_variance"] <= var_cap)]

    # Outlier filtering using IQR for location/quality features.
    outlier_cols = ["latitude_mean", "longitude_mean", "accuracy_mean", "gps_variance"]
    df = _remove_outliers_iqr(df, outlier_cols, k=2.0)

    # Keep classes with enough samples to support stratified train/val/test.
    class_counts = df[TARGET_COLUMN].value_counts()
    valid_classes = class_counts[class_counts >= 3].index
    df = df[df[TARGET_COLUMN].isin(valid_classes)].copy()

    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Dict[str, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    val_ratio_within_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_within_temp),
        stratify=temp_df[TARGET_COLUMN],
        random_state=random_state,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def scale_numeric_features(splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    numeric_cols = [
        "latitude_mean",
        "longitude_mean",
        "accuracy_mean",
        "gps_variance",
        "sample_count",
    ]
    available_numeric_cols = [col for col in numeric_cols if col in splits["train"].columns]

    scaled = {name: frame.copy() for name, frame in splits.items()}
    if not available_numeric_cols:
        return scaled

    scaler = StandardScaler()
    train_features = splits["train"][available_numeric_cols].fillna(0.0)
    scaler.fit(train_features)

    for split_name in ("train", "val", "test"):
        features = splits[split_name][available_numeric_cols].fillna(0.0)
        scaled_values = scaler.transform(features)
        for i, col in enumerate(available_numeric_cols):
            scaled[split_name][f"{col}_scaled"] = scaled_values[:, i]

    return scaled


def save_outputs(
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    raw_splits: Dict[str, pd.DataFrame],
    scaled_splits: Dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_dir / "gps_data_clean.csv", index=False)
    for split_name, frame in raw_splits.items():
        frame.to_csv(output_dir / f"{split_name}.csv", index=False)
    for split_name, frame in scaled_splits.items():
        frame.to_csv(output_dir / f"{split_name}_scaled.csv", index=False)

    summary = {
        "rows_cleaned": int(len(cleaned_df)),
        "class_distribution": cleaned_df[TARGET_COLUMN].value_counts().to_dict(),
        "split_sizes": {name: int(len(frame)) for name, frame in raw_splits.items()},
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df_raw = pd.read_csv(input_path)
    df_clean = prepare_dataset(df_raw)
    splits = split_dataset(
        df_clean,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    scaled_splits = scale_numeric_features(splits)
    save_outputs(output_dir, df_clean, splits, scaled_splits)

    print(f"Cleaned dataset saved to: {output_dir / 'gps_data_clean.csv'}")
    print(f"Train/Val/Test sizes: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print("Class distribution:")
    print(df_clean[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
