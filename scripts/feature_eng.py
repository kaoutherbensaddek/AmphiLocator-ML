
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Create subdirectories
(PROCESSED / "train").mkdir(parents=True, exist_ok=True)
(PROCESSED / "val").mkdir(parents=True, exist_ok=True)
(PROCESSED / "test").mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_label"
AMPHI_LABELS = [f"Amphi {i}" for i in range(1, 9)]

# Columns to drop initially
DROP_COLS = [
    "id", "user", "year", "section", "amphitheatre", "module",
    "navigator_context", "screen_context", "network_information",
    "battery_status", "device_info", "raw_gps_readings",
    "collection_metadata", "created_at",
]


def load_splits():
    """Load train/val/test splits from data/processed/"""
    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED / f"{name}.csv")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        splits[name] = df
    print(f"Loaded splits: train={splits['train'].shape}, val={splits['val'].shape}, test={splits['test'].shape}")
    return splits["train"], splits["val"], splits["test"]


def drop_initial_cols(df):
    """Remove metadata columns not needed for feature engineering"""
    existing = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=existing)


def compute_centroids(train):
    """Compute centroids from training set only (no leakage)"""
    centroids = (
        train[train[TARGET_COL].isin(AMPHI_LABELS)]
        .groupby(TARGET_COL)[["latitude_mean", "longitude_mean"]]
        .mean()
        .rename(columns={"latitude_mean": "c_lat", "longitude_mean": "c_lon"})
    )
    print("\nCentroids (from train set):")
    print(centroids.round(6).to_string())
    return centroids


def add_distance_features(df, centroids):
    """Add Euclidean distance to each amphitheatre centroid"""
    df = df.copy()
    dist_cols = []

    for lbl, row in centroids.iterrows():
        col = f"dist_{lbl.replace(' ', '_')}"
        df[col] = np.sqrt(
            (df["latitude_mean"] - row["c_lat"]) ** 2 +
            (df["longitude_mean"] - row["c_lon"]) ** 2
        )
        dist_cols.append(col)

    # Summary distance features
    dist_matrix = df[dist_cols]
    df["dist_nearest"] = dist_matrix.min(axis=1)
    df["dist_2nd"] = dist_matrix.apply(lambda r: r.nsmallest(2).iloc[-1], axis=1)
    df["dist_gap"] = df["dist_2nd"] - df["dist_nearest"]

    # Nearest amphitheatre (string)
    df["nearest_amphi"] = dist_matrix.idxmin(axis=1).str.replace("dist_", "").str.replace("_", " ")

    return df, dist_cols


def encode_nearest_amphi(train, val, test):
    """Label encode the nearest_amphi column"""
    le = LabelEncoder()
    le.fit(train["nearest_amphi"])
    for df in [train, val, test]:
        df["nearest_amphi_enc"] = le.transform(df["nearest_amphi"])
    return train, val, test, le


def add_gps_quality_features(df, acc_median=None):
    """Add derived GPS quality metrics"""
    df = df.copy()
    if acc_median is None:
        acc_median = df["accuracy_mean"].median()

    df["log_accuracy"] = np.log1p(df["accuracy_mean"].fillna(acc_median))
    df["accuracy_bin"] = pd.cut(
        df["accuracy_mean"].fillna(999),
        bins=[0, 30, 80, 999],
        labels=[0, 1, 2]
    ).astype(int)
    df["high_accuracy_flag"] = (df["accuracy_mean"] < 30).astype(int)

    return df, acc_median


def add_seat_features(df):
    """Encode seat information (zone, block, has_seat flag)"""
    df = df.copy()
    df["has_seat"] = df["seat_row"].notna().astype(int)

    block_map = {"Left": 0, "Center": 1, "Right": 2}
    df["seat_block_enc"] = df["seat_block"].map(block_map).fillna(3).astype(int)

    df["seat_row_filled"] = df["seat_row"].fillna(-1)
    df["seat_column_filled"] = df["seat_column"].fillna(-1)

    # Composite zone ID: row*100 + column (or -1 for no seat)
    df["seat_zone_id"] = df.apply(
        lambda row: -1 if row["has_seat"] == 0 else int(row["seat_row_filled"]) * 100 + int(row["seat_column_filled"]),
        axis=1
    )
    return df


def add_temporal_features(df):
    """Add cyclic hour features (day_of_week intentionally excluded - collection bias)"""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["hour_of_day"] = ts.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df


def encode_labels(train, val, test):
    """Encode target labels to integers"""
    le = LabelEncoder()
    le.fit(train[TARGET_COL])

    for df in [train, val, test]:
        df["label_enc"] = le.transform(df[TARGET_COL])

    label_map = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    print("\nLabel mapping:", label_map)

    # Save label map
    (PROCESSED / "label_map.json").write_text(json.dumps(label_map, indent=2))
    return train, val, test, label_map


def drop_final_columns(df):
    """Drop columns replaced by engineered features"""
    drop_cols = [
        "timestamp",
        "seat_block",
        "gps_variance",  # constant 0 after preprocessing
        "target_label",  # replaced by label_enc
        "hour_of_day",   # only keep sin/cos
        "seat_row",
        "seat_column",
        "nearest_amphi",  # replaced by nearest_amphi_enc
    ]
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def save_feature_cols_json(dist_cols):
    """Save feature column names for interface to use"""
    feature_cols = {
        "distance_features": dist_cols + ["dist_nearest", "dist_2nd", "dist_gap"],
        "gps_quality_features": ["log_accuracy", "accuracy_bin", "high_accuracy_flag"],
        "temporal_features": ["hour_sin", "hour_cos"],
        "seat_features": ["has_seat", "seat_block_enc", "seat_row_filled", "seat_column_filled", "seat_zone_id"],
        "spatial_features": ["latitude_mean", "longitude_mean"],
        "other_features": ["accuracy_mean", "is_outside", "sample_count", "nearest_amphi_enc"],
        "target_col": "label_enc",
    }

    # Flatten for easy access
    feature_cols["all_features"] = (
        feature_cols["distance_features"] +
        feature_cols["gps_quality_features"] +
        feature_cols["temporal_features"] +
        feature_cols["seat_features"] +
        feature_cols["spatial_features"] +
        feature_cols["other_features"]
    )

    (PROCESSED / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    print(f"\nSaved feature_cols.json with {len(feature_cols['all_features'])} features")
    return feature_cols


def main():
    print("=" * 60)
    print("AmphiLocator — Feature Engineering (v3.2)")
    print("Matches notebook: 03_feature_engineering.ipynb")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading train/val/test splits...")
    train, val, test = load_splits()

    # 2. Drop metadata columns
    print("\n[2] Dropping metadata columns...")
    train = drop_initial_cols(train)
    val = drop_initial_cols(val)
    test = drop_initial_cols(test)

    # 3. Compute centroids from train only
    print("\n[3] Computing centroids (from train set only)...")
    centroids = compute_centroids(train)

    # 4. Add distance features
    print("\n[4] Adding distance-to-centroid features...")
    train, dist_cols = add_distance_features(train, centroids)
    val, _ = add_distance_features(val, centroids)
    test, _ = add_distance_features(test, centroids)
    print(f"    Added {len(dist_cols)} distance columns")

    # 5. Encode nearest amphitheatre
    print("\n[5] Encoding nearest_amphi...")
    train, val, test, le_nearest = encode_nearest_amphi(train, val, test)

    # 6. Add GPS quality features
    print("\n[6] Adding GPS quality features...")
    train, acc_median = add_gps_quality_features(train)
    val, _ = add_gps_quality_features(val, acc_median)
    test, _ = add_gps_quality_features(test, acc_median)

    # 7. Add seat features
    print("\n[7] Adding seat features...")
    train = add_seat_features(train)
    val = add_seat_features(val)
    test = add_seat_features(test)
    print(f"    Seat coverage (train): {train['has_seat'].mean():.1%}")

    # 8. Add temporal features
    print("\n[8] Adding temporal features (hour only)...")
    train = add_temporal_features(train)
    val = add_temporal_features(val)
    test = add_temporal_features(test)

    # 9. Encode target labels
    print("\n[9] Encoding target labels...")
    train, val, test, label_map = encode_labels(train, val, test)

    # 10. Drop final redundant columns
    print("\n[10] Dropping redundant columns...")
    train = drop_final_columns(train)
    val = drop_final_columns(val)
    test = drop_final_columns(test)

    # Convert is_outside to int if needed
    for df in [train, val, test]:
        if "is_outside" in df.columns and df["is_outside"].dtype == bool:
            df["is_outside"] = df["is_outside"].astype(int)

    # 11. Save feature columns JSON (for interface)
    print("\n[11] Saving feature_cols.json...")
    feature_cols = save_feature_cols_json(dist_cols)

    # 12. Save processed datasets (ONLY in subdirectories, not in root)
    print("\n[12] Saving ready-to-use datasets...")
    train.to_csv(PROCESSED / "train" / "train_ready.csv", index=False)
    val.to_csv(PROCESSED / "val" / "val_ready.csv", index=False)
    test.to_csv(PROCESSED / "test" / "test_ready.csv", index=False)

    # Save centroids
    (PROCESSED / "centroids.json").write_text(
        centroids.reset_index().to_json(orient="records", indent=2)
    )

    # Final summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {PROCESSED}")
    print(f"  - train/train_ready.csv     ({train.shape})")
    print(f"  - val/val_ready.csv         ({val.shape})")
    print(f"  - test/test_ready.csv       ({test.shape})")
    print(f"  - feature_cols.json         ({len(feature_cols['all_features'])} features)")
    print(f"  - centroids.json")
    print(f"  - label_map.json")
    print("\n✓ Ready for model training")


if __name__ == "__main__":
    main()