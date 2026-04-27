import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"
RESULTS      = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "target_label"
AMPHI_LABELS = [f"Amphi {i}" for i in range(1, 9)]

NEVER_SCALE = {
    "id", "year", "label_enc",
    "is_outside",
    "has_seat", "high_accuracy_flag",
    "accuracy_bin",
    "hour_of_day",
    "seat_block_enc", "seat_zone_id",
}


def load_splits():
    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED / f"{name}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        splits[name] = df
    shapes = {k: v.shape for k, v in splits.items()}
    print(f"Loaded splits: {shapes}")
    return splits["train"], splits["val"], splits["test"]


def compute_centroids(train: pd.DataFrame) -> pd.DataFrame:
    centroids = (
        train[train[TARGET_COL].isin(AMPHI_LABELS)]
        .groupby(TARGET_COL)[["latitude_mean", "longitude_mean"]]
        .mean()
        .rename(columns={"latitude_mean": "c_lat", "longitude_mean": "c_lon"})
    )
    print("Centroids (from train):")
    print(centroids.round(6).to_string())
    return centroids


def add_distance_features(df: pd.DataFrame, centroids: pd.DataFrame):
    df = df.copy()
    dist_cols = []

    for lbl, row in centroids.iterrows():
        col = f"dist_{lbl.replace(' ', '_')}"
        df[col] = np.sqrt(
            (df["latitude_mean"] - row["c_lat"]) ** 2 +
            (df["longitude_mean"] - row["c_lon"]) ** 2
        )
        dist_cols.append(col)

    dist_matrix = df[dist_cols]
    df["dist_nearest"] = dist_matrix.min(axis=1)
    df["dist_2nd"]     = dist_matrix.apply(lambda r: r.nsmallest(2).iloc[-1], axis=1)
    df["dist_gap"]     = df["dist_2nd"] - df["dist_nearest"]

    df["nearest_amphi"] = (
        dist_matrix.idxmin(axis=1)
        .str.replace("dist_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    return df, dist_cols


def add_gps_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    acc_median = df["accuracy_mean"].median()
    df["log_accuracy"] = np.log1p(df["accuracy_mean"].fillna(acc_median))

    df["accuracy_bin"] = pd.cut(
        df["accuracy_mean"].fillna(999),
        bins=[0, 30, 80, 999],
        labels=[0, 1, 2],
    ).astype(int)

    df["high_accuracy_flag"] = (df["accuracy_mean"] < 30).astype(int)

    return df


def add_seat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_seat"] = df["seat_row"].notna().astype(int)

    block_map = {"Left": 0, "Center": 1, "Right": 2}
    df["seat_block_enc"]     = df["seat_block"].map(block_map).fillna(3).astype(int)
    df["seat_row_filled"]    = df["seat_row"].fillna(-1)
    df["seat_column_filled"] = df["seat_column"].fillna(-1)

    df["seat_zone_id"] = np.where(
        df["has_seat"] == 0,
        -1,
        (df["seat_row_filled"] * 100 + df["seat_column_filled"]).astype(int),
    )

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]

    df["hour_of_day"] = ts.dt.hour
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    return df


def encode_labels(train, val, test):
    le = LabelEncoder()
    le.fit(train[TARGET_COL])

    for df in [train, val, test]:
        df["label_enc"] = le.transform(df[TARGET_COL])

    label_map = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    print("Label mapping:", label_map)
    (PROCESSED / "label_map.json").write_text(json.dumps(label_map, indent=2))
    return train, val, test, label_map


def build_feature_lists(dist_cols: list) -> dict:
    distance_features = dist_cols + ["dist_nearest", "dist_2nd", "dist_gap"]

    gps_quality_features = [
        "log_accuracy", "accuracy_bin", "high_accuracy_flag"
    ]

    temporal_features = [
        "hour_of_day", "hour_sin", "hour_cos"
    ]

    seat_features = [
        "has_seat", "seat_block_enc", "seat_row_filled", "seat_column_filled"
    ]

    raw_gps = [
        "latitude_mean", "longitude_mean", "accuracy_mean"
    ]

    all_features = (
        distance_features + gps_quality_features +
        temporal_features + seat_features + raw_gps
    )

    return {
        "distance_features":    distance_features,
        "gps_quality_features": gps_quality_features,
        "temporal_features":    temporal_features,
        "seat_features":        seat_features,
        "raw_gps":              raw_gps,
        "all_features":         all_features,
        "target_col":           "label_enc",
        "target_col_str":       TARGET_COL,
    }


def scale_features(train, val, test, all_features: list):
    scaling_features = [
        col for col in all_features
        if col in train.columns
        and col not in NEVER_SCALE
        and train[col].dtype in ("float64", "float32", "int64", "int32")
        and train[col].std() > 1e-6
    ]

    print(f"\nScaling {len(scaling_features)} features:")
    for f in scaling_features:
        print(f"  {f}")

    scaler = StandardScaler()
    scaler.fit(train[scaling_features])

    train_s, val_s, test_s = train.copy(), val.copy(), test.copy()
    for ds in [train_s, val_s, test_s]:
        ds[scaling_features] = scaler.transform(ds[scaling_features])

    mean_ok = abs(train_s[scaling_features].mean().mean()) < 0.01
    std_ok  = abs(train_s[scaling_features].std().mean() - 1) < 0.05
    print(f"\nScaling sanity — mean~0: {mean_ok}  |  std~1: {std_ok}")

    if not std_ok:
        print("WARNING: std check failed")

    return train_s, val_s, test_s, scaler, scaling_features


def save_outputs(train, val, test, train_s, val_s, test_s,
                 scaler, feature_config, centroids):

    train.to_csv(PROCESSED / "train_fe.csv",  index=False)
    val.to_csv(PROCESSED   / "val_fe.csv",    index=False)
    test.to_csv(PROCESSED  / "test_fe.csv",   index=False)

    train_s.to_csv(PROCESSED / "train_fe_scaled.csv", index=False)
    val_s.to_csv(PROCESSED   / "val_fe_scaled.csv",   index=False)
    test_s.to_csv(PROCESSED  / "test_fe_scaled.csv",  index=False)

    joblib.dump(scaler, PROCESSED / "scaler_fe.pkl")
    (PROCESSED / "feature_cols.json").write_text(json.dumps(feature_config, indent=2))
    (PROCESSED / "centroids.json").write_text(
        centroids.reset_index().to_json(orient="records", indent=2)
    )

    print("\nSaved to data/processed/:")
    for f in ["train_fe.csv", "val_fe.csv", "test_fe.csv",
              "train_fe_scaled.csv", "val_fe_scaled.csv", "test_fe_scaled.csv",
              "scaler_fe.pkl", "feature_cols.json", "label_map.json", "centroids.json"]:
        print(f"  {f}")

    print(f"\nFinal train shape (unscaled): {train.shape}")
    print(f"Final train shape (scaled):   {train_s.shape}")


def main():
    print("=" * 60)
    print("AmphiLocator — Feature Engineering")
    print("=" * 60)

    train, val, test = load_splits()

    centroids = compute_centroids(train)

    print("\n[3] Distance features...")
    train, dist_cols = add_distance_features(train, centroids)
    val,   _         = add_distance_features(val,   centroids)
    test,  _         = add_distance_features(test,  centroids)
    print(f"    {len(dist_cols)} centroid distances + 3 summary features added")

    print("\n[4] GPS quality features...")
    train = add_gps_quality_features(train)
    val   = add_gps_quality_features(val)
    test  = add_gps_quality_features(test)

    print("\n[5] Seat features...")
    train = add_seat_features(train)
    val   = add_seat_features(val)
    test  = add_seat_features(test)
    print(f"    Seat coverage (train): {train['has_seat'].mean():.1%}")

    print("\n[6] Temporal features...")
    train = add_temporal_features(train)
    val   = add_temporal_features(val)
    test  = add_temporal_features(test)

    print("\n[7] Label encoding...")
    train, val, test, _ = encode_labels(train, val, test)

    feature_config = build_feature_lists(dist_cols)
    all_features   = feature_config["all_features"]
    print(f"\n[8] Feature set: {len(all_features)} total features")
    for group, cols in feature_config.items():
        if isinstance(cols, list) and group != "all_features":
            print(f"    {group:25s}: {len(cols)}")

    print("\n[9] Scaling...")
    train_s, val_s, test_s, scaler, scaling_features = scale_features(
        train, val, test, all_features
    )
    feature_config["scaling_features"] = scaling_features

    print("\n[10] Saving outputs...")
    save_outputs(train, val, test, train_s, val_s, test_s,
                 scaler, feature_config, centroids)

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()