"""
Find Austin geographic accident hotspot zones using DBSCAN.

By default, this version is tuned for all-severity hotspot mapping
(Severity >= 1, eps=0.15 km, min_samples=5). You can still focus on severe-only
hotspots by passing a higher --min-severity value.

Usage:
    python hotspot_worst_zones.py
    python hotspot_worst_zones.py --min-severity 4 --eps-km 0.6 --min-samples 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


DATA_PATH = Path(__file__).resolve().parent / "austin_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "austin_hotspots.csv"
EARTH_RADIUS_KM = 6371.0088
AUSTIN_BOUNDS = {
    "lat_min": 29.9,
    "lat_max": 30.7,
    "lng_min": -98.1,
    "lng_max": -97.4,
}


def load_points(path: Path, min_severity: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Severity", "Start_Lat", "Start_Lng"], low_memory=False)
    df = df.dropna(subset=["Severity", "Start_Lat", "Start_Lng"])
    df = df[(df["Start_Lat"] != 0) & (df["Start_Lng"] != 0)]
    df = df[df["Severity"] >= min_severity].copy()
    df = df[
        (df["Start_Lat"] >= AUSTIN_BOUNDS["lat_min"])
        & (df["Start_Lat"] <= AUSTIN_BOUNDS["lat_max"])
        & (df["Start_Lng"] >= AUSTIN_BOUNDS["lng_min"])
        & (df["Start_Lng"] <= AUSTIN_BOUNDS["lng_max"])
    ].copy()
    return df


def cluster_points(df: pd.DataFrame, eps_km: float, min_samples: int) -> pd.DataFrame:
    if df.empty:
        return df.assign(cluster_id=pd.Series(dtype=int))

    coords_rad = np.radians(df[["Start_Lat", "Start_Lng"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = model.fit_predict(coords_rad)

    out = df.copy()
    out["cluster_id"] = labels
    return out


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    clustered = clustered[clustered["cluster_id"] != -1].copy()
    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "zone_rank",
                "cluster_id",
                "accident_count",
                "avg_severity",
                "max_severity",
                "center_lat",
                "center_lng",
                "lat_min",
                "lat_max",
                "lng_min",
                "lng_max",
            ]
        )

    grouped = clustered.groupby("cluster_id", as_index=False).agg(
        accident_count=("Severity", "size"),
        avg_severity=("Severity", "mean"),
        max_severity=("Severity", "max"),
        center_lat=("Start_Lat", "mean"),
        center_lng=("Start_Lng", "mean"),
        lat_min=("Start_Lat", "min"),
        lat_max=("Start_Lat", "max"),
        lng_min=("Start_Lng", "min"),
        lng_max=("Start_Lng", "max"),
    )

    grouped = grouped.sort_values(
        by=["accident_count", "avg_severity"], ascending=[False, False]
    ).reset_index(drop=True)
    grouped.insert(0, "zone_rank", np.arange(1, len(grouped) + 1))
    grouped["avg_severity"] = grouped["avg_severity"].round(3)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Austin accident hotspot zones with DBSCAN."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to austin_data.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path for ranked hotspot zones",
    )
    parser.add_argument(
        "--min-severity",
        type=int,
        default=1,
        help="Minimum severity included (default: 1 for all severities)",
    )
    parser.add_argument(
        "--eps-km",
        type=float,
        default=0.15,
        help="DBSCAN neighborhood radius in kilometers (default: 0.15)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum points to form a cluster (default: 5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many top zones to print (default: 15)",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")
    if args.min_severity < 1 or args.min_severity > 4:
        raise SystemExit("--min-severity must be between 1 and 4 for this dataset.")

    points_df = load_points(args.data, args.min_severity)
    print(f"Rows with Severity >= {args.min_severity}: {len(points_df):,}")

    clustered = cluster_points(points_df, args.eps_km, args.min_samples)
    noise_count = int((clustered["cluster_id"] == -1).sum()) if not clustered.empty else 0
    print(f"Noise points (not assigned to a zone): {noise_count:,}")

    summary = summarize_clusters(clustered)
    summary.to_csv(args.output, index=False)
    print(f"\nSaved ranked zones to: {args.output}")

    if summary.empty:
        print("No clusters found. Try increasing --eps-km or lowering --min-samples.")
        return

    print(f"\nTop {min(args.top_n, len(summary))} hotspot zones:")
    print(summary.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
