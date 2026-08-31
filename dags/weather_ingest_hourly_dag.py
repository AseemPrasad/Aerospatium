"""Airflow DAG: Hourly ERA5-Land weather raster ingestion.

Downloads the latest ERA5-Land hourly forecasts, processes them into
the H3 topology (bilinear interpolation of U10/V10/T2m/RH to hex
centroids), and writes the dynamic weather tensor.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "aerospatium",
    "depends_on_past": False,
    "start_date": datetime(2026, 8, 15),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

PROCESSED_DIR = os.environ.get("AEROSPATIUM_PROCESSED_DIR", "/app/data/processed")
RAW_DIR = os.environ.get("AEROSPATIUM_RAW_DIR", "/app/data/raw")

# Canonical simulation time (pinned to scenario day)
SIMULATION_DATE = "2023-08-15"


def ingest_era5_weather(**context):
    """Ingests ERA5 hourly weather and interpolates to H3 centroids."""
    import numpy as np
    import pandas as pd
    import h3
    import xarray as xr

    weather_path = os.path.join(RAW_DIR, "weather", "era5_hourly_aoi.nc")
    static_path = os.path.join(PROCESSED_DIR, "h3_static_features.parquet")
    output_path = os.path.join(PROCESSED_DIR, "h3_weather_t12.npz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"ERA5 file not found: {weather_path}")

    if not os.path.exists(static_path):
        raise FileNotFoundError(f"Static features not found: {static_path}")

    # Load H3 centroids
    df_static = pd.read_parquet(static_path)
    lats = df_static["centroid_lat"].values
    lons = df_static["centroid_lon"].values

    # Load ERA5 data (pre-sorted by latitude)
    ds = xr.open_dataset(weather_path).sortby("latitude")

    T = 12
    points_lat = xr.DataArray(lats, dims="points")
    points_lon = xr.DataArray(lons, dims="points")

    u10 = ds["u10"].isel(time=slice(0, T)).interp(
        latitude=points_lat, longitude=points_lon, method="linear"
    ).values
    v10 = ds["v10"].isel(time=slice(0, T)).interp(
        latitude=points_lat, longitude=points_lon, method="linear"
    ).values
    t2m = ds["t2m"].isel(time=slice(0, T)).interp(
        latitude=points_lat, longitude=points_lon, method="linear"
    ).values
    d2m = ds["d2m"].isel(time=slice(0, T)).interp(
        latitude=points_lat, longitude=points_lon, method="linear"
    ).values

    # Convert to derived features
    tc = t2m - 273.15
    td = d2m - 273.15
    rh = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * tc / (243.04 + tc))

    temp_norm = (tc - 25.0) / 15.0
    rh_norm = (rh - 30.0) / 30.0

    x_dyn = np.stack([u10, v10, temp_norm, rh_norm], axis=-1).astype(np.float32)

    np.savez_compressed(
        output_path,
        X_dyn=x_dyn,
        h3_id=df_static["h3_id"].values,
        timestamp=SIMULATION_DATE,
    )

    context["ti"].xcom_push("h3_count", len(df_static))
    context["ti"].xcom_push("time_steps", T)
    print(f"Ingested ERA5 weather: {x_dyn.shape} → {output_path}")


with DAG(
    dag_id="weather_ingest_hourly",
    default_args=DEFAULT_ARGS,
    description="Hourly ERA5-Land weather ingestion & H3 interpolation",
    schedule_interval="0 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["aerospatium", "weather", "data-ingestion"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_era5_weather",
        python_callable=ingest_era5_weather,
    )

    ingest_task
