"""Airflow DAG: Hourly FIRMS thermal hotspot ingestion & H3 buffer.

Pulls NASA FIRMS VIIRS/MODIS active-fire data, filters to the AOI
bounding box, maps detections to H3 res-8 cells, and writes a
parquet table for downstream hazard model seeding.
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

AOI_BBOX = {
    "min_lat": 39.6,
    "max_lat": 40.0,
    "min_lon": -121.8,
    "max_lon": -121.3,
}

PROCESSED_DIR = os.environ.get("AEROSPATIUM_PROCESSED_DIR", "/app/data/processed")


def ingest_firms_hotspots(**context):
    """Download and ingest the latest FIRMS hotspots into H3 cells."""
    import json
    import urllib.request

    import h3
    import pandas as pd

    source_url = (
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/viirs"
        "/SNPP-HIMARS/catalog/today.csv"
    )

    output_path = os.path.join(PROCESSED_DIR, "firms_hotspots_latest.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download
    try:
        with urllib.request.urlopen(source_url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception:
        # Fallback: use existing local copy
        raw_path = "/app/data/raw/firms/viirs_snpp_california.csv"
        if os.path.exists(raw_path):
            with open(raw_path, "r") as f:
                raw = f.read()
        else:
            raise RuntimeError("Could not download or locate FIRMS data")

    df = pd.read_csv(pd.io.common.StringIO(raw))

    # Filter to AOI
    df_aoi = df[
        (df["longitude"] >= AOI_BBOX["min_lon"])
        & (df["longitude"] <= AOI_BBOX["max_lon"])
        & (df["latitude"] >= AOI_BBOX["min_lat"])
        & (df["latitude"] <= AOI_BBOX["max_lat"])
    ].copy()

    if len(df_aoi) > 0:
        df_aoi["h3_id"] = df_aoi.apply(
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], 8), axis=1
        )
        df_aoi["detection_utc"] = pd.to_datetime(df_aoi["acq_date"] + " " + df_aoi["acq_time"].astype(str), errors="coerce")

    df_aoi.to_parquet(output_path, compression="snappy")
    context["ti"].xcom_push("hotspot_count", len(df_aoi))
    print(f"Ingested {len(df_aoi)} AOI hotspots → {output_path}")


with DAG(
    dag_id="fire_ingest_hourly",
    default_args=DEFAULT_ARGS,
    description="Hourly NASA FIRMS thermal hotspot ingestion & H3 buffer",
    schedule_interval="*/30 * * * *",  # every 30 minutes
    catchup=False,
    max_active_runs=1,
    tags=["aerospatium", "fire", "data-ingestion"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_firms_hotspots",
        python_callable=ingest_firms_hotspots,
    )

    ingest_task
