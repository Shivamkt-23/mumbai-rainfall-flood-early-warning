#!/usr/bin/env python3
"""
download_imerg_2010_2025_daily_fast.py

NASA GPM IMERG Final Daily V07 -> Mumbai historical feature CSV.

LOCKED DATA PIPELINE
--------------------
ERA5 2010-2025 (already local)
        +
NASA GPM IMERG Final Daily V07
        |
        +-- download small batch concurrently
        +-- extract Mumbai 0.1-degree pixel
        +-- delete batch
        |
        v
data/processed/imerg_2010_2025_daily.csv

This version uses the SAME WORKING NASA access path that the
single-file preflight already confirmed:

    earthaccess.login()
    earthaccess.search_data()
    earthaccess.download()

It does NOT use:
    - OPeNDAP
    - S3
    - Harmony
    - Giovanni

Performance design:
    - one CMR search per year
    - batches of files
    - concurrent NASA downloads inside each batch
    - only one batch is kept on disk at a time
    - Mumbai pixel extraction is local and fast
    - checkpoint after every batch and every year
    - resume from an existing CSV

Default:
    DOWNLOAD_THREADS = 8
    BATCH_SIZE = 16

At ~30 MB/file, a 16-file batch is ~480 MB maximum temporary disk use.

Requirements:
    python -m pip install -U earthaccess h5py numpy pandas
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import sys
import time
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd


# ================================================================
# CONFIG
# ================================================================

MUMBAI_LAT = 19.05
MUMBAI_LON = 73.05

START_DATE = "2010-01-01"
END_DATE_EXCLUSIVE = "2025-10-01"

SHORT_NAME = "GPM_3IMERGDF"
VERSION = "07"
CONCEPT_ID = "C2723754864-GES_DISC"

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "imerg_2010_2025_daily.csv"

TEMP_ROOT = BASE_DIR / "data" / "tmp_imerg_daily_fast"

# Performance controls.
DOWNLOAD_THREADS = 8
BATCH_SIZE = 16

# Retry an entire batch if earthaccess.download raises.
BATCH_RETRIES = 3
RETRY_SLEEP_SECONDS = 8

# Optional fields available in the Daily V07 product.
OPTIONAL_FIELDS = [
    "precipitation_cnt",
    "precipitation_cnt_cond",
    "randomError",
    "randomError_cnt",
    "probabilityLiquidPrecipitation",
]


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("IMERG_FAST")


# ================================================================
# GENERAL HELPERS
# ================================================================

def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(np.asarray(arr, dtype=float) - value).argmin())


def clean_scalar(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not math.isfinite(value):
        return float("nan")

    # IMERG fill values are strongly negative.
    if value <= -9000:
        return float("nan")

    return value


def find_dataset(group: h5py.Group, name: str):
    """Find a dataset by its basename anywhere in the HDF5 tree."""
    result = None

    def visitor(path, obj):
        nonlocal result

        if result is not None:
            return

        if isinstance(obj, h5py.Dataset):
            if path.rsplit("/", 1)[-1] == name:
                result = obj

    group.visititems(visitor)
    return result


def infer_axes(
    precipitation: h5py.Dataset,
    lat_len: int,
    lon_len: int,
) -> tuple[int, int, int]:
    """
    Infer (time_axis, lon_axis, lat_axis).

    IMERG Daily normally uses (time, lon, lat), but we determine the
    axes from their lengths rather than relying on a hard-coded order.
    """
    shape = precipitation.shape

    if len(shape) != 3:
        raise RuntimeError(
            f"Unexpected precipitation shape: {shape}; expected 3 dimensions."
        )

    lon_candidates = [
        i for i, n in enumerate(shape) if n == lon_len
    ]
    lat_candidates = [
        i for i, n in enumerate(shape) if n == lat_len
    ]

    if not lon_candidates or not lat_candidates:
        raise RuntimeError(
            "Could not identify latitude/longitude axes. "
            f"shape={shape}, lon_len={lon_len}, lat_len={lat_len}"
        )

    # For IMERG, longitude and latitude dimensions are different sizes,
    # so their axes should be uniquely identifiable.
    lon_axis = lon_candidates[0]
    lat_axis = lat_candidates[0]

    if lon_axis == lat_axis:
        raise RuntimeError(f"Could not distinguish lon/lat axes: {shape}")

    time_candidates = [
        i for i in range(3)
        if i not in (lon_axis, lat_axis)
    ]

    if len(time_candidates) != 1:
        raise RuntimeError(f"Could not identify time axis: {shape}")

    time_axis = time_candidates[0]

    return time_axis, lon_axis, lat_axis


# ================================================================
# HDF5 EXTRACTION
# ================================================================

def read_daily_granule(
    file_path: Path,
    cached_indices: dict | None = None,
) -> dict:
    """
    Read one Daily V07 .nc4 file and extract the Mumbai pixel only.

    Returns a dictionary containing the daily precipitation plus the
    optional quality variables when present.
    """
    with h5py.File(file_path, "r") as f:

        lat_ds = find_dataset(f, "lat")
        lon_ds = find_dataset(f, "lon")
        precip_ds = find_dataset(f, "precipitation")

        if lat_ds is None:
            raise RuntimeError("Dataset 'lat' not found.")
        if lon_ds is None:
            raise RuntimeError("Dataset 'lon' not found.")
        if precip_ds is None:
            raise RuntimeError("Dataset 'precipitation' not found.")

        lat = np.asarray(lat_ds[:], dtype=float).reshape(-1)
        lon = np.asarray(lon_ds[:], dtype=float).reshape(-1)

        if cached_indices is None:
            lat_idx = nearest_index(lat, MUMBAI_LAT)
            lon_idx = nearest_index(lon, MUMBAI_LON)

            time_axis, lon_axis, lat_axis = infer_axes(
                precip_ds,
                lat_len=len(lat),
                lon_len=len(lon),
            )

        else:
            lat_idx = int(cached_indices["lat_idx"])
            lon_idx = int(cached_indices["lon_idx"])
            time_axis = int(cached_indices["time_axis"])
            lon_axis = int(cached_indices["lon_axis"])
            lat_axis = int(cached_indices["lat_axis"])

        # Read one time slice + one grid cell.
        idx = [slice(None)] * 3
        idx[time_axis] = 0
        idx[lon_axis] = lon_idx
        idx[lat_axis] = lat_idx

        raw = precip_ds[tuple(idx)]
        precip = clean_scalar(np.asarray(raw).reshape(-1)[0])

        values = {
            "precip_mm_day": precip,
            "lat": float(lat[lat_idx]),
            "lon": float(lon[lon_idx]),
        }

        # Optional quality/uncertainty fields.
        for field in OPTIONAL_FIELDS:
            ds = find_dataset(f, field)

            if ds is None:
                continue

            try:
                raw = ds[tuple(idx)]
                values[field] = clean_scalar(
                    np.asarray(raw).reshape(-1)[0]
                )
            except Exception as exc:
                log.warning(
                    "%s: optional field %s failed: %s",
                    file_path.name,
                    field,
                    exc,
                )

        # Cache grid indices/axis information for future files.
        values["_lat_idx"] = lat_idx
        values["_lon_idx"] = lon_idx
        values["_time_axis"] = time_axis
        values["_lon_axis"] = lon_axis
        values["_lat_axis"] = lat_axis

        return values


# ================================================================
# FILE / DATE HELPERS
# ================================================================

DATE_RE = re.compile(r"3IMERG\.(\d{8})-")


def date_from_filename(path: Path) -> str:
    match = DATE_RE.search(path.name)

    if not match:
        raise RuntimeError(
            f"Could not extract YYYYMMDD from filename: {path.name}"
        )

    yyyymmdd = match.group(1)

    return (
        f"{yyyymmdd[0:4]}-"
        f"{yyyymmdd[4:6]}-"
        f"{yyyymmdd[6:8]}"
    )


def valid_record(record: dict) -> bool:
    try:
        value = float(record.get("precip_mm_day", np.nan))
    except (TypeError, ValueError):
        return False

    return math.isfinite(value)


def load_records() -> dict[str, dict]:
    if not OUTPUT_CSV.exists():
        return {}

    df = pd.read_csv(
        OUTPUT_CSV,
        dtype={"date": str},
    )

    if "date" not in df.columns:
        raise RuntimeError(
            f"Existing file has no 'date' column: {OUTPUT_CSV}"
        )

    records: dict[str, dict] = {}

    for _, row in df.iterrows():
        date = str(row["date"])

        values = {}

        for col in df.columns:
            if col == "date":
                continue

            val = row[col]

            if pd.isna(val):
                values[col] = np.nan
            else:
                try:
                    values[col] = float(val)
                except (TypeError, ValueError):
                    values[col] = val

        records[date] = values

    return records


def save_records(records: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for date, values in sorted(records.items()):
        row = {"date": date}
        row.update(values)
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("date")

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )


def delete_files(paths) -> None:
    for path in paths:
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except Exception as exc:
            log.warning(
                "Could not delete temporary file %s: %s",
                path,
                exc,
            )


def cleanup_directory(path: Path) -> None:
    if not path.exists():
        return

    try:
        shutil.rmtree(path)
    except Exception as exc:
        log.warning(
            "Could not remove temporary directory %s: %s",
            path,
            exc,
        )


# ================================================================
# NASA SEARCH / DOWNLOAD
# ================================================================

def search_year(year: int):
    """
    One CMR call for a complete year.
    """
    start = f"{year}-01-01"

    if year == 2025:
        end = "2025-09-30"
    else:
        end = f"{year}-12-31"

    return earthaccess.search_data(
        concept_id=CONCEPT_ID,
        version=VERSION,
        temporal=(start, end),
        cloud_hosted=False,
    )


def download_batch(
    granules,
    batch_dir: Path,
) -> list[Path]:
    """
    Download one small batch concurrently.

    earthaccess.download handles the concurrent transfer using the
    threads parameter. We keep BATCH_SIZE small so temporary disk use
    remains bounded.
    """
    batch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_exc = None

    for attempt in range(1, BATCH_RETRIES + 1):

        try:
            result = earthaccess.download(
                granules,
                local_path=str(batch_dir),
                threads=DOWNLOAD_THREADS,
                show_progress=True,
            )

            paths: list[Path] = []

            if isinstance(result, (list, tuple)):
                for item in result:
                    paths.append(Path(item))
            else:
                paths.append(Path(result))

            # Some versions can return relative paths.
            normalized = []

            for path in paths:
                if path.exists():
                    normalized.append(path)
                    continue

                candidate = batch_dir / path.name

                if candidate.exists():
                    normalized.append(candidate)
                    continue

                raise FileNotFoundError(
                    f"Downloaded file missing: {path}"
                )

            return normalized

        except Exception as exc:

            last_exc = exc

            log.warning(
                "Batch download attempt %d/%d failed: %s",
                attempt,
                BATCH_RETRIES,
                exc,
            )

            if attempt < BATCH_RETRIES:
                cleanup_directory(batch_dir)
                batch_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                time.sleep(
                    RETRY_SLEEP_SECONDS * attempt
                )

    raise RuntimeError(
        f"Batch download failed after {BATCH_RETRIES} attempts: "
        f"{last_exc}"
    )


# ================================================================
# PREFLIGHT
# ================================================================

def preflight() -> dict:
    """
    Verify the exact working NASA download path using one Daily V07 file.
    """
    banner("ONE-FILE PREFLIGHT")

    print("Searching for 2010-01-01...")

    results = earthaccess.search_data(
        concept_id=CONCEPT_ID,
        version=VERSION,
        temporal=("2010-01-01", "2010-01-01"),
        cloud_hosted=False,
        count=10,
    )

    if not results:
        raise RuntimeError(
            "No GPM_3IMERGDF V07 granule returned for 2010-01-01."
        )

    print("Granule found.")

    preflight_dir = TEMP_ROOT / "_preflight"
    cleanup_directory(preflight_dir)

    files = []

    try:
        files = download_batch(
            [results[0]],
            preflight_dir,
        )

        if len(files) != 1:
            raise RuntimeError(
                f"Expected one preflight file, got {len(files)}"
            )

        path = files[0]

        print(f"Downloaded: {path}")
        print(
            f"Size     : "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

        values = read_daily_granule(path)

        print()
        print("PRE-FLIGHT DOWNLOAD SUCCESS")
        print(
            f"Mumbai grid : "
            f"{values['lat']:.5f}, {values['lon']:.5f}"
        )
        print(
            f"Precip      : "
            f"{values['precip_mm_day']:.4f} mm/day"
        )

        for field in OPTIONAL_FIELDS:
            if field in values:
                print(
                    f"{field:28s}: {values[field]}"
                )

        return {
            "lat_idx": values["_lat_idx"],
            "lon_idx": values["_lon_idx"],
            "time_axis": values["_time_axis"],
            "lon_axis": values["_lon_axis"],
            "lat_axis": values["_lat_axis"],
        }

    finally:
        cleanup_directory(preflight_dir)


# ================================================================
# YEAR PROCESSING
# ================================================================

def process_year(
    year: int,
    records: dict[str, dict],
    cached_indices: dict,
) -> tuple[int, int]:

    banner(f"YEAR {year}")

    try:
        granules = search_year(year)
    except Exception as exc:
        log.error(
            "CMR search failed for %d: %s",
            year,
            exc,
        )
        return 0, 1

    print(f"Granules found: {len(granules)}")

    # Sort in chronological order when possible by granule metadata.
    # Filename sorting is not available until after download, so we
    # simply preserve the returned CMR order for batching.
    todo = []

    for granule in granules:

        # We cannot rely on granule object formatting for the date.
        # Search result order is chronological enough for the product,
        # and existing CSV records are used for resume.
        todo.append(granule)

    success_count = 0
    fail_count = 0

    total_batches = math.ceil(
        len(todo) / BATCH_SIZE
    )

    for batch_number, start in enumerate(
        range(0, len(todo), BATCH_SIZE),
        start=1,
    ):

        batch_granules = todo[
            start:start + BATCH_SIZE
        ]

        batch_dir = (
            TEMP_ROOT
            / str(year)
            / f"batch_{batch_number:04d}"
        )

        cleanup_directory(batch_dir)

        print()
        print(
            f"Batch {batch_number}/{total_batches} "
            f"({len(batch_granules)} files)"
        )

        try:
            downloaded = download_batch(
                batch_granules,
                batch_dir,
            )

            # Extraction is deliberately sequential; HDF5 reads of a
            # single cell are tiny compared with network transfer time.
            for path in downloaded:

                try:
                    date = date_from_filename(path)

                    if (
                        date < START_DATE
                        or date >= END_DATE_EXCLUSIVE
                    ):
                        continue

                    # Do not overwrite a valid prior record.
                    if (
                        date in records
                        and valid_record(records[date])
                    ):
                        continue

                    values = read_daily_granule(
                        path,
                        cached_indices=cached_indices,
                    )

                    persisted = {
                        key: value
                        for key, value in values.items()
                        if not key.startswith("_")
                    }

                    records[date] = persisted
                    success_count += 1

                    log.info(
                        "%s -> %.4f mm/day",
                        date,
                        float(
                            persisted["precip_mm_day"]
                        ),
                    )

                except Exception as exc:
                    fail_count += 1

                    try:
                        date_for_error = date_from_filename(path)
                    except Exception:
                        date_for_error = path.name

                    log.error(
                        "Extraction failed for %s: %s",
                        date_for_error,
                        exc,
                    )

        except Exception as exc:

            # A batch failed. Preserve no fake data. The same batch will
            # be attempted again on the next script run because no valid
            # records were written for its files.
            fail_count += len(batch_granules)

            log.error(
                "Batch %d failed completely: %s",
                batch_number,
                exc,
            )

        finally:
            # Delete every downloaded NASA file immediately.
            cleanup_directory(batch_dir)

        # Checkpoint after every batch.
        save_records(records)

        log.info(
            "Checkpoint: %d total dates saved.",
            len(records),
        )

    # Remove empty year folder if possible.
    cleanup_directory(
        TEMP_ROOT / str(year)
    )

    print()
    print(
        f"YEAR {year} COMPLETE | "
        f"new/updated: {success_count} | "
        f"failed: {fail_count} | "
        f"total CSV: {len(records)}"
    )

    return success_count, fail_count


# ================================================================
# FINAL VALIDATION
# ================================================================

def final_validation(records: dict[str, dict]) -> None:

    banner("FINAL VALIDATION")

    save_records(records)

    df = pd.read_csv(
        OUTPUT_CSV,
        parse_dates=["date"],
    )

    df = df.sort_values("date").reset_index(drop=True)

    expected_dates = pd.date_range(
        START_DATE,
        pd.Timestamp(END_DATE_EXCLUSIVE)
        - pd.Timedelta(days=1),
        freq="D",
    )

    expected = set(
        expected_dates.strftime("%Y-%m-%d")
    )

    actual = set(
        df["date"].dt.strftime("%Y-%m-%d")
    )

    missing_dates = sorted(
        expected - actual
    )

    nan_count = int(
        df["precip_mm_day"].isna().sum()
    )

    duplicate_count = int(
        df["date"].duplicated().sum()
    )

    print(f"CSV rows            : {len(df)}")
    print(f"Expected days       : {len(expected_dates)}")
    print(f"Missing dates       : {len(missing_dates)}")
    print(f"NaN precipitation   : {nan_count}")
    print(f"Duplicate dates     : {duplicate_count}")
    print(f"Output              : {OUTPUT_CSV}")

    if missing_dates:
        print()
        print("First missing dates:")
        for date in missing_dates[:20]:
            print("   ", date)

    if not df.empty:
        print()
        print("Precipitation statistics:")
        print(
            df["precip_mm_day"].describe().to_string()
        )

    if (
        not missing_dates
        and nan_count == 0
        and duplicate_count == 0
    ):
        print()
        print(
            "SUCCESS: Complete IMERG Daily V07 "
            "2010-01-01 through 2025-09-30."
        )
    else:
        print()
        print(
            "Dataset is incomplete. Re-run the same script "
            "to resume missing dates."
        )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    banner("NASA GPM IMERG FINAL DAILY V07 — FAST DOWNLOADER")

    print(f"Mumbai latitude : {MUMBAI_LAT}")
    print(f"Mumbai longitude: {MUMBAI_LON}")
    print()
    print(
        f"Period          : "
        f"{START_DATE} -> {END_DATE_EXCLUSIVE} (exclusive)"
    )
    print(
        f"Threads         : {DOWNLOAD_THREADS}"
    )
    print(
        f"Batch size      : {BATCH_SIZE}"
    )
    print(
        f"Output          : {OUTPUT_CSV}"
    )
    print(
        f"Temp directory  : {TEMP_ROOT}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clean abandoned temporary files from an interrupted run.
    banner("CLEANING OLD TEMPORARY DOWNLOADS")
    cleanup_directory(TEMP_ROOT)
    TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    print("Temporary directory ready.")

    # ------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------

    banner("1. NASA EARTHDATA AUTHENTICATION")

    try:
        earthaccess.login()
    except Exception as exc:
        sys.exit(
            "Earthdata login failed:\n"
            f"{type(exc).__name__}: {exc}"
        )

    print("Earthdata authentication successful.")

    # ------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------

    try:
        cached_indices = preflight()
    except Exception as exc:
        banner("PREFLIGHT FAILED")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "The script stopped before the historical loop."
        )
        cleanup_directory(TEMP_ROOT)
        sys.exit(1)

    # ------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------

    banner("2. RESUME CHECK")

    records = load_records()

    print(
        f"Existing valid/partial CSV records: {len(records)}"
    )

    # ------------------------------------------------------------
    # Historical processing
    # ------------------------------------------------------------

    total_success = 0
    total_failures = 0

    for year in range(2010, 2026):

        success, failures = process_year(
            year,
            records,
            cached_indices,
        )

        total_success += success
        total_failures += failures

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    final_validation(records)

    cleanup_directory(TEMP_ROOT)

    banner("BUILD COMPLETE")

    print(
        f"New/updated records : {total_success}"
    )
    print(
        f"Failed file attempts: {total_failures}"
    )
    print(
        f"Final CSV            : {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
