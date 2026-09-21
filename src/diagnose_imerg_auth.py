#!/usr/bin/env python3
"""
download_imerg_2010_2025_daily.py

NASA GPM IMERG Final Daily V07
--------------------------------
Period:
    2010-01-01 through 2025-09-30

Workflow:
    Earthdata login
      -> CMR search (one search per year)
      -> download ONE granule at a time
      -> extract only Mumbai grid-cell values
      -> append/checkpoint CSV
      -> delete downloaded granule
      -> continue

This intentionally DOES NOT use OPeNDAP.
It also does NOT keep ~12 MB/day files on disk permanently.

Output:
    data/processed/imerg_2010_2025_daily.csv

Temporary download directory:
    data/tmp_imerg_daily/

The script resumes from an existing CSV.

Required:
    python -m pip install -U earthaccess h5py numpy pandas

Important:
    NASA GES DISC access must be authorized for the Earthdata account.
    The script performs a ONE-FILE connectivity test before starting
    the full historical run.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

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

TEMP_DIR = BASE_DIR / "data" / "tmp_imerg_daily"

CHECKPOINT_EVERY = 30
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 10

# Optional extra satellite fields to extract when present.
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

log = logging.getLogger("IMERG")


# ================================================================
# HELPERS
# ================================================================

def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(np.asarray(arr, dtype=float) - value).argmin())


def clean_scalar(value) -> float:
    """Convert fill/masked values to NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not math.isfinite(v):
        return float("nan")

    # IMERG fill values are very negative.
    if v <= -9000:
        return float("nan")

    return v


def find_dataset(group: h5py.Group, name: str):
    """
    Find a dataset anywhere under the HDF5 group by basename.
    Returns the first matching dataset, or None.
    """
    found = None

    def visitor(path, obj):
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset):
            if path.rsplit("/", 1)[-1] == name:
                found = obj

    group.visititems(visitor)
    return found


def find_all_dataset_names(group: h5py.Group) -> list[str]:
    names: list[str] = []

    def visitor(path, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(path)

    group.visititems(visitor)
    return names


def infer_axes(
    precipitation: h5py.Dataset,
    lat_len: int,
    lon_len: int,
) -> tuple[int, int, int]:
    """
    Infer (time_axis, lon_axis, lat_axis) from array shape.

    Daily IMERG normally has shape (time, lon, lat), but we infer
    the axes using the coordinate lengths rather than assuming the
    order blindly.
    """
    shape = precipitation.shape

    if len(shape) != 3:
        raise RuntimeError(
            f"Expected 3-D precipitation array, got shape={shape}"
        )

    lon_axes = [i for i, n in enumerate(shape) if n == lon_len]
    lat_axes = [i for i, n in enumerate(shape) if n == lat_len]

    if not lon_axes or not lat_axes:
        raise RuntimeError(
            "Could not infer lon/lat axes. "
            f"precipitation shape={shape}, lon_len={lon_len}, lat_len={lat_len}"
        )

    lon_axis = lon_axes[0]
    lat_axis = lat_axes[0]

    if lon_axis == lat_axis:
        raise RuntimeError(
            f"Could not distinguish lon and lat axes for shape={shape}"
        )

    time_axes = [i for i in range(3) if i not in (lon_axis, lat_axis)]

    if len(time_axes) != 1:
        raise RuntimeError(
            f"Could not infer time axis for precipitation shape={shape}"
        )

    return time_axes[0], lon_axis, lat_axis


def read_daily_granule(
    file_path: Path,
    cached_indices: dict | None = None,
) -> dict:
    """
    Open one downloaded IMERG Daily V07 file and extract the Mumbai
    grid-cell values.

    Returns:
        {
            "lat": matched_lat,
            "lon": matched_lon,
            "precip_mm_day": ...,
            ...
        }
    """

    with h5py.File(file_path, "r") as f:

        lat_ds = find_dataset(f, "lat")
        lon_ds = find_dataset(f, "lon")
        precip_ds = find_dataset(f, "precipitation")

        if lat_ds is None:
            raise RuntimeError("Could not find dataset 'lat'")
        if lon_ds is None:
            raise RuntimeError("Could not find dataset 'lon'")
        if precip_ds is None:
            raise RuntimeError("Could not find dataset 'precipitation'")

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

        # Build an index that reads only one grid cell and the single
        # daily time slice instead of loading the full global array.
        index = [slice(None)] * 3
        index[time_axis] = 0
        index[lon_axis] = lon_idx
        index[lat_axis] = lat_idx

        precip_raw = precip_ds[tuple(index)]
        precip_val = clean_scalar(np.asarray(precip_raw).reshape(-1)[0])

        result = {
            "lat": float(lat[lat_idx]),
            "lon": float(lon[lon_idx]),
            "precip_mm_day": precip_val,
        }

        # Optional daily quality variables.
        for field in OPTIONAL_FIELDS:
            ds = find_dataset(f, field)
            if ds is None:
                continue

            try:
                raw = ds[tuple(index)]
                result[field] = clean_scalar(
                    np.asarray(raw).reshape(-1)[0]
                )
            except Exception as exc:
                log.warning(
                    "Could not extract optional field %s from %s: %s",
                    field,
                    file_path.name,
                    exc,
                )

        # Add useful diagnostics when this is the first successful file.
        result["_lat_idx"] = lat_idx
        result["_lon_idx"] = lon_idx
        result["_time_axis"] = time_axis
        result["_lon_axis"] = lon_axis
        result["_lat_axis"] = lat_axis

        return result


def granule_date(granule) -> str:
    """
    Derive YYYY-MM-DD from the CMR granule temporal metadata.
    Falls back to parsing the filename.
    """
    try:
        meta = granule["umm"]

        # UMM temporal ranges commonly appear here.
        temporal = meta.get("TemporalExtent", {})
        ranges = temporal.get("RangeDateTime", [])
        if ranges:
            start = ranges[0].get("BeginningDateTime")
            if start:
                return str(start)[:10]

        # Some responses may use a different structure.
        for key in ("BeginningDateTime", "StartDateTime"):
            value = meta.get(key)
            if value:
                return str(value)[:10]

    except Exception:
        pass

    # Filename fallback.
    text = str(granule)
    for year in range(2010, 2026):
        marker = str(year)
        pos = text.find(marker)
        if pos >= 0 and pos + 10 <= len(text):
            candidate = text[pos:pos + 10]
            if candidate[4] == "-" and candidate[7] == "-":
                return candidate

    raise RuntimeError("Could not determine granule date.")


def extract_downloaded_path(download_result) -> Path:
    """
    earthaccess.download() normally returns Path-like results.
    Handle a few forms defensively.
    """
    if isinstance(download_result, (str, os.PathLike)):
        return Path(download_result)

    if isinstance(download_result, (list, tuple)) and download_result:
        first = download_result[0]

        if isinstance(first, (str, os.PathLike)):
            return Path(first)

        # Some wrappers may return objects with a path attribute.
        for attr in ("path", "filename", "name"):
            value = getattr(first, attr, None)
            if value:
                return Path(value)

    raise RuntimeError(
        f"Could not determine downloaded file path from result: "
        f"{download_result!r}"
    )


def download_one(
    granule,
    target_dir: Path,
) -> Path:
    """
    Download exactly one granule.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    result = earthaccess.download(
        [granule],
        local_path=str(target_dir),
        threads=1,
        show_progress=True,
    )

    file_path = extract_downloaded_path(result)

    if not file_path.exists():
        # Sometimes the library returns a relative name.
        candidate = target_dir / file_path.name
        if candidate.exists():
            file_path = candidate

    if not file_path.exists():
        raise FileNotFoundError(
            f"Downloaded file was not found. Result={result!r}"
        )

    return file_path


def safe_delete(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        log.warning("Could not delete temporary file %s: %s", path, exc)


def cleanup_temp_directory() -> None:
    if TEMP_DIR.exists():
        for p in TEMP_DIR.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception as exc:
                log.warning("Could not clean %s: %s", p, exc)


def save_checkpoint(records: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for date_str, values in sorted(records.items()):
        row = {"date": date_str}
        row.update(values)
        rows.append(row)

    df = pd.DataFrame(rows)

    if "date" in df.columns:
        df = df.sort_values("date")

    df.to_csv(OUTPUT_CSV, index=False)


def load_existing_records() -> dict[str, dict]:
    if not OUTPUT_CSV.exists():
        return {}

    df = pd.read_csv(OUTPUT_CSV, dtype={"date": str})

    if "date" not in df.columns:
        raise RuntimeError(
            f"Existing output file has no 'date' column: {OUTPUT_CSV}"
        )

    records: dict[str, dict] = {}

    for _, row in df.iterrows():
        date_str = str(row["date"])

        values = {}
        for col in df.columns:
            if col == "date":
                continue

            value = row[col]

            if pd.isna(value):
                values[col] = np.nan
            else:
                try:
                    values[col] = float(value)
                except (TypeError, ValueError):
                    values[col] = value

        records[date_str] = values

    return records


def is_valid_record(record: dict) -> bool:
    value = record.get("precip_mm_day", np.nan)

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def yearly_search(year: int):
    """
    Search all Daily V07 granules for one calendar year.
    One CMR search per year, not one search per day.
    """
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    # Do not request dates outside the desired overall range.
    if year == 2025:
        end = "2025-09-30"

    return earthaccess.search_data(
        concept_id=CONCEPT_ID,
        version=VERSION,
        temporal=(start, end),
        cloud_hosted=False,
    )


def test_one_download(granule) -> tuple[Path, dict]:
    """
    Download exactly one granule and extract its data.
    This is the connectivity test before starting the historical run.
    """
    test_dir = TEMP_DIR / "preflight"
    test_dir.mkdir(parents=True, exist_ok=True)

    file_path = None

    try:
        print()
        print("Downloading ONE NASA IMERG file as pre-flight...")
        file_path = download_one(granule, test_dir)

        print(f"Downloaded: {file_path}")
        print(f"Size     : {file_path.stat().st_size / (1024 * 1024):.2f} MB")

        result = read_daily_granule(file_path)

        return file_path, result

    except Exception:
        if file_path is not None:
            safe_delete(file_path)
        raise


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    banner("NASA GPM IMERG FINAL DAILY V07")
    print(f"Mumbai latitude : {MUMBAI_LAT}")
    print(f"Mumbai longitude: {MUMBAI_LON}")
    print()
    print(f"Period          : {START_DATE} -> {END_DATE_EXCLUSIVE} (exclusive)")
    print("Product         : GPM_3IMERGDF V07")
    print("Resolution      : 0.1 degree, daily")
    print(f"Output          : {OUTPUT_CSV}")
    print(f"Temp directory  : {TEMP_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Authenticate
    # ------------------------------------------------------------

    banner("1. NASA EARTHDATA AUTHENTICATION")

    try:
        auth = earthaccess.login()
    except Exception as exc:
        sys.exit(
            "\nEarthdata login failed.\n"
            f"{type(exc).__name__}: {exc}\n"
        )

    print("Earthdata authentication successful.")

    # ------------------------------------------------------------
    # 2. Preflight search
    # ------------------------------------------------------------

    banner("2. ONE-FILE PREFLIGHT")

    print("Searching CMR for 2010-01-01...")

    try:
        preflight_results = earthaccess.search_data(
            concept_id=CONCEPT_ID,
            version=VERSION,
            temporal=("2010-01-01", "2010-01-01"),
            cloud_hosted=False,
            count=10,
        )
    except Exception as exc:
        sys.exit(
            "\nCMR search failed.\n"
            f"{type(exc).__name__}: {exc}\n"
        )

    if not preflight_results:
        sys.exit(
            "\nNo IMERG V07 Daily granule was returned for 2010-01-01.\n"
        )

    print("Granule found.")

    test_granule = preflight_results[0]

    try:
        test_path, test_values = test_one_download(test_granule)

        print()
        print("PRE-FLIGHT DOWNLOAD SUCCESS")
        print(f"Mumbai grid : {test_values['lat']:.5f}, {test_values['lon']:.5f}")
        print(f"Precip      : {test_values['precip_mm_day']:.4f} mm/day")

        for field in OPTIONAL_FIELDS:
            if field in test_values:
                print(f"{field:28s}: {test_values[field]}")

        cached_indices = {
            "lat_idx": test_values["_lat_idx"],
            "lon_idx": test_values["_lon_idx"],
            "time_axis": test_values["_time_axis"],
            "lon_axis": test_values["_lon_axis"],
            "lat_axis": test_values["_lat_axis"],
        }

        safe_delete(test_path)

    except Exception as exc:
        print()
        print("=" * 78)
        print("PRE-FLIGHT DOWNLOAD FAILED")
        print("=" * 78)
        print(type(exc).__name__, ":", exc)
        print()
        print(
            "This is now a pure NASA download/access test. "
            "The script has NOT started the 2010-2025 loop."
        )
        print()
        print(
            "If the error is HTTP 401/403, the Earthdata/GES DISC "
            "application authorization is still the blocker."
        )
        cleanup_temp_directory()
        sys.exit(1)

    # ------------------------------------------------------------
    # 3. Load existing records
    # ------------------------------------------------------------

    banner("3. RESUME CHECK")

    records = load_existing_records()

    print(f"Existing records: {len(records)}")

    # ------------------------------------------------------------
    # 4. Year-by-year search and one-file-at-a-time download
    # ------------------------------------------------------------

    banner("4. NASA DOWNLOAD + MUMBAI EXTRACTION")

    years = range(2010, 2026)

    total_processed = 0
    total_failed = 0

    for year in years:

        print()
        print("-" * 78)
        print(f"YEAR {year}")
        print("-" * 78)

        try:
            granules = yearly_search(year)
        except Exception as exc:
            log.error(
                "%d: CMR search failed: %s",
                year,
                exc,
            )
            total_failed += 1
            continue

        print(f"Granules found: {len(granules)}")

        # Sort by expected temporal/file order.
        try:
            granules = sorted(
                granules,
                key=lambda g: granule_date(g),
            )
        except Exception:
            pass

        year_done = 0

        for idx, granule in enumerate(granules, start=1):

            try:
                day = granule_date(granule)
            except Exception as exc:
                log.error(
                    "%d [%d/%d]: could not determine date: %s",
                    year,
                    idx,
                    len(granules),
                    exc,
                )
                total_failed += 1
                continue

            # Respect final overall period.
            if day < START_DATE or day >= END_DATE_EXCLUSIVE:
                continue

            # Already have a valid record -> skip.
            if day in records and is_valid_record(records[day]):
                continue

            success = False

            for attempt in range(1, MAX_RETRIES + 1):

                temp_file = None

                try:
                    log.info(
                        "%d [%d/%d] %s — downloading",
                        year,
                        idx,
                        len(granules),
                        day,
                    )

                    day_dir = TEMP_DIR / str(year)
                    day_dir.mkdir(parents=True, exist_ok=True)

                    temp_file = download_one(granule, day_dir)

                    values = read_daily_granule(
                        temp_file,
                        cached_indices=cached_indices,
                    )

                    # Keep only science fields in the persisted record.
                    persisted = {
                        k: v
                        for k, v in values.items()
                        if not k.startswith("_")
                    }

                    records[day] = persisted
                    success = True
                    total_processed += 1
                    year_done += 1

                    tag = persisted.get("precip_mm_day")
                    if isinstance(tag, (float, int)) and math.isfinite(float(tag)):
                        log.info(
                            "%d [%d/%d] %s -> %.4f mm/day",
                            year,
                            idx,
                            len(granules),
                            day,
                            float(tag),
                        )
                    else:
                        log.info(
                            "%d [%d/%d] %s -> NaN",
                            year,
                            idx,
                            len(granules),
                            day,
                        )

                    break

                except Exception as exc:

                    log.warning(
                        "%d [%d/%d] %s attempt %d/%d failed: %s",
                        year,
                        idx,
                        len(granules),
                        day,
                        attempt,
                        MAX_RETRIES,
                        exc,
                    )

                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_SLEEP_SECONDS * attempt)

                finally:
                    if temp_file is not None:
                        safe_delete(temp_file)

            if not success:
                records[day] = {
                    "precip_mm_day": np.nan,
                }
                total_failed += 1

            # Checkpoint regularly.
            if year_done > 0 and year_done % CHECKPOINT_EVERY == 0:
                save_checkpoint(records)
                log.info(
                    "Checkpoint saved: %d total records",
                    len(records),
                )

        # Save at end of every year.
        save_checkpoint(records)

        log.info(
            "Year %d complete. New records: %d | Total records: %d",
            year,
            year_done,
            len(records),
        )

        # Clean year temp folder.
        year_temp = TEMP_DIR / str(year)
        if year_temp.exists():
            try:
                shutil.rmtree(year_temp)
            except Exception as exc:
                log.warning(
                    "Could not remove %s: %s",
                    year_temp,
                    exc,
                )

    # ------------------------------------------------------------
    # 5. Final validation
    # ------------------------------------------------------------

    banner("5. FINAL VALIDATION")

    save_checkpoint(records)

    df = pd.read_csv(
        OUTPUT_CSV,
        parse_dates=["date"],
    )

    df = df.sort_values("date").reset_index(drop=True)

    expected_dates = pd.date_range(
        START_DATE,
        pd.Timestamp(END_DATE_EXCLUSIVE) - pd.Timedelta(days=1),
        freq="D",
    )

    expected = set(expected_dates.strftime("%Y-%m-%d"))
    actual = set(df["date"].dt.strftime("%Y-%m-%d"))

    missing_dates = sorted(expected - actual)

    nan_count = int(df["precip_mm_day"].isna().sum())

    print(f"Rows                  : {len(df)}")
    print(f"Expected days         : {len(expected_dates)}")
    print(f"Missing dates         : {len(missing_dates)}")
    print(f"NaN precipitation     : {nan_count}")
    print(f"Total successful new  : {total_processed}")
    print(f"Total failed attempts : {total_failed}")

    if not df.empty:
        print()
        print("Precipitation statistics:")
        print(df["precip_mm_day"].describe().to_string())

    if missing_dates:
        print()
        print("First missing dates:")
        for d in missing_dates[:20]:
            print("  ", d)

    if nan_count:
        print()
        print(
            "Some dates have NaN precipitation. Re-running this same "
            "script will retry those dates because only valid records "
            "are treated as complete."
        )

    if not missing_dates and nan_count == 0:
        print()
        print("FULL NASA IMERG 2010-09-30 COVERAGE: SUCCESS")

    # ------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------

    cleanup_temp_directory()

    print()
    print("=" * 78)
    print("BUILD FINISHED")
    print("=" * 78)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
