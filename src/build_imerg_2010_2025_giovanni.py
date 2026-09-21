
"""
Build hourly NASA GPM IMERG V07B satellite rainfall for Mumbai, 2010-2025.

Project:
D:/SIH 2026/mumbai_rainfall_flood_prototype

Output:
data/processed/imerg_giovanni/imerg_2010_2025_hourly.csv

Important:
- Uses NASA Giovanni Time Series API.
- Uses the validated IMERG V07 variable:
      GPM_3IMERGHH_07_precipitation
- Earthdata token is read from EARTHDATA_TOKEN when available.
- If EARTHDATA_TOKEN is not present, earthaccess is used to obtain a token.
- API timestamps are deliberately formatted as:
      YYYY-MM-DDTHH:MM:SS
  without "+00:00". This matches NASA's documented Giovanni API format.
- Monthly requests are used for reliability.
- Failed monthly requests are retried and then automatically split into
  smaller periods before giving up.
- Yearly checkpoint files are written, so an interrupted run can resume.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 19.05
LONGITUDE = 73.05

# ERA5 project coverage already built in this project:
# 2010-01-01 through 2025-09-30 23:00 UTC
START_DATE = pd.Timestamp("2010-01-01 00:00:00", tz="UTC")
END_DATE = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")  # exclusive

GIOVANNI_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"
CONFIGURED_VARIABLES_URL = (
    "https://api.giovanni.earthdata.nasa.gov/configured-variables"
)

# Validated from the NASA configured-variables endpoint in the project work.
KNOWN_IMERG_VARIABLE = "GPM_3IMERGHH_07_precipitation"

# Optional explicit version. The request function first follows NASA's
# documented time-series pattern, then retries with version=07 if required.
IMERG_VERSION = "07"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "imerg_giovanni"

FINAL_OUTPUT = OUTPUT_DIR / "imerg_2010_2025_hourly.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "imerg_2010_2025_summary.json"

REQUEST_TIMEOUT = 120
MAX_RETRIES_PER_VARIANT = 3
RETRY_BACKOFF_SECONDS = (5, 15, 30)

# Monthly is the normal request size.
# Failed months are recursively split, first to ~half month, then smaller.
MIN_SPLIT_DURATION = pd.Timedelta(days=1)

SLEEP_BETWEEN_SUCCESSFUL_REQUESTS = 0.75
SLEEP_AFTER_RETRYABLE_FAILURE = 2.0


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def info(message: str) -> None:
    print(message, flush=True)


# ============================================================
# TIME HELPERS
# ============================================================

def ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def api_time(ts: pd.Timestamp) -> str:
    """
    IMPORTANT:
    Giovanni documented examples use YYYY-MM-DDTHH:MM:SS and do not
    include '+00:00'. Using Timestamp.isoformat() here can produce
    '+00:00', which is exactly what we want to avoid.
    """
    ts = ensure_utc(ts)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    """Yield [month_start, month_end) ranges."""
    current = ensure_utc(start)
    end = ensure_utc(end)

    while current < end:
        month_start = current
        next_month = current + pd.offsets.MonthBegin(1)
        next_month = ensure_utc(next_month)
        month_end = min(next_month, end)

        yield month_start, month_end
        current = month_end


# ============================================================
# EARTHDATA AUTHENTICATION
# ============================================================

def get_earthdata_token() -> str:
    """
    Preferred order:
    1. EARTHDATA_TOKEN environment variable
    2. earthaccess.get_edl_token() after earthaccess login

    Never print the token.
    """
    banner("EARTHDATA AUTHENTICATION")

    token = os.environ.get("EARTHDATA_TOKEN", "").strip()

    if token:
        info("Using EARTHDATA_TOKEN from the environment.")
        return token

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError(
            "earthaccess is not installed and EARTHDATA_TOKEN is not set.\n"
            "Install with:\n"
            "    pip install earthaccess"
        ) from exc

    info("EARTHDATA_TOKEN not found. Starting earthaccess login...")

    # The environment strategy works with an already configured Earthdata
    # environment and avoids hardcoding credentials in this source file.
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        # Fall back to the normal earthaccess login flow if needed.
        earthaccess.login()

    try:
        token_data = earthaccess.get_edl_token()
        token = token_data["access_token"]
    except Exception as exc:
        raise RuntimeError(
            "Earthdata login succeeded, but an Earthdata access token "
            "could not be obtained."
        ) from exc

    token = str(token).strip()

    if not token:
        raise RuntimeError("Earthdata returned an empty access token.")

    info("Earthdata authentication successful.")
    return token


# ============================================================
# GIOVANNI VARIABLE DISCOVERY
# ============================================================

def flatten_strings(obj):
    """Yield strings recursively from arbitrary JSON-like objects."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from flatten_strings(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from flatten_strings(item)


def discover_imerg_variable(session: requests.Session) -> str:
    """
    Query NASA's configured-variables endpoint.

    The known V07 variable is retained as a fallback because it has already
    been validated in this project. This avoids making the entire pipeline
    dependent on minor configured-variables response format changes.
    """
    banner("DISCOVERING IMERG V07 VARIABLE")

    try:
        response = session.get(
            CONFIGURED_VARIABLES_URL,
            timeout=REQUEST_TIMEOUT,
        )

        info(f"Configured-variables HTTP status: {response.status_code}")

        if response.ok:
            try:
                payload = response.json()
                strings = list(flatten_strings(payload))

                exact = [
                    s for s in strings
                    if s == KNOWN_IMERG_VARIABLE
                ]
                if exact:
                    info(f"Selected variable: {KNOWN_IMERG_VARIABLE}")
                    return KNOWN_IMERG_VARIABLE

                candidates = sorted(
                    {
                        s
                        for s in strings
                        if s.startswith("GPM_3IMERGHH_07_")
                    }
                )

                if candidates:
                    precipitation = [
                        s for s in candidates
                        if "precipitation" in s.lower()
                        and "quality" not in s.lower()
                        and "probability" not in s.lower()
                        and "random" not in s.lower()
                    ]

                    if precipitation:
                        info("IMERG precipitation candidates:")
                        for item in precipitation:
                            info(f" - {item}")

                        info(f"Selected variable: {precipitation[0]}")
                        return precipitation[0]

            except Exception as exc:
                info(f"Variable-response parsing warning: {exc}")

    except requests.RequestException as exc:
        info(f"Configured-variables request warning: {exc}")

    info(
        "Falling back to the validated V07 precipitation variable:\n"
        f"  {KNOWN_IMERG_VARIABLE}"
    )
    return KNOWN_IMERG_VARIABLE


# ============================================================
# GIOVANNI RESPONSE PARSING
# ============================================================

def response_looks_like_csv(text: str) -> bool:
    if not text or not text.strip():
        return False

    lowered = text.lower()

    # JSON error response
    if lowered.lstrip().startswith("{") and '"message"' in lowered:
        return False

    return (
        "timestamp" in lowered
        or "param_name" in lowered
        or "time series" in lowered
    )


def parse_giovanni_csv(text: str) -> pd.DataFrame:
    """
    Parse Giovanni's CSV response robustly.

    NASA's tutorial currently describes metadata before the actual CSV table.
    We detect the table header rather than assuming an exact number of
    metadata rows.
    """
    if not text or not text.strip():
        raise ValueError("Giovanni returned an empty response.")

    text = text.replace("\ufeff", "").strip()
    lines = text.splitlines()

    # Find the CSV table header.
    header_idx: Optional[int] = None

    for i, line in enumerate(lines):
        first = line.split(",", 1)[0].strip().strip('"').lower()
        if first in {"timestamp", "time"}:
            header_idx = i
            break

        # Some versions use Timestamp inside a longer first field.
        if first.startswith("timestamp"):
            header_idx = i
            break

    if header_idx is None:
        # A second parser for responses with comment-style metadata.
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            raise ValueError(
                "Could not locate Giovanni CSV table header."
            ) from exc
    else:
        csv_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text))

    if df.empty:
        raise ValueError("Giovanni returned an empty data table.")

    # Standardize first column to timestamp.
    timestamp_col = None
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in {"timestamp", "time", "date"}:
            timestamp_col = col
            break

    if timestamp_col is None:
        # Fall back to first column.
        timestamp_col = df.columns[0]

    df = df.rename(columns={timestamp_col: "timestamp"})

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["timestamp"]).copy()

    if df.empty:
        raise ValueError("No valid timestamps were found in Giovanni response.")

    # Find the measurement column.
    measurement_candidates = [
        c for c in df.columns
        if c != "timestamp"
    ]

    if not measurement_candidates:
        raise ValueError("No measurement column found in Giovanni response.")

    value_col = measurement_candidates[0]

    # Prefer a numeric-looking column if several exist.
    for c in measurement_candidates:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().sum() > 0:
            value_col = c
            break

    df["imerg_rain_rate_mmhr"] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    # Replace common missing/fill values.
    df.loc[
        ~np.isfinite(df["imerg_rain_rate_mmhr"]),
        "imerg_rain_rate_mmhr",
    ] = np.nan

    df.loc[
        df["imerg_rain_rate_mmhr"] < -9000,
        "imerg_rain_rate_mmhr",
    ] = np.nan

    # Physical guard.
    df.loc[
        df["imerg_rain_rate_mmhr"] < 0,
        "imerg_rain_rate_mmhr",
    ] = 0.0

    result = df[
        ["timestamp", "imerg_rain_rate_mmhr"]
    ].copy()

    result = (
        result
        .dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError("No valid IMERG rainfall values were parsed.")

    return result


# ============================================================
# SINGLE GIOVANNI REQUEST
# ============================================================

def request_timeseries_once(
    session: requests.Session,
    token: str,
    variable: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_version: bool,
) -> pd.DataFrame:
    """
    One Giovanni request.

    First/normal format:
        data=<variable>
        location=[19.05,73.05]
        time=YYYY-MM-DDTHH:MM:SS/YYYY-MM-DDTHH:MM:SS

    Optional fallback:
        version=07
    """
    start = ensure_utc(start)
    end = ensure_utc(end)

    if not start < end:
        raise ValueError(f"Invalid request period: {start} -> {end}")

    params = {
        "data": variable,
        "location": f"[{LATITUDE},{LONGITUDE}]",
        "time": f"{api_time(start)}/{api_time(end)}",
    }

    if use_version:
        params["version"] = IMERG_VERSION

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/csv,*/*",
        "User-Agent": "SIH2026-Mumbai-IMERG-Pipeline/1.0",
    }

    response = session.get(
        GIOVANNI_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        error_preview = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"HTTP {response.status_code}: {error_preview}"
        )

    if not response_looks_like_csv(response.text):
        preview = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            "Giovanni returned a non-CSV response: " + preview
        )

    return parse_giovanni_csv(response.text)


# ============================================================
# ROBUST PERIOD REQUEST
# ============================================================

def fetch_period(
    session: requests.Session,
    token: str,
    variable: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    depth: int = 0,
) -> pd.DataFrame:
    """
    Fetch one interval with:
      1. normal Giovanni request
      2. version=07 fallback
      3. exponential retries
      4. recursive interval splitting if the service rejects the range

    This is intentionally designed to avoid the previous:
        HTTP 500: {"message":null}
    stopping the entire build.
    """
    start = ensure_utc(start)
    end = ensure_utc(end)

    indent = "  " * depth
    duration = end - start

    info(
        f"{indent}Requesting "
        f"{api_time(start)} → {api_time(end)} "
        f"({duration})"
    )

    last_error: Optional[Exception] = None

    variants = [
        ("standard", False),
        ("explicit-version", True),
    ]

    for variant_name, use_version in variants:
        for attempt in range(1, MAX_RETRIES_PER_VARIANT + 1):
            try:
                data = request_timeseries_once(
                    session=session,
                    token=token,
                    variable=variable,
                    start=start,
                    end=end,
                    use_version=use_version,
                )

                if data.empty:
                    raise ValueError(
                        f"{variant_name} request returned no rows."
                    )

                info(
                    f"{indent}Success: {variant_name}, "
                    f"{len(data):,} rows"
                )

                return data

            except Exception as exc:
                last_error = exc
                info(
                    f"{indent}{variant_name} attempt "
                    f"{attempt}/{MAX_RETRIES_PER_VARIANT} failed: {exc}"
                )

                if attempt < MAX_RETRIES_PER_VARIANT:
                    delay = RETRY_BACKOFF_SECONDS[
                        min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    time.sleep(delay)

        info(
            f"{indent}Variant '{variant_name}' failed."
        )

    # At this point, automatically split the request.
    if duration > MIN_SPLIT_DURATION:
        midpoint = start + duration / 2

        # Snap to 30-minute grid.
        midpoint = midpoint.floor("30min")

        if midpoint <= start or midpoint >= end:
            midpoint = start + MIN_SPLIT_DURATION

        if midpoint < end:
            info(
                f"{indent}Splitting failed request into smaller periods."
            )

            left = fetch_period(
                session=session,
                token=token,
                variable=variable,
                start=start,
                end=midpoint,
                depth=depth + 1,
            )

            time.sleep(SLEEP_AFTER_RETRYABLE_FAILURE)

            right = fetch_period(
                session=session,
                token=token,
                variable=variable,
                start=midpoint,
                end=end,
                depth=depth + 1,
            )

            return (
                pd.concat([left, right], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

    error_text = str(last_error) if last_error else "unknown error"
    raise RuntimeError(
        f"Giovanni could not retrieve period "
        f"{api_time(start)} → {api_time(end)}. "
        f"Final error: {error_text}"
    )


# ============================================================
# HOURLY AGGREGATION
# ============================================================

def halfhour_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IMERG half-hourly rainfall rate (mm/hr) to hourly rainfall amount.

    For a complete two-half-hour hour:
        hourly_mm = rate_1 * 0.5 + rate_2 * 0.5

    Since each rate is mm/hr, the result is mm of rain over the full hour.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "imerg_hourly_mm",
                "imerg_mean_rate_mmhr",
                "imerg_halfhour_count",
            ]
        )

    work = df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
        utc=True,
    )

    work["imerg_rain_rate_mmhr"] = pd.to_numeric(
        work["imerg_rain_rate_mmhr"],
        errors="coerce",
    )

    work = work.dropna(
        subset=["timestamp", "imerg_rain_rate_mmhr"]
    )

    # Floor each sample to its containing hour.
    work["hour"] = work["timestamp"].dt.floor("1h")

    grouped = work.groupby("hour")["imerg_rain_rate_mmhr"]

    out = grouped.agg(
        imerg_halfhour_count="count",
        imerg_mean_rate_mmhr="mean",
    ).reset_index()

    # Exact hourly rainfall from half-hourly rates.
    rainfall_sum = (
        grouped.sum(min_count=1) * 0.5
    ).rename("imerg_hourly_mm").reset_index()

    out = out.merge(
        rainfall_sum,
        on="hour",
        how="left",
    )

    out = out.rename(columns={"hour": "timestamp"})

    # We want complete two-sample hours in the satellite series.
    # One-sample hours remain NaN instead of silently doubling a half-hour.
    out.loc[
        out["imerg_halfhour_count"] < 2,
        "imerg_hourly_mm",
    ] = np.nan

    out.loc[
        out["imerg_halfhour_count"] < 2,
        "imerg_mean_rate_mmhr",
    ] = np.nan

    return (
        out[
            [
                "timestamp",
                "imerg_hourly_mm",
                "imerg_mean_rate_mmhr",
                "imerg_halfhour_count",
            ]
        ]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


# ============================================================
# YEAR PROCESSING
# ============================================================

def expected_hour_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start) / pd.Timedelta(hours=1))


def yearly_checkpoint_path(year: int) -> Path:
    return OUTPUT_DIR / f"imerg_{year}_hourly.csv"


def read_valid_checkpoint(
    path: Path,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Read a yearly checkpoint only when it is structurally usable.
    """
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        required = {
            "timestamp",
            "imerg_hourly_mm",
            "imerg_mean_rate_mmhr",
            "imerg_halfhour_count",
        }

        if not required.issubset(df.columns):
            return None

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if df.empty:
            return None

        # Do not trust a checkpoint that is incomplete. Rebuild it so the
        # final 2010-2025 series is not silently missing satellite hours.
        expected = expected_hour_count(year_start, year_end)

        expected_index = pd.date_range(
            year_start,
            year_end - pd.Timedelta(hours=1),
            freq="1h",
            tz="UTC",
        )
        actual_index = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()

        if len(df) != expected:
            return None

        if len(expected_index.difference(actual_index)) != 0:
            return None

        return df.reset_index(drop=True)

    except Exception:
        return None


def build_year(
    session: requests.Session,
    token: str,
    variable: str,
    year: int,
) -> pd.DataFrame:
    """
    Build one calendar year from monthly Giovanni requests.
    """
    year_start = max(
        START_DATE,
        pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC"),
    )
    year_end = min(
        END_DATE,
        pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC"),
    )

    if year_start >= year_end:
        return pd.DataFrame()

    checkpoint = yearly_checkpoint_path(year)

    cached = read_valid_checkpoint(
        checkpoint,
        year_start,
        year_end,
    )

    if cached is not None:
        info(
            f"Checkpoint found for {year}: "
            f"{len(cached):,} rows. Skipping network download."
        )
        return cached

    banner(f"PROCESSING YEAR {year}")

    pieces = []

    for month_start, month_end in month_ranges(
        year_start,
        year_end,
    ):
        info(
            f"\nProcessing month: "
            f"{api_time(month_start)} → {api_time(month_end)}"
        )

        try:
            piece = fetch_period(
                session=session,
                token=token,
                variable=variable,
                start=month_start,
                end=month_end,
            )

            pieces.append(piece)

            time.sleep(SLEEP_BETWEEN_SUCCESSFUL_REQUESTS)

        except Exception as exc:
            info(
                "\nMONTH FAILED AFTER ALL FALLBACKS\n"
                f"Year : {year}\n"
                f"Month: {api_time(month_start)} → {api_time(month_end)}\n"
                f"Reason: {exc}"
            )

            # Retry the month once as smaller fixed chunks before declaring it
            # failed. This provides another layer of protection against a
            # Giovanni server-side 500 for a larger time range.
            retry_pieces = []
            cursor = month_start

            while cursor < month_end:
                chunk_end = min(
                    cursor + pd.Timedelta(days=3),
                    month_end,
                )

                try:
                    chunk = fetch_period(
                        session=session,
                        token=token,
                        variable=variable,
                        start=cursor,
                        end=chunk_end,
                    )
                    retry_pieces.append(chunk)
                except Exception as chunk_exc:
                    info(
                        "3-day fallback chunk failed:\n"
                        f"  {api_time(cursor)} → {api_time(chunk_end)}\n"
                        f"  {chunk_exc}"
                    )

                    # Try one-day chunks only for this failed 3-day chunk.
                    one_day = cursor

                    while one_day < chunk_end:
                        one_day_end = min(
                            one_day + pd.Timedelta(days=1),
                            chunk_end,
                        )

                        try:
                            one_day_df = fetch_period(
                                session=session,
                                token=token,
                                variable=variable,
                                start=one_day,
                                end=one_day_end,
                            )
                            retry_pieces.append(one_day_df)
                        except Exception as day_exc:
                            raise RuntimeError(
                                "Satellite download failed even at one-day "
                                "resolution:\n"
                                f"{api_time(one_day)} → "
                                f"{api_time(one_day_end)}\n"
                                f"{day_exc}"
                            ) from day_exc

                        one_day = one_day_end

                cursor = chunk_end

            if retry_pieces:
                pieces.extend(retry_pieces)

    if not pieces:
        raise RuntimeError(f"No IMERG data were collected for {year}.")

    raw = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Monthly/time-range API responses should be half-hourly. Convert them.
    hourly = halfhour_to_hourly(raw)

    # Restrict exactly to this year's requested interval.
    hourly = hourly[
        (hourly["timestamp"] >= year_start)
        & (hourly["timestamp"] < year_end)
    ].copy()

    hourly = (
        hourly
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    hourly.to_csv(
        checkpoint,
        index=False,
        float_format="%.6f",
    )

    info(
        f"\nSaved yearly checkpoint:\n"
        f"  {checkpoint}\n"
        f"  Rows: {len(hourly):,}"
    )

    return hourly


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_output(df: pd.DataFrame) -> dict:
    result = {}

    if df.empty:
        raise RuntimeError("Final IMERG dataframe is empty.")

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    result["rows"] = int(len(df))
    result["start"] = str(df["timestamp"].min())
    result["end"] = str(df["timestamp"].max())
    result["duplicate_timestamps"] = int(
        df["timestamp"].duplicated().sum()
    )
    result["missing_timestamps"] = int(df["timestamp"].isna().sum())
    result["missing_hourly_rainfall"] = int(
        df["imerg_hourly_mm"].isna().sum()
    )
    result["mean_hourly_mm"] = float(
        df["imerg_hourly_mm"].mean()
    )
    result["max_hourly_mm"] = float(
        df["imerg_hourly_mm"].max()
    )

    # Expected complete period.
    expected_index = pd.date_range(
        START_DATE,
        END_DATE - pd.Timedelta(hours=1),
        freq="1h",
        tz="UTC",
    )

    actual_index = pd.DatetimeIndex(
        df["timestamp"].dropna().unique()
    ).sort_values()

    missing_expected = expected_index.difference(actual_index)

    result["expected_hourly_rows"] = int(len(expected_index))
    result["missing_expected_hours"] = int(len(missing_expected))

    # Basic sanity checks.
    if result["duplicate_timestamps"] != 0:
        raise RuntimeError(
            f"Final output has {result['duplicate_timestamps']} "
            "duplicate timestamps."
        )

    if result["missing_timestamps"] != 0:
        raise RuntimeError(
            f"Final output has {result['missing_timestamps']} invalid timestamps."
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    banner("NASA GPM IMERG V07B 2010-2025 BUILD")

    info("Mumbai point:")
    info(f"Latitude : {LATITUDE}")
    info(f"Longitude: {LONGITUDE}")

    info("\nPeriod:")
    info(f"{START_DATE} → {END_DATE} (END_DATE exclusive)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = get_earthdata_token()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "SIH2026-Mumbai-IMERG-Pipeline/1.0",
            "Accept": "text/csv,*/*",
        }
    )

    variable = discover_imerg_variable(session)

    all_years = []

    first_year = START_DATE.year
    last_year = (END_DATE - pd.Timedelta(seconds=1)).year

    for year in range(first_year, last_year + 1):
        year_df = build_year(
            session=session,
            token=token,
            variable=variable,
            year=year,
        )

        if not year_df.empty:
            all_years.append(year_df)

            # Persist a cumulative checkpoint after every successful year.
            cumulative = (
                pd.concat(all_years, ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            cumulative.to_csv(
                FINAL_OUTPUT,
                index=False,
                float_format="%.6f",
            )

            info(
                f"Cumulative output updated: "
                f"{len(cumulative):,} rows"
            )

    if not all_years:
        raise RuntimeError(
            "No yearly IMERG data were produced."
        )

    final_df = (
        pd.concat(all_years, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Exact final temporal restriction.
    final_df = final_df[
        (final_df["timestamp"] >= START_DATE)
        & (final_df["timestamp"] < END_DATE)
    ].copy()

    final_df = (
        final_df
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    validation = validate_final_output(final_df)

    final_df.to_csv(
        FINAL_OUTPUT,
        index=False,
        float_format="%.6f",
    )

    summary = {
        "source": "NASA GPM IMERG V07B",
        "collection": "GPM_3IMERGHH",
        "variable": variable,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "period_start": START_DATE.isoformat(),
        "period_end_exclusive": END_DATE.isoformat(),
        "output_file": str(FINAL_OUTPUT),
        "validation": validation,
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    banner("IMERG BUILD COMPLETE")

    info(f"Output CSV : {FINAL_OUTPUT}")
    info(f"Summary    : {SUMMARY_OUTPUT}")
    info(f"Rows       : {len(final_df):,}")
    info(f"Start      : {final_df['timestamp'].min()}")
    info(f"End        : {final_df['timestamp'].max()}")
    info(
        f"Missing expected hours: "
        f"{validation['missing_expected_hours']:,}"
    )
    info(
        f"Missing rainfall values: "
        f"{validation['missing_hourly_rainfall']:,}"
    )
    info(f"Max hourly rain (mm): {validation['max_hourly_mm']:.3f}")

    if validation["missing_expected_hours"] == 0:
        info("\nFULL HOURLY COVERAGE CONFIRMED.")
    else:
        info(
            "\nWARNING: Some expected hourly timestamps are missing. "
            "Do not call the final ML dataset complete until those gaps "
            "are handled."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as exc:
        print("\n" + "=" * 70)
        print("BUILD STOPPED")
        print("=" * 70)
        print(str(exc))
        print(
            "\nThe script stopped cleanly and did not print a Python traceback."
        )
        raise SystemExit(1)
