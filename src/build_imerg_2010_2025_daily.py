
"""
FINAL IMERG V07 DAILY BUILDER — PYDAP + EARTHACCESS
====================================================

This is the locked access method for the SIH 2026 project.

Why this version is different from the previous failed OPeNDAP script:
1. It uses Earthdata's Auth session directly:
       auth.get_session()
2. It wraps that session with PyDAP's create_session():
       create_session(session=auth.get_session())
3. It explicitly requests DAP4:
       open_url(..., session=..., protocol="dap4")
4. It uses PyDAP directly instead of asking xarray to construct the
   PyDAP session. This follows the current PyDAP + earthaccess NASA
   authentication workflow.

The project already successfully opened an IMERG V07 half-hourly granule
with PyDAP. The earlier failure was during variable indexing. This version
keeps the known-good authentication pattern and performs the point
extraction directly with PyDAP.

Product:
    GPM_3IMERGDF
    Version 07
    Final Daily
    0.1 degree
    Daily precipitation in mm/day

Period:
    2010-01-01 -> 2025-09-30

Output:
    data/processed/imerg_2010_2025_daily.csv

Year checkpoints:
    data/processed/imerg_yearly/imerg_YYYY_daily.csv

A mandatory 2010-01-01 preflight runs before the full build.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 19.05
LONGITUDE = 73.05

START_DATE = pd.Timestamp(
    "2010-01-01 00:00:00",
    tz="UTC",
)

END_DATE_EXCLUSIVE = pd.Timestamp(
    "2025-10-01 00:00:00",
    tz="UTC",
)

SHORT_NAME = "GPM_3IMERGDF"
VERSION = "07"

# Confirmed from the NASA GES DISC V07 collection metadata.
CONCEPT_ID = "C2723754864-GES_DISC"

OPENDAP_ROOT = (
    "https://opendap.earthdata.nasa.gov"
)

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

YEARLY_DIR = (
    PROCESSED_DIR / "imerg_yearly"
)

FINAL_OUTPUT = (
    PROCESSED_DIR /
    "imerg_2010_2025_daily.csv"
)

SUMMARY_OUTPUT = (
    PROCESSED_DIR /
    "imerg_2010_2025_daily_summary.json"
)

RETRIES = 3
RETRY_SLEEP = 4
SLEEP_BETWEEN_GRANULES = 0.10


# ============================================================
# CONSOLE
# ============================================================

def banner(text: str) -> None:
    print(
        "\n" + "=" * 70
    )
    print(
        text
    )
    print(
        "=" * 70,
        flush=True,
    )


def ensure_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize(
            "UTC"
        )

    return ts.tz_convert(
        "UTC"
    )


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate():
    """
    Authenticate with NASA Earthdata and create the PyDAP session using
    the Earthdata bearer token directly.

    IMPORTANT:
    The previous approach used:

        create_session(session=auth.get_session())

    That session authenticated CMR successfully but still received a 401
    from the Cloud OPeNDAP .dmr endpoint in the local environment.

    PyDAP also supports direct token authentication. We therefore obtain
    the EDL token from earthaccess and pass it through:

        create_session(session_kwargs={"token": token})

    The token value is never printed.
    """
    banner(
        "NASA EARTHDATA AUTHENTICATION"
    )

    import os
    import earthaccess
    from pydap.net import create_session

    # First use the existing EARTHDATA_TOKEN from the current PowerShell
    # session. If it is not present, fall back to earthaccess's normal
    # authentication mechanism without ever asking the user to paste a
    # token into this program.
    env_token = os.environ.get("EARTHDATA_TOKEN", "").strip()

    if env_token:
        auth = earthaccess.login(
            strategy="environment"
        )
    else:
        auth = earthaccess.login(
            strategy="all"
        )

    if auth is None:
        raise RuntimeError(
            "Earthdata authentication returned no Auth object."
        )

    # earthaccess exposes the current EDL token directly.
    token = ""
    try:
        token = auth.get_edl_token()
    except Exception:
        token = ""

    if not token:
        token = env_token

    if not token:
        raise RuntimeError(
            "Earthdata authentication succeeded, but no EDL token "
            "could be obtained."
        )

    # CRITICAL FIX:
    # Pass the token directly to PyDAP instead of passing the requests
    # session returned by auth.get_session().
    pydap_session = create_session(
        session_kwargs={
            "token": token
        }
    )

    print(
        "Earthdata authentication successful.",
        flush=True,
    )

    print(
        "PyDAP token-authenticated session created.",
        flush=True,
    )

    return earthaccess, pydap_session


# ============================================================
# CMR SEARCH
# ============================================================

def year_bounds(year: int):
    start = max(
        START_DATE,
        pd.Timestamp(
            f"{year}-01-01 00:00:00",
            tz="UTC",
        ),
    )

    end = min(
        END_DATE_EXCLUSIVE,
        pd.Timestamp(
            f"{year + 1}-01-01 00:00:00",
            tz="UTC",
        ),
    )

    return start, end


def expected_days(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    return pd.date_range(
        start,
        end - pd.Timedelta(
            days=1
        ),
        freq="1D",
        tz="UTC",
    )


def search_year(
    earthaccess,
    year: int,
):
    start, end = year_bounds(
        year
    )

    query_end = (
        end -
        pd.Timedelta(
            seconds=1
        )
    )

    print(
        f"\nSearching {year}: "
        f"{start.strftime('%Y-%m-%d')} -> "
        f"{query_end.strftime('%Y-%m-%d')}",
        flush=True,
    )

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(
            start.strftime(
                "%Y-%m-%dT00:00:00Z"
            ),
            query_end.strftime(
                "%Y-%m-%dT23:59:59Z"
            ),
        ),
        count=400,
    )

    def sort_key(granule):
        try:
            return str(
                granule.get(
                    "umm",
                    {}
                )
                .get(
                    "TemporalExtent",
                    {}
                )
                .get(
                    "RangeDateTime",
                    {}
                )
                .get(
                    "BeginningDateTime",
                    ""
                )
            )
        except Exception:
            return ""

    results = sorted(
        results,
        key=sort_key,
    )

    print(
        f"Granules found: {len(results)}",
        flush=True,
    )

    return results


# ============================================================
# OPENDAP URL
# ============================================================

def external_url(granule) -> str:
    try:
        links = granule.data_links(
            access="external"
        )

        for link in links:
            if (
                isinstance(link, str)
                and link.startswith(
                    "https://"
                )
                and "GPM_3IMERGDF" in link
            ):
                return link

    except Exception:
        pass

    related_urls = (
        granule.get(
            "umm",
            {}
        )
        .get(
            "RelatedUrls",
            []
        )
    )

    for item in related_urls:
        url = item.get(
            "URL",
            ""
        )

        if (
            isinstance(url, str)
            and url.startswith(
                "https://"
            )
            and "GPM_3IMERGDF" in url
        ):
            return url

    raise RuntimeError(
        "Could not find the IMERG data URL."
    )


def opendap_url(granule) -> str:
    """
    Construct the Cloud OPeNDAP granule URL.

    Example:
      https://opendap.earthdata.nasa.gov/collections/
      C2723754864-GES_DISC/granules/
      GPM_3IMERGDF.07%3A3B-DAY....V07B.nc4

    DAP4 is requested explicitly in open_url().
    """
    from urllib.parse import quote

    source = external_url(
        granule
    )

    filename = (
        source
        .split("/")
        [-1]
        .split("?")
        [0]
    )

    if not filename:
        raise RuntimeError(
            "Could not determine the IMERG filename."
        )

    granule_id = (
        f"{SHORT_NAME}.{VERSION}:{filename}"
    )

    return (
        f"{OPENDAP_ROOT}/collections/"
        f"{CONCEPT_ID}/granules/"
        f"{quote(granule_id, safe='')}"
    )


# ============================================================
# PYDAP HELPERS
# ============================================================

def node_keys(node):
    try:
        return list(
            node.keys()
        )
    except Exception:
        return []


def find_named_node(
    root,
    wanted_name: str,
    max_depth: int = 5,
):
    """
    Recursively find a PyDAP variable/group by its final name.

    This handles both:
        root.precipitation
    and:
        root.Grid.precipitation

    without assuming the exact NetCDF grouping layout.
    """
    wanted = (
        wanted_name
        .strip("/")
        .split("/")[-1]
    )

    visited = set()

    def walk(
        node,
        depth: int,
    ):
        if depth > max_depth:
            return None

        marker = id(node)

        if marker in visited:
            return None

        visited.add(
            marker
        )

        keys = node_keys(
            node
        )

        for key in keys:
            try:
                child = node[key]
            except Exception:
                continue

            clean_key = (
                str(key)
                .strip("/")
                .split("/")[-1]
            )

            if clean_key == wanted:
                return child

            if (
                hasattr(child, "keys")
                and depth < max_depth
            ):
                found = walk(
                    child,
                    depth + 1,
                )

                if found is not None:
                    return found

        return None

    return walk(
        root,
        0,
    )


def node_dimensions(node):
    try:
        dims = tuple(
            node.dimensions
        )
        return dims
    except Exception:
        return ()


def get_scalar_or_first(
    node,
):
    values = np.asarray(
        node.data
    ).reshape(-1)

    if values.size == 0:
        raise RuntimeError(
            "Remote variable returned no values."
        )

    return values[0]


# ============================================================
# ONE GRANULE
# ============================================================

def read_one_granule(
    session,
    granule,
):
    from pydap.client import open_url

    url = opendap_url(
        granule
    )

    print(
        f"      OPeNDAP URL:\n"
        f"      {url}",
        flush=True,
    )

    last_error = None

    for attempt in range(
        1,
        RETRIES + 1,
    ):
        dataset = None

        try:
            print(
                f"      Opening OPeNDAP with PyDAP "
                f"DAP4 ({attempt}/{RETRIES})...",
                flush=True,
            )

            # THIS is the important fix:
            # PyDAP receives the authenticated session directly.
            dataset = open_url(
                url,
                session=session,
                protocol="dap4",
                timeout=120,
            )

            print(
                "      OPeNDAP dataset opened.",
                flush=True,
            )

            # Locate coordinates and precipitation without assuming a
            # particular Group/ path.
            lat_node = find_named_node(
                dataset,
                "lat",
            )

            lon_node = find_named_node(
                dataset,
                "lon",
            )

            precipitation = find_named_node(
                dataset,
                "precipitation",
            )

            if (
                lat_node is None
                or lon_node is None
                or precipitation is None
            ):
                raise RuntimeError(
                    "Could not locate the required IMERG variables.\n"
                    f"Top-level keys: {node_keys(dataset)}"
                )

            # Load only the 1-D coordinate arrays.
            lat_values = np.asarray(
                lat_node.data
            ).reshape(-1)

            lon_values = np.asarray(
                lon_node.data
            ).reshape(-1)

            if lat_values.size == 0:
                raise RuntimeError(
                    "Latitude coordinate array is empty."
                )

            if lon_values.size == 0:
                raise RuntimeError(
                    "Longitude coordinate array is empty."
                )

            lat_idx = int(
                np.abs(
                    lat_values
                    - LATITUDE
                ).argmin()
            )

            lon_idx = int(
                np.abs(
                    lon_values
                    - LONGITUDE
                ).argmin()
            )

            nearest_lat = float(
                lat_values[
                    lat_idx
                ]
            )

            nearest_lon = float(
                lon_values[
                    lon_idx
                ]
            )

            print(
                f"      Mumbai pixel: "
                f"{nearest_lat:.5f}, "
                f"{nearest_lon:.5f}",
                flush=True,
            )

            dims = node_dimensions(
                precipitation
            )

            print(
                f"      precipitation dimensions: "
                f"{dims}",
                flush=True,
            )

            # Known IMERG ordering from the V07 gridded product:
            # time, lon, lat.
            #
            # We identify the indexes by dimension names instead of blindly
            # assuming the order.
            if len(dims) == 3:
                index_tuple = []

                for dim in dims:
                    clean = (
                        str(dim)
                        .strip("/")
                        .split("/")[-1]
                        .lower()
                    )

                    if clean == "time":
                        index_tuple.append(0)

                    elif clean == "lon":
                        index_tuple.append(
                            lon_idx
                        )

                    elif clean == "lat":
                        index_tuple.append(
                            lat_idx
                        )

                    else:
                        # Daily IMERG precipitation should only have
                        # time/lon/lat. If an unexpected dimension exists,
                        # select its first index.
                        index_tuple.append(0)

                # PyDAP performs a remote constrained request.
                selected = (
                    precipitation[
                        tuple(index_tuple)
                    ]
                )

            elif len(dims) == 2:
                # Fallback for a time-less daily field.
                index_tuple = []

                for dim in dims:
                    clean = (
                        str(dim)
                        .strip("/")
                        .split("/")[-1]
                        .lower()
                    )

                    if clean == "lon":
                        index_tuple.append(
                            lon_idx
                        )

                    elif clean == "lat":
                        index_tuple.append(
                            lat_idx
                        )

                    else:
                        index_tuple.append(0)

                selected = (
                    precipitation[
                        tuple(index_tuple)
                    ]
                )

            else:
                raise RuntimeError(
                    "Unexpected precipitation dimensions: "
                    f"{dims}"
                )

            rain = float(
                np.asarray(
                    selected.data
                ).reshape(-1)[0]
            )

            # Quality variables.
            def optional_variable(
                name: str,
            ):
                node = find_named_node(
                    dataset,
                    name,
                )

                if node is None:
                    return np.nan

                try:
                    node_dims = (
                        node_dimensions(
                            node
                        )
                    )

                    if len(node_dims) == 3:
                        indices = []

                        for dim in node_dims:
                            clean = (
                                str(dim)
                                .strip("/")
                                .split("/")[-1]
                                .lower()
                            )

                            if clean == "time":
                                indices.append(0)

                            elif clean == "lon":
                                indices.append(
                                    lon_idx
                                )

                            elif clean == "lat":
                                indices.append(
                                    lat_idx
                                )

                            else:
                                indices.append(0)

                        value = float(
                            np.asarray(
                                node[
                                    tuple(indices)
                                ].data
                            )
                            .reshape(-1)[0]
                        )

                    elif len(node_dims) == 2:
                        indices = []

                        for dim in node_dims:
                            clean = (
                                str(dim)
                                .strip("/")
                                .split("/")[-1]
                                .lower()
                            )

                            if clean == "lon":
                                indices.append(
                                    lon_idx
                                )

                            elif clean == "lat":
                                indices.append(
                                    lat_idx
                                )

                            else:
                                indices.append(0)

                        value = float(
                            np.asarray(
                                node[
                                    tuple(indices)
                                ].data
                            )
                            .reshape(-1)[0]
                        )

                    else:
                        value = float(
                            get_scalar_or_first(
                                node
                            )
                        )

                    if (
                        not np.isfinite(
                            value
                        )
                        or value < -9000
                    ):
                        return np.nan

                    return value

                except Exception:
                    return np.nan

            valid_count = optional_variable(
                "precipitation_cnt"
            )

            rainy_count = optional_variable(
                "precipitation_cnt_cond"
            )

            random_error = optional_variable(
                "randomError"
            )

            random_error_count = optional_variable(
                "randomError_cnt"
            )

            liquid_probability = optional_variable(
                "probabilityLiquidPrecipitation"
            )

            # Every Daily granule corresponds to one UTC day. Use its
            # CMR beginning timestamp rather than performing another
            # remote time-variable request.
            begin = (
                granule.get(
                    "umm",
                    {}
                )
                .get(
                    "TemporalExtent",
                    {}
                )
                .get(
                    "RangeDateTime",
                    {}
                )
                .get(
                    "BeginningDateTime"
                )
            )

            if not begin:
                raise RuntimeError(
                    "Granule metadata contains no beginning timestamp."
                )

            timestamp = (
                ensure_utc(
                    begin
                )
                .floor("1D")
            )

            if (
                not np.isfinite(
                    rain
                )
                or rain < -9000
                or rain < 0
            ):
                rain = np.nan

            return {
                "timestamp": timestamp,
                "latitude": nearest_lat,
                "longitude": nearest_lon,
                "imerg_daily_mm": rain,
                "imerg_valid_halfhour_count":
                    valid_count,
                "imerg_rainy_halfhour_count":
                    rainy_count,
                "imerg_random_error_mm":
                    random_error,
                "imerg_random_error_count":
                    random_error_count,
                "imerg_probability_liquid_pct":
                    liquid_probability,
            }

        except Exception as exc:
            last_error = exc

            print(
                f"      PyDAP attempt "
                f"{attempt}/{RETRIES} failed: "
                f"{exc}",
                flush=True,
            )

            if attempt < RETRIES:
                time.sleep(
                    RETRY_SLEEP * attempt
                )

        finally:
            if dataset is not None:
                try:
                    dataset.close()
                except Exception:
                    pass

    raise RuntimeError(
        "IMERG granule failed through authenticated PyDAP "
        f"after {RETRIES} attempts.\n"
        f"Final error: {last_error}"
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight(
    earthaccess,
    session,
):
    banner(
        "FINAL V07 PYDAP PREFLIGHT"
    )

    print(
        "Testing exactly ONE IMERG V07 Daily granule "
        "for 2010-01-01.",
        flush=True,
    )

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(
            "2010-01-01T00:00:00Z",
            "2010-01-01T23:59:59Z",
        ),
        count=1,
    )

    if not results:
        raise RuntimeError(
            "NASA CMR returned no IMERG V07 Daily granule "
            "for 2010-01-01."
        )

    print(
        "Granule found.",
        flush=True,
    )

    row = read_one_granule(
        session=session,
        granule=results[0],
    )

    if pd.isna(
        row["imerg_daily_mm"]
    ):
        raise RuntimeError(
            "Preflight opened the granule but the Mumbai "
            "precipitation value is missing."
        )

    print(
        "\nPRE-FLIGHT SUCCESS.",
        flush=True,
    )

    print(
        f"Mumbai grid: "
        f"{row['latitude']:.5f}, "
        f"{row['longitude']:.5f}",
        flush=True,
    )

    print(
        f"2010-01-01 IMERG rainfall: "
        f"{row['imerg_daily_mm']:.4f} mm/day",
        flush=True,
    )


# ============================================================
# CHECKPOINTS
# ============================================================

def checkpoint_path(
    year: int,
):
    return (
        YEARLY_DIR /
        f"imerg_{year}_daily.csv"
    )


def checkpoint_complete(
    path: Path,
    expected,
):
    if not path.exists():
        return False

    try:
        df = pd.read_csv(
            path
        )

        if "timestamp" not in df.columns:
            return False

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        actual = pd.DatetimeIndex(
            df["timestamp"]
            .dropna()
            .drop_duplicates()
        ).sort_values()

        return (
            len(actual)
            == len(expected)
            and len(
                expected.difference(
                    actual
                )
            ) == 0
        )

    except Exception:
        return False


# ============================================================
# BUILD YEAR
# ============================================================

def build_year(
    earthaccess,
    session,
    year: int,
):
    start, end = year_bounds(
        year
    )

    expected = expected_days(
        start,
        end
    )

    checkpoint = checkpoint_path(
        year
    )

    if checkpoint_complete(
        checkpoint,
        expected,
    ):
        print(
            f"\nComplete checkpoint found for {year}: "
            f"{checkpoint}",
            flush=True,
        )

        return pd.read_csv(
            checkpoint,
            parse_dates=[
                "timestamp"
            ],
        )

    banner(
        f"PROCESSING IMERG DAILY {year}"
    )

    granules = search_year(
        earthaccess,
        year
    )

    if not granules:
        raise RuntimeError(
            f"No IMERG V07 Daily granules found for {year}."
        )

    if len(granules) != len(expected):
        print(
            f"WARNING: expected {len(expected)} days "
            f"but NASA returned {len(granules)} granules.",
            flush=True,
        )

    rows = []

    for index, granule in enumerate(
        granules,
        start=1,
    ):
        print(
            f"[{index}/{len(granules)}] "
            f"Reading IMERG day...",
            flush=True,
        )

        row = read_one_granule(
            session=session,
            granule=granule,
        )

        rows.append(
            row
        )

        if index % 25 == 0:
            print(
                f"      {index} days completed.",
                flush=True,
            )

        time.sleep(
            SLEEP_BETWEEN_GRANULES
        )

    if not rows:
        raise RuntimeError(
            f"No IMERG values extracted for {year}."
        )

    df = pd.DataFrame(
        rows
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=[
                "timestamp"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        (df["timestamp"] >= start)
        & (
            df["timestamp"]
            < end
        )
    ].copy()

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        checkpoint,
        index=False,
        float_format="%.6f",
    )

    actual = pd.DatetimeIndex(
        df["timestamp"]
    ).sort_values()

    missing = expected.difference(
        actual
    )

    print(
        f"\nYear {year} complete."
        f"\nRows: {len(df)}"
        f"\nMissing days: {len(missing)}"
        f"\nSaved: {checkpoint}",
        flush=True,
    )

    if len(missing):
        print(
            "Missing examples:",
            list(
                missing[:10]
            ),
            flush=True,
        )

    return df


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final(
    df: pd.DataFrame,
):
    expected = pd.date_range(
        START_DATE,
        END_DATE_EXCLUSIVE
        - pd.Timedelta(
            days=1
        ),
        freq="1D",
        tz="UTC",
    )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    actual = pd.DatetimeIndex(
        df["timestamp"]
        .dropna()
        .drop_duplicates()
    ).sort_values()

    missing = expected.difference(
        actual
    )

    return {
        "expected_days": int(
            len(expected)
        ),
        "actual_days": int(
            len(actual)
        ),
        "missing_days": int(
            len(missing)
        ),
        "duplicates": int(
            df["timestamp"]
            .duplicated()
            .sum()
        ),
        "missing_precipitation": int(
            df[
                "imerg_daily_mm"
            ]
            .isna()
            .sum()
        ),
        "start": str(
            df["timestamp"].min()
        ),
        "end": str(
            df["timestamp"].max()
        ),
        "mean_daily_mm": (
            float(
                df[
                    "imerg_daily_mm"
                ].mean()
            )
            if df[
                "imerg_daily_mm"
            ].notna().any()
            else None
        ),
        "max_daily_mm": (
            float(
                df[
                    "imerg_daily_mm"
                ].max()
            )
            if df[
                "imerg_daily_mm"
            ].notna().any()
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    banner(
        "NASA GPM IMERG FINAL DAILY V07"
    )

    print(
        f"Mumbai latitude : {LATITUDE}"
    )

    print(
        f"Mumbai longitude: {LONGITUDE}"
    )

    print(
        f"\nPeriod: "
        f"{START_DATE} -> "
        f"{END_DATE_EXCLUSIVE} "
        f"(exclusive)"
    )

    print(
        "\nProduct:"
        "\n  GPM_3IMERGDF"
        "\n  Version 07"
        "\n  Final Daily"
        "\n  0.1 degree"
        "\n  NetCDF / Cloud OPeNDAP / DAP4"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    YEARLY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    earthaccess, session = authenticate()

    # Mandatory one-granule test.
    preflight(
        earthaccess=earthaccess,
        session=session,
    )

    all_years = []

    for year in range(
        START_DATE.year,
        END_DATE_EXCLUSIVE.year + 1,
    ):
        year_df = build_year(
            earthaccess=earthaccess,
            session=session,
            year=year,
        )

        if year_df.empty:
            continue

        all_years.append(
            year_df
        )

        cumulative = (
            pd.concat(
                all_years,
                ignore_index=True,
            )
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=[
                    "timestamp"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        cumulative.to_csv(
            FINAL_OUTPUT,
            index=False,
            float_format="%.6f",
        )

        print(
            f"\nCumulative output:"
            f"\n{FINAL_OUTPUT}"
            f"\nRows: {len(cumulative):,}",
            flush=True,
        )

    if not all_years:
        raise RuntimeError(
            "No IMERG satellite data were produced."
        )

    final_df = (
        pd.concat(
            all_years,
            ignore_index=True,
        )
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=[
                "timestamp"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    final_df = final_df[
        (final_df["timestamp"] >= START_DATE)
        & (
            final_df["timestamp"]
            < END_DATE_EXCLUSIVE
        )
    ].copy()

    final_df.to_csv(
        FINAL_OUTPUT,
        index=False,
        float_format="%.6f",
    )

    validation = validate_final(
        final_df
    )

    SUMMARY_OUTPUT.write_text(
        json.dumps(
            {
                "source":
                    "NASA GPM IMERG Final Daily V07",
                "short_name":
                    SHORT_NAME,
                "version":
                    VERSION,
                "concept_id":
                    CONCEPT_ID,
                "latitude":
                    LATITUDE,
                "longitude":
                    LONGITUDE,
                "period_start":
                    START_DATE.isoformat(),
                "period_end_exclusive":
                    END_DATE_EXCLUSIVE.isoformat(),
                "access":
                    "PyDAP DAP4 + earthaccess Auth session",
                "output":
                    str(FINAL_OUTPUT),
                "validation":
                    validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    banner(
        "IMERG BUILD COMPLETE"
    )

    print(
        f"Output: {FINAL_OUTPUT}"
    )

    print(
        f"Rows: {len(final_df):,}"
    )

    print(
        f"Start: {final_df['timestamp'].min()}"
    )

    print(
        f"End: {final_df['timestamp'].max()}"
    )

    print(
        f"Missing days: "
        f"{validation['missing_days']}"
    )

    print(
        f"Missing precipitation: "
        f"{validation['missing_precipitation']}"
    )

    print(
        f"Duplicates: "
        f"{validation['duplicates']}"
    )

    print(
        f"Max daily rainfall: "
        f"{validation['max_daily_mm']} mm/day"
    )

    if (
        validation["missing_days"]
        == 0
    ):
        print(
            "\nFULL 2010-2025 DAILY "
            "SATELLITE COVERAGE CONFIRMED."
        )
    else:
        print(
            "\nWARNING: daily satellite data contain gaps."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )
        sys.exit(130)
    except Exception as exc:
        print(
            "\n" + "=" * 70
        )
        print(
            "BUILD FAILED"
        )
        print(
            "=" * 70
        )
        print(
            str(exc)
        )
        sys.exit(1)
