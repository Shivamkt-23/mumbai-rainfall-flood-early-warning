"""

Live multisource feature builder for the Mumbai rainfall model.



Operational adapter for the frozen 2010-2025 ERA5 + IMERG Random Forest.



Important:



- The frozen model was trained with ERA5 hourly fields plus previous-day IMERG

  Final Daily V07 features.



- Live inference uses NASA IMERG Late Daily V07 because the Final Daily product

  is a research product with roughly 3.5 months of latency and cannot provide

  current-day satellite context.



- The live weather feed is converted to the same feature schema and its

  local timestamps are converted to UTC for the calendar features, matching

  the ERA5 training convention.



- IMERG features use only completed days strictly before the prediction hour.



- The current live deployment is an operational adapter, not a re-validation

  of the historical 2025 holdout metrics.



- IMERG extraction is location-aware. The default location remains the

  original Mumbai model point for backward compatibility.

"""



from __future__ import annotations



import math

import os

import re

import shutil

import tempfile

from pathlib import Path



import numpy as np

import pandas as pd





# ============================================================

# DEFAULT LOCATION / TIMEZONE

# ============================================================



MUMBAI_LAT = 19.05

MUMBAI_LON = 73.05



MODEL_TIMEZONE = "Asia/Kolkata"





# ============================================================

# PATHS

# ============================================================



BASE_DIR = Path(__file__).resolve().parents[1]



LIVE_CACHE_PATH = (

    BASE_DIR

    / "data"

    / "processed"

    / "imerg_live_late_cache.csv"

)



TEMP_DIR = (

    BASE_DIR

    / "data"

    / "tmp_imerg_live"

)



HISTORICAL_FALLBACK_PATH = (

    BASE_DIR

    / "data"

    / "processed"

    / "imerg_2010_2025_daily.csv"

)





# ============================================================

# NASA GPM IMERG LATE DAILY V07

# ============================================================



IMERG_CONCEPT_ID = "C2723754859-GES_DISC"

IMERG_VERSION = "07"

IMERG_SHORT_NAME = "GPM_3IMERGDL"



IMERG_FIELDS = [

    "precipitation_cnt",

    "precipitation_cnt_cond",

    "randomError",

    "randomError_cnt",

    "probabilityLiquidPrecipitation",

]



DATE_RE = re.compile(

    r"3IMERG\.(\d{8})-"

)





# ============================================================

# FROZEN MODEL FEATURE SCHEMA

# ============================================================



BASE_MODEL_FEATURES = [

    "rainfall_mm",

    "temperature_c",

    "dewpoint_c",

    "pressure_hpa",

    "u10_ms",

    "v10_ms",



    "rainfall_lag_1h",

    "rainfall_lag_2h",

    "rainfall_lag_3h",

    "rainfall_lag_6h",

    "rainfall_lag_12h",

    "rainfall_lag_24h",



    "rain_3h",

    "rain_6h",

    "rain_24h",



    "year",

    "month",

    "day",

    "hour",

    "day_of_year",



    "hour_sin",

    "hour_cos",

    "doy_sin",

    "doy_cos",



    "imerg_prev_day_mm",

    "imerg_3day_sum_mm",

    "imerg_7day_sum_mm",

    "imerg_3day_max_mm",

    "imerg_7day_max_mm",



    "imerg_prev_day_random_error",

    "imerg_prev_day_quality_ratio",

    "imerg_prev_day_liquid_probability",

    "imerg_prev_day_count",

    "imerg_prev_day_cond_count",

]





# ============================================================

# GENERIC HELPERS

# ============================================================



def _clean_scalar(value) -> float:

    """

    Convert a scalar to a finite float.



    IMERG fill / invalid values are converted to NaN.

    """



    try:

        value = float(value)

    except (TypeError, ValueError):

        return float("nan")



    if not math.isfinite(value) or value <= -9000:

        return float("nan")



    return value





def _find_dataset(group, name: str):

    """

    Find a dataset by its final path component.



    HDF5 dataset paths can vary, so search recursively.

    """



    result = None



    def visitor(path, obj):

        nonlocal result



        if result is not None:

            return



        try:

            import h5py



            if (

                isinstance(obj, h5py.Dataset)

                and path.rsplit("/", 1)[-1] == name

            ):

                result = obj



        except Exception:

            return



    group.visititems(visitor)



    return result





def _nearest_index(

    arr: np.ndarray,

    value: float,

) -> int:

    """

    Return the nearest coordinate index.

    """



    return int(

        np.abs(

            np.asarray(arr, dtype=float)

            - float(value)

        ).argmin()

    )





def _infer_axes(

    precipitation,

    lat_len: int,

    lon_len: int,

):

    """

    Infer the time, longitude and latitude axes.

    """



    shape = precipitation.shape



    if len(shape) != 3:

        raise RuntimeError(

            f"Unexpected IMERG precipitation shape: {shape}"

        )



    lon_candidates = [

        i

        for i, n in enumerate(shape)

        if n == lon_len

    ]



    lat_candidates = [

        i

        for i, n in enumerate(shape)

        if n == lat_len

    ]



    if not lon_candidates or not lat_candidates:

        raise RuntimeError(

            f"Could not identify IMERG axes: "

            f"shape={shape}, "

            f"lon_len={lon_len}, "

            f"lat_len={lat_len}"

        )



    lon_axis = lon_candidates[0]

    lat_axis = lat_candidates[0]



    time_axis = next(

        i

        for i in range(3)

        if i not in (lon_axis, lat_axis)

    )



    return (

        time_axis,

        lon_axis,

        lat_axis,

    )





# ============================================================

# IMERG FILE EXTRACTION

# ============================================================



def _extract_imerg_file(

    path: Path,

    latitude: float = MUMBAI_LAT,

    longitude: float = MUMBAI_LON,

) -> dict:

    """

    Extract one IMERG Late Daily granule for a requested location.



    Returns:

        - precip_mm_day

        - nearest IMERG grid lat/lon

        - requested monitoring lat/lon

        - IMERG quality fields

        - observation date

    """



    import h5py



    with h5py.File(path, "r") as h5:



        lat_ds = _find_dataset(

            h5,

            "lat",

        )



        lon_ds = _find_dataset(

            h5,

            "lon",

        )



        precip_ds = _find_dataset(

            h5,

            "precipitation",

        )



        if (

            lat_ds is None

            or lon_ds is None

            or precip_ds is None

        ):

            raise RuntimeError(

                f"Required IMERG datasets missing "

                f"in {path.name}"

            )



        lat = np.asarray(

            lat_ds[:],

            dtype=float,

        ).reshape(-1)



        lon = np.asarray(

            lon_ds[:],

            dtype=float,

        ).reshape(-1)



        # ----------------------------------------------------

        # Requested monitoring location

        # -> nearest IMERG grid cell

        # ----------------------------------------------------



        lat_idx = _nearest_index(

            lat,

            latitude,

        )



        lon_idx = _nearest_index(

            lon,

            longitude,

        )



        time_axis, lon_axis, lat_axis = (

            _infer_axes(

                precip_ds,

                lat_len=len(lat),

                lon_len=len(lon),

            )

        )



        idx = [slice(None)] * 3



        idx[time_axis] = 0

        idx[lon_axis] = lon_idx

        idx[lat_axis] = lat_idx



        values = {

            "precip_mm_day": _clean_scalar(

                np.asarray(

                    precip_ds[

                        tuple(idx)

                    ]

                ).reshape(-1)[0]

            ),



            # Actual nearest IMERG grid cell.

            "lat": float(

                lat[lat_idx]

            ),



            "lon": float(

                lon[lon_idx]

            ),



            # Actual requested monitoring location.

            "requested_latitude": float(

                latitude

            ),



            "requested_longitude": float(

                longitude

            ),

        }



        # ----------------------------------------------------

        # IMERG quality fields

        # ----------------------------------------------------



        for field in IMERG_FIELDS:



            ds = _find_dataset(

                h5,

                field,

            )



            if ds is None:

                continue



            try:



                values[field] = _clean_scalar(

                    np.asarray(

                        ds[

                            tuple(idx)

                        ]

                    ).reshape(-1)[0]

                )



            except Exception:



                values[field] = np.nan



    # --------------------------------------------------------

    # Date from official IMERG filename

    # --------------------------------------------------------



    match = DATE_RE.search(

        path.name

    )



    if not match:

        raise RuntimeError(

            f"Could not infer IMERG date "

            f"from {path.name}"

        )



    date_text = match.group(1)



    values["date"] = pd.to_datetime(

        date_text,

        format="%Y%m%d",

    ).normalize()



    return values





# ============================================================

# LOCATION-AWARE CACHE

# ============================================================



def _load_cache(

    latitude: float | None = None,

    longitude: float | None = None,

) -> pd.DataFrame:

    """

    Load location-aware IMERG live cache.



    Older cache files without requested coordinates are treated

    as the original Mumbai point for backward compatibility.

    """



    if not LIVE_CACHE_PATH.exists():

        return pd.DataFrame()



    try:



        df = pd.read_csv(

            LIVE_CACHE_PATH,

            parse_dates=["date"],

        )



    except Exception:



        return pd.DataFrame()



    if (

        df.empty

        or "date" not in df.columns

    ):

        return pd.DataFrame()



    # --------------------------------------------------------

    # Backward compatibility with old cache

    # --------------------------------------------------------



    if "requested_latitude" not in df.columns:

        df["requested_latitude"] = MUMBAI_LAT



    if "requested_longitude" not in df.columns:

        df["requested_longitude"] = MUMBAI_LON



    df["requested_latitude"] = pd.to_numeric(

        df["requested_latitude"],

        errors="coerce",

    )



    df["requested_longitude"] = pd.to_numeric(

        df["requested_longitude"],

        errors="coerce",

    )



    df = (

        df

        .dropna(

            subset=[

                "date",

                "requested_latitude",

                "requested_longitude",

            ]

        )

        .sort_values("date")

        .drop_duplicates(

            subset=[

                "date",

                "requested_latitude",

                "requested_longitude",

            ],

            keep="last",

        )

        .reset_index(drop=True)

    )



    # --------------------------------------------------------

    # Optional location filter

    # --------------------------------------------------------



    if (

        latitude is not None

        and longitude is not None

    ):



        df = df[

            np.isclose(

                df["requested_latitude"],

                float(latitude),

            )

            &

            np.isclose(

                df["requested_longitude"],

                float(longitude),

            )

        ].copy()



    return df





def _save_cache(

    df: pd.DataFrame,

) -> None:

    """

    Merge newly extracted IMERG data into the persistent cache.



    Cache key:

        date + requested latitude + requested longitude

    """



    LIVE_CACHE_PATH.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    if df.empty:

        return



    # Load all existing locations.

    existing = _load_cache()



    if not existing.empty:



        df = pd.concat(

            [

                existing,

                df,

            ],

            ignore_index=True,

        )



    if "requested_latitude" not in df.columns:

        df["requested_latitude"] = MUMBAI_LAT



    if "requested_longitude" not in df.columns:

        df["requested_longitude"] = MUMBAI_LON



    df["requested_latitude"] = pd.to_numeric(

        df["requested_latitude"],

        errors="coerce",

    )



    df["requested_longitude"] = pd.to_numeric(

        df["requested_longitude"],

        errors="coerce",

    )



    df = (

        df

        .dropna(

            subset=[

                "date",

                "requested_latitude",

                "requested_longitude",

            ]

        )

        .sort_values("date")

        .drop_duplicates(

            subset=[

                "date",

                "requested_latitude",

                "requested_longitude",

            ],

            keep="last",

        )

    )



    df.to_csv(

        LIVE_CACHE_PATH,

        index=False,

    )





# ============================================================

# EARTHDATA AUTHENTICATION

# ============================================================



def _normalise_earthdata_token():

    """

    Return a raw Earthdata token without exposing it.

    """



    token = os.environ.get(

        "EARTHDATA_TOKEN",

        "",

    )



    token = (

        token

        .strip()

        .strip('"')

        .strip("'")

    )



    if token.lower().startswith(

        "bearer "

    ):

        token = token[7:].strip()



    if token:

        os.environ[

            "EARTHDATA_TOKEN"

        ] = token



    return token





def _login_earthdata():

    """

    Authenticate using the configured Earthdata token or netrc.

    """



    import earthaccess



    token = _normalise_earthdata_token()



    if token:



        return earthaccess.login(

            strategy="environment"

        )



    return earthaccess.login(

        strategy="netrc"

    )





# ============================================================

# IMERG SEARCH

# ============================================================



def _search_imerg(

    start_date: pd.Timestamp,

    end_date: pd.Timestamp,

):

    """

    Search NASA GES DISC for IMERG Late Daily V07.

    """



    _login_earthdata()



    import earthaccess



    return earthaccess.search_data(

        concept_id=IMERG_CONCEPT_ID,

        version=IMERG_VERSION,

        temporal=(

            start_date.strftime("%Y-%m-%d"),

            end_date.strftime("%Y-%m-%d"),

        ),

        cloud_hosted=False,

    )





# ============================================================

# AUTHENTICATED NASA DOWNLOAD

# ============================================================



def _download_granule_authenticated(

    granule,

    work_dir: Path,

) -> list[Path]:

    """

    Download a GES DISC IMERG granule using an authenticated

    Earthaccess HTTPS session.



    Retries transient NASA/GES DISC failures such as:

        - HTTP 502 Bad Gateway

        - HTTP 503 Service Unavailable

        - HTTP 504 Gateway Timeout

        - temporary connection/time-out errors



    Authentication errors such as HTTP 401 fail immediately.

    """



    import time

    import requests

    import earthaccess



    _login_earthdata()



    work_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    links = []



    try:

        links = granule.data_links(

            access="external"

        )



    except TypeError:

        links = granule.data_links()



    urls = [

        str(url)

        for url in links

        if str(url).lower().endswith(

            (

                ".nc4",

                ".hdf5",

                ".h5",

                ".nc",

            )

        )

    ]



    if not urls:

        urls = [

            str(url)

            for url in links

            if (

                "data.gesdisc.earthdata.nasa.gov"

                in str(url)

            )

        ]



    if not urls:

        raise RuntimeError(

            "No direct GES DISC data URL was "

            "found for the IMERG Late granule."

        )



    session = (

        earthaccess

        .get_requests_https_session()

    )



    paths: list[Path] = []



    # Temporary/transient failures worth retrying.

    retry_statuses = {

        429,  # Too Many Requests

        500,

        502,

        503,

        504,

    }



    max_attempts = 5



    for url in urls:



        filename = (

            url

            .split("/")[-1]

            .split("?")[0]

        )



        if not filename:

            filename = "imerg_live.nc4"



        output = (

            work_dir

            / filename

        )



        # Reuse an already completed download.

        if (

            output.exists()

            and output.stat().st_size > 0

        ):

            print(

                f"[IMERG] Using existing file: {output}",

                flush=True,

            )

            paths.append(output)

            continue



        partial = Path(

            str(output) + ".part"

        )



        # Remove any stale partial file from an

        # earlier interrupted download.

        if partial.exists():

            try:

                partial.unlink()

            except OSError:

                pass



        success = False



        for attempt in range(

            1,

            max_attempts + 1,

        ):



            response = None



            try:



                print(

                    f"[IMERG] Download attempt "

                    f"{attempt}/{max_attempts}: "

                    f"{filename}",

                    flush=True,

                )



                response = session.get(

                    url,

                    stream=True,

                    timeout=180,

                    allow_redirects=True,

                )



                status = response.status_code



                # Authentication failure should not be retried.

                if status == 401:

                    raise RuntimeError(

                        "GES DISC rejected the Earthdata credential "

                        "(HTTP 401). The Earthdata token may be invalid "

                        "or expired, or the GES DISC authorization may "

                        "not be active."

                    )



                # Other 4xx errors are normally not transient.

                if (

                    status >= 400

                    and status not in retry_statuses

                ):

                    response.raise_for_status()



                # Retry transient server/rate-limit errors.

                if status in retry_statuses:



                    if attempt == max_attempts:

                        raise RuntimeError(

                            "GES DISC IMERG download failed after "

                            f"{max_attempts} attempts with HTTP "

                            f"{status}: {url}"

                        )



                    wait_seconds = min(

                        5 * (2 ** (attempt - 1)),

                        60,

                    )



                    print(

                        f"[IMERG] HTTP {status}. "

                        f"Retrying in {wait_seconds}s...",

                        flush=True,

                    )



                    time.sleep(

                        wait_seconds

                    )



                    continue



                # Any remaining unsuccessful response.

                response.raise_for_status()



                # Write to .part first.

                with partial.open(

                    "wb"

                ) as handle:



                    for chunk in response.iter_content(

                        chunk_size=1024 * 1024

                    ):



                        if chunk:

                            handle.write(chunk)



                if (

                    not partial.exists()

                    or partial.stat().st_size == 0

                ):

                    raise RuntimeError(

                        "IMERG download produced an "

                        "empty temporary file."

                    )



                # Atomic-ish replacement after successful download.

                partial.replace(output)



                if (

                    not output.exists()

                    or output.stat().st_size == 0

                ):

                    raise RuntimeError(

                        f"IMERG download produced an empty file: "

                        f"{output}"

                    )



                print(

                    f"[IMERG] Download complete: "

                    f"{output.name} "

                    f"({output.stat().st_size / (1024 * 1024):.1f} MB)",

                    flush=True,

                )



                paths.append(output)

                success = True

                break



            except RuntimeError:

                raise



            except requests.exceptions.RequestException as exc:



                if attempt == max_attempts:

                    raise RuntimeError(

                        "IMERG download failed after "

                        f"{max_attempts} attempts due to "

                        f"network error: {exc}"

                    ) from exc



                wait_seconds = min(

                    5 * (2 ** (attempt - 1)),

                    60,

                )



                print(

                    f"[IMERG] Temporary network error: "

                    f"{exc}. "

                    f"Retrying in {wait_seconds}s...",

                    flush=True,

                )



                time.sleep(

                    wait_seconds

                )



            finally:



                if response is not None:

                    try:

                        response.close()

                    except Exception:

                        pass



        if not success:

            raise RuntimeError(

                f"IMERG download did not complete: {url}"

            )



    return paths

    """

    Download a GES DISC IMERG granule using an authenticated

    Earthaccess HTTPS session.

    """



    import earthaccess



    _login_earthdata()



    work_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    links = []



    try:



        links = granule.data_links(

            access="external"

        )



    except TypeError:



        links = granule.data_links()



    urls = [

        str(url)

        for url in links

        if str(url).lower().endswith(

            (

                ".nc4",

                ".hdf5",

                ".h5",

                ".nc",

            )

        )

    ]



    if not urls:



        urls = [

            str(url)

            for url in links

            if (

                "data.gesdisc.earthdata.nasa.gov"

                in str(url)

            )

        ]



    if not urls:



        raise RuntimeError(

            "No direct GES DISC data URL was "

            "found for the IMERG Late granule."

        )



    session = (

        earthaccess

        .get_requests_https_session()

    )



    paths: list[Path] = []



    for url in urls:



        filename = (

            url

            .split("/")[-1]

            .split("?")[0]

        )



        if not filename:

            filename = "imerg_live.nc4"



        output = (

            work_dir

            / filename

        )



        response = session.get(

            url,

            stream=True,

            timeout=180,

            allow_redirects=True,

        )



        if response.status_code == 401:



            raise RuntimeError(

                "GES DISC rejected the Earthdata credential "

                "(HTTP 401). The Earthdata token may be invalid "

                "or expired, or the GES DISC authorization may "

                "not be active."

            )



        response.raise_for_status()



        with output.open("wb") as handle:



            for chunk in response.iter_content(

                chunk_size=1024 * 1024

            ):



                if chunk:

                    handle.write(chunk)



        if (

            not output.exists()

            or output.stat().st_size == 0

        ):



            raise RuntimeError(

                f"IMERG download produced an empty file: {output}"

            )



        paths.append(output)



    return paths





# ============================================================

# HISTORICAL FALLBACK

# ============================================================



def _load_historical_fallback(

    days: int = 7,

) -> pd.DataFrame:

    """

    Load the latest archived IMERG history.



    This is a demonstration fallback only.

    It does not represent current satellite conditions.

    """



    if not HISTORICAL_FALLBACK_PATH.exists():



        raise RuntimeError(

            "Live IMERG is unavailable and the historical "

            "IMERG fallback file does not exist: "

            f"{HISTORICAL_FALLBACK_PATH}"

        )



    df = pd.read_csv(

        HISTORICAL_FALLBACK_PATH,

        parse_dates=["date"],

    )



    if (

        df.empty

        or "date" not in df.columns

        or "precip_mm_day" not in df.columns

    ):



        raise RuntimeError(

            "Historical IMERG fallback file is empty or malformed."

        )



    df["date"] = (

        pd.to_datetime(

            df["date"],

            errors="coerce",

        )

        .dt

        .normalize()

    )



    df = (

        df

        .dropna(

            subset=["date"]

        )

        .sort_values("date")

        .drop_duplicates("date")

    )



    if len(df) < days:



        raise RuntimeError(

            f"Historical IMERG fallback contains "

            f"only {len(df)} days; need {days}."

        )



    result = (

        df

        .tail(days)

        .copy()

    )



    result.attrs["source"] = (

        "historical_fallback"

    )



    result.attrs["source_note"] = (

        "Latest archived IMERG Final Daily history "

        "used because live IMERG Late Daily download "

        "was unavailable. Demo continuity only."

    )



    return result





# ============================================================

# SINGLE-LOCATION LIVE IMERG

# ============================================================



def fetch_imerg_live_context(

    target_utc: pd.Timestamp,

    history_days: int = 7,

    latitude: float = MUMBAI_LAT,

    longitude: float = MUMBAI_LON,

) -> pd.DataFrame:

    """

    Fetch previous completed IMERG Late Daily V07 days for

    one monitoring location.



    Existing Live Mumbai behavior remains compatible because

    the original Mumbai coordinates are the defaults.

    """




    target_utc = pd.Timestamp(
        target_utc
    )

    if target_utc.tzinfo is not None:
        target_utc = (
            target_utc
            .tz_convert("UTC")
            .tz_localize(None)
        )

    target_utc = target_utc.normalize()
    end_date = (

        target_utc

        - pd.Timedelta(days=1)

    )



    start_date = (

        target_utc

        - pd.Timedelta(days=history_days)

    )



    required_dates = pd.date_range(

        start_date,

        end_date,

        freq="D",

    )



    # --------------------------------------------------------

    # Location-specific cache

    # --------------------------------------------------------




    cache = _load_cache(
        latitude=latitude,
        longitude=longitude,
    )

    if not cache.empty:
        cache["date"] = (
            pd.to_datetime(
                cache["date"],
                errors="coerce",
            )
            .dt.normalize()
        )

        available = (
            cache[
                cache["date"] <= end_date
            ]
            .dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(
                subset=["date"],
                keep="last",
            )
        )
    else:
        available = pd.DataFrame()

    # Use the latest seven available completed live IMERG days.
    # A missing newest calendar day must not force a full historical
    # fallback when enough recent live observations are available.
    if len(available) >= history_days:
        result = (
            available
            .tail(history_days)
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        result.attrs["source"] = "live_late_cache"
        result.attrs["requested_latitude"] = float(latitude)
        result.attrs["requested_longitude"] = float(longitude)

        return result
    # --------------------------------------------------------

    # Earthaccess availability

    # --------------------------------------------------------



    try:



        import earthaccess  # noqa: F401



    except ImportError:



        return _load_historical_fallback(

            history_days

        )



    # --------------------------------------------------------

    # NASA live path

    # --------------------------------------------------------



    try:



        TEMP_DIR.mkdir(

            parents=True,

            exist_ok=True,

        )



        granules = _search_imerg(

            start_date,

            end_date,

        )



        if not granules:



            return _load_historical_fallback(

                history_days

            )



        extracted = []






        for granule in granules:



            work_dir = Path(

                tempfile.mkdtemp(

                    prefix="imerg_live_",

                    dir=TEMP_DIR,

                )

            )



            try:



                files = (

                    _download_granule_authenticated(

                        granule,

                        work_dir,

                    )

                )



                for file_path in files:



                    path = Path(

                        file_path

                    )



                    if not path.exists():

                        continue



                    try:



                        extracted.append(

                            _extract_imerg_file(

                                path,

                                latitude=latitude,

                                longitude=longitude,

                            )

                        )



                    except Exception:



                        continue



            finally:



                shutil.rmtree(

                    work_dir,

                    ignore_errors=True,

                )



    # ----------------------------------------------------

        # Save newly downloaded observations

        # ----------------------------------------------------



        if extracted:



            fresh = pd.DataFrame(

                extracted

            )



            _save_cache(

                fresh

            )



        # ----------------------------------------------------

        # Read this location back

        # ----------------------------------------------------



        final = _load_cache(

            latitude=latitude,

            longitude=longitude,

        )




        if not final.empty:
            final["date"] = (
                pd.to_datetime(
                    final["date"],
                    errors="coerce",
                )
                .dt.normalize()
            )

            available_final = (
                final[
                    final["date"] <= end_date
                ]
                .dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(
                    subset=["date"],
                    keep="last",
                )
            )
        else:
            available_final = pd.DataFrame()

        if len(available_final) >= history_days:
            result = (
                available_final
                .tail(history_days)
                .copy()
                .sort_values("date")
                .reset_index(drop=True)
            )

            result.attrs["source"] = "live_late"
            result.attrs["requested_latitude"] = float(latitude)
            result.attrs["requested_longitude"] = float(longitude)

            return result

        return _load_historical_fallback(
            history_days
        )
    except Exception:



        return _load_historical_fallback(

            history_days

        )





# ============================================================

# MULTI-LOCATION LIVE IMERG

# ============================================================



def fetch_imerg_live_contexts(

    target_utc: pd.Timestamp,

    locations: list[dict],

    history_days: int = 7,

) -> dict[str, pd.DataFrame]:

    """

    Fetch location-specific IMERG Late Daily V07 context

    for multiple monitoring locations.



    The important optimization is:



        9 locations × 7 downloads

        is NOT performed.



    Instead:



        7 NASA granules

            ↓

        extract all requested locations from each file



    Returns:

        {

            "Borivali": DataFrame,

            "Andheri": DataFrame,

            ...

        }

    """



    target_utc = pd.Timestamp(target_utc)



# IMERG daily observations are treated as date-only values.

# Convert any timezone-aware input to UTC, then remove the

# timezone so date comparisons remain consistent.

    if target_utc.tzinfo is not None:

        target_utc = (

        target_utc

        .tz_convert("UTC")

        .tz_localize(None)

    )

    target_utc = target_utc.normalize()



    end_date = (

        target_utc

        - pd.Timedelta(days=1)

    )



    start_date = (

        target_utc

        - pd.Timedelta(days=history_days)

    )



    required_dates = pd.date_range(

        start_date,

        end_date,

        freq="D",

    )



    # ========================================================

    # NORMALIZE LOCATION INPUT

    # ========================================================



    normalized_locations = []



    seen_names = set()



    for location in locations:



        name = str(

            location.get(

                "name",

                "Unknown",

            )

        )



        latitude = float(

            location["latitude"]

        )



        longitude = float(

            location["longitude"]

        )



        if name in seen_names:

            continue



        seen_names.add(name)



        normalized_locations.append(

            {

                "name": name,

                "latitude": latitude,

                "longitude": longitude,

            }

        )



    if not normalized_locations:



        raise ValueError(

            "No valid locations were supplied for IMERG context."

        )



    # ========================================================

    # CHECK EXISTING CACHE

    # ========================================================



    results: dict[str, pd.DataFrame] = {}



    incomplete_locations = []



    for location in normalized_locations:



        name = location["name"]



        latitude = location["latitude"]

        longitude = location["longitude"]



        cached = _load_cache(

            latitude=latitude,

            longitude=longitude,

        )




        if not cached.empty:
            cached["date"] = (
                pd.to_datetime(
                    cached["date"],
                    errors="coerce",
                )
                .dt.normalize()
            )

            available = (
                cached[
                    cached["date"] <= end_date
                ]
                .dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(
                    subset=["date"],
                    keep="last",
                )
            )
        else:
            available = pd.DataFrame()

        # Use the latest seven available completed live IMERG days.
        # Do not require every calendar date in the target window.
        if len(available) >= history_days:
            result = (
                available
                .tail(history_days)
                .copy()
                .sort_values("date")
                .reset_index(drop=True)
            )

            result.attrs["source"] = "live_late_cache"
            result.attrs["requested_latitude"] = latitude
            result.attrs["requested_longitude"] = longitude

            results[name] = result
        else:
            incomplete_locations.append(
                location
            )
    # ========================================================

    # EVERYTHING IS CACHED

    # ========================================================



    if not incomplete_locations:



        return results



    # ========================================================

    # EARTHACCESS CHECK

    # ========================================================



    try:



        import earthaccess  # noqa: F401



    except ImportError:



        for location in incomplete_locations:



            fallback = (

                _load_historical_fallback(

                    history_days

                )

                .copy()

            )



            fallback[

                "requested_latitude"

            ] = float(

                location["latitude"]

            )



            fallback[

                "requested_longitude"

            ] = float(

                location["longitude"]

            )



            fallback.attrs[

                "source"

            ] = "historical_fallback"



            results[

                location["name"]

            ] = fallback



        return results



    # ========================================================

    # DOWNLOAD NASA GRANULES ONCE

    # ========================================================



    fresh_rows = []






    TEMP_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    granules = _search_imerg(

        start_date,

        end_date,

    )



    if not granules:



        raise RuntimeError(

            "No IMERG Late Daily granules were returned."

        )



    for granule in granules:



        work_dir = Path(

            tempfile.mkdtemp(

                prefix="imerg_multi_",

                dir=TEMP_DIR,

            )

        )



        try:



            files = (

                _download_granule_authenticated(

                    granule,

                    work_dir,

                )

            )



            for file_path in files:



                path = Path(

                    file_path

                )



                if not path.exists():

                    continue



                # ----------------------------------------

                # SAME NASA FILE

                # -> ALL REQUESTED LOCATIONS

                # ----------------------------------------



                for location in incomplete_locations:



                    try:



                        row = (

                            _extract_imerg_file(

                                path,

                                latitude=float(

                                    location[

                                        "latitude"

                                    ]

                                ),

                                longitude=float(

                                    location[

                                        "longitude"

                                    ]

                                ),

                            )

                        )



                        row[

                            "location"

                        ] = location[

                            "name"

                        ]



                        fresh_rows.append(

                            row

                        )



                    except Exception as exc:

                        print(

                    f"[IMERG] EXTRACTION FAILED for "

                    f"{location['name']}: {exc}",

                    flush=True,

                )

                        continue



        finally:



            shutil.rmtree(

                work_dir,

                ignore_errors=True,

            )



# ========================================================

    # SAVE ALL LOCATION-SPECIFIC OBSERVATIONS

    # ========================================================



    if fresh_rows:



        fresh = pd.DataFrame(

            fresh_rows

        )



        _save_cache(

            fresh

        )



    # ========================================================

    # BUILD FINAL RESULT FOR EACH LOCATION

    # ========================================================



    for location in incomplete_locations:



        name = location["name"]



        latitude = location["latitude"]

        longitude = location["longitude"]



        final = _load_cache(

            latitude=latitude,

            longitude=longitude,

        )




        if not final.empty:
            final["date"] = (
                pd.to_datetime(
                    final["date"],
                    errors="coerce",
                )
                .dt.normalize()
            )

            available_final = (
                final[
                    final["date"] <= end_date
                ]
                .dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(
                    subset=["date"],
                    keep="last",
                )
            )
        else:
            available_final = pd.DataFrame()

        if len(available_final) >= history_days:
            result = (
                available_final
                .tail(history_days)
                .copy()
                .sort_values("date")
                .reset_index(drop=True)
            )

            result.attrs["source"] = "live_late"
            result.attrs["requested_latitude"] = latitude
            result.attrs["requested_longitude"] = longitude

            results[name] = result

        else:
            fallback = (
                _load_historical_fallback(
                    history_days
                )
                .copy()
            )

            fallback[
                "requested_latitude"
            ] = latitude

            fallback[
                "requested_longitude"
            ] = longitude

            fallback.attrs[
                "source"
            ] = "historical_fallback"

            fallback.attrs[
                "requested_latitude"
            ] = latitude

            fallback.attrs[
                "requested_longitude"
            ] = longitude

            results[name] = fallback
    return results





# ============================================================

# DEWPOINT

# ============================================================



def _dewpoint_from_temperature_humidity(

    temp_c: pd.Series,

    rh_pct: pd.Series,

) -> pd.Series:

    """

    Estimate dew point from temperature and relative humidity.

    """



    rh = rh_pct.clip(

        lower=1.0,

        upper=100.0,

    )



    gamma = (

        np.log(rh / 100.0)

        + (

            17.625 * temp_c

        )

        / (

            243.04 + temp_c

        )

    )



    return (

        243.04 * gamma

        / (

            17.625 - gamma

        )

    )





# ============================================================

# DATETIME NORMALIZATION

# ============================================================



def _ensure_datetime_utc(

    series: pd.Series,

) -> pd.Series:

    """

    Convert live weather timestamps to naive UTC timestamps.

    """



    values = pd.to_datetime(

        series,

        errors="coerce",

    )



    if values.isna().any():



        raise ValueError(

            "Live weather data contains invalid datetime values."

        )



    if values.dt.tz is None:



        values = (

            values

            .dt

            .tz_localize(

                MODEL_TIMEZONE

            )

            .dt

            .tz_convert("UTC")

        )



    else:



        values = (

            values

            .dt

            .tz_convert("UTC")

        )



    return (

        values

        .dt

        .tz_localize(None)

    )





# ============================================================

# BUILD LIVE MULTISOURCE FEATURES

# ============================================================



def build_live_multisource_features(

    hourly_live: pd.DataFrame,

    imerg_daily: pd.DataFrame,

    feature_columns: list[str],

) -> tuple[

    pd.DataFrame,

    pd.Series,

]:

    """

    Build the exact frozen-model feature schema for the

    latest live hour.

    """



    required_live = [

        "datetime",

        "rainfall_mm",

        "temperature_c",

        "pressure_hpa",

        "u10_ms",

        "v10_ms",

    ]



    missing = [

        c

        for c in required_live

        if c not in hourly_live.columns

    ]



    if missing:



        raise ValueError(

            "Live weather feed is missing: "

            + ", ".join(missing)

        )



    live = (

        hourly_live.copy()

    )



    original_datetime = pd.to_datetime(

        live["datetime"],

        errors="coerce",

    )



    live["_model_datetime_utc"] = (

        _ensure_datetime_utc(

            live["datetime"]

        )

    )



    live = (

        live

        .sort_values(

            "_model_datetime_utc"

        )

        .drop_duplicates(

            "_model_datetime_utc"

        )

        .reset_index(drop=True)

    )



    # Preserve original datetime parsing behavior.

    _ = original_datetime



    # --------------------------------------------------------

    # Minimum hourly history

    # --------------------------------------------------------



    if len(live) < 25:



        raise ValueError(

            "Need at least 25 hourly observations "

            "to construct the 24-hour lag features; "

            f"received {len(live)}."

        )



    # --------------------------------------------------------

    # Dewpoint

    # --------------------------------------------------------



    if "dewpoint_c" not in live.columns:



        if "humidity_pct" not in live.columns:



            raise ValueError(

                "Live weather feed needs dewpoint_c "

                "or humidity_pct."

            )



        live["dewpoint_c"] = (

            _dewpoint_from_temperature_humidity(

                pd.to_numeric(

                    live["temperature_c"],

                    errors="coerce",

                ),

                pd.to_numeric(

                    live["humidity_pct"],

                    errors="coerce",

                ),

            )

        )



    # --------------------------------------------------------

    # Numeric conversion

    # --------------------------------------------------------



    numeric_columns = [

        "rainfall_mm",

        "temperature_c",

        "dewpoint_c",

        "pressure_hpa",

        "u10_ms",

        "v10_ms",

    ]



    for column in numeric_columns:



        live[column] = pd.to_numeric(

            live[column],

            errors="coerce",

        )



    if (

        live[

            numeric_columns

        ]

        .iloc[-1]

        .isna()

        .any()

    ):



        bad = (

            live[

                numeric_columns

            ]

            .iloc[-1]

            .loc[

                lambda s: s.isna()

            ]

            .index

            .tolist()

        )



        raise ValueError(

            "Latest live weather row has missing values: "

            + ", ".join(bad)

        )



    # --------------------------------------------------------

    # Rainfall lags

    # --------------------------------------------------------



    for lag in [

        1,

        2,

        3,

        6,

        12,

        24,

    ]:



        live[

            f"rainfall_lag_{lag}h"

        ] = (

            live[

                "rainfall_mm"

            ]

            .shift(lag)

        )



    # --------------------------------------------------------

    # Rolling rainfall

    # --------------------------------------------------------



    live["rain_3h"] = (

        live[

            "rainfall_mm"

        ]

        .rolling(

            3,

            min_periods=3,

        )

        .sum()

    )



    live["rain_6h"] = (

        live[

            "rainfall_mm"

        ]

        .rolling(

            6,

            min_periods=6,

        )

        .sum()

    )



    live["rain_24h"] = (

        live[

            "rainfall_mm"

        ]

        .rolling(

            24,

            min_periods=24,

        )

        .sum()

    )



    # --------------------------------------------------------

    # Time features

    # --------------------------------------------------------



    model_time = pd.Timestamp(

        live.iloc[-1][

            "_model_datetime_utc"

        ]

    )



    live["year"] = (

        live[

            "_model_datetime_utc"

        ]

        .dt

        .year

    )



    live["month"] = (

        live[

            "_model_datetime_utc"

        ]

        .dt

        .month

    )



    live["day"] = (

        live[

            "_model_datetime_utc"

        ]

        .dt

        .day

    )



    live["hour"] = (

        live[

            "_model_datetime_utc"

        ]

        .dt

        .hour

    )



    live["day_of_year"] = (

        live[

            "_model_datetime_utc"

        ]

        .dt

        .dayofyear

    )



    live["hour_sin"] = np.sin(

        2

        * np.pi

        * live["hour"]

        / 24.0

    )



    live["hour_cos"] = np.cos(

        2

        * np.pi

        * live["hour"]

        / 24.0

    )



    live["doy_sin"] = np.sin(

        2

        * np.pi

        * live["day_of_year"]

        / 365.25

    )



    live["doy_cos"] = np.cos(

        2

        * np.pi

        * live["day_of_year"]

        / 365.25

    )



    # --------------------------------------------------------

    # IMERG context

    # --------------------------------------------------------



    sat = (

        imerg_daily.copy()

    )



    sat["date"] = (

        pd.to_datetime(

            sat["date"],

            errors="coerce",

        )

        .dt

        .normalize()

    )



    sat = (

        sat

        .dropna(

            subset=["date"]

        )

        .sort_values("date")

        .drop_duplicates("date")

        .reset_index(drop=True)

    )



    needed = pd.date_range(

        model_time.normalize()

        - pd.Timedelta(days=7),



        model_time.normalize()

        - pd.Timedelta(days=1),



        freq="D",

    )



    sat_index = (

        sat

        .set_index("date")

    )



    # --------------------------------------------------------

    # Use the exact seven completed days when available.

    # --------------------------------------------------------



    if all(

        d in sat_index.index

        for d in needed

    ):



        prior = (

            sat_index

            .loc[needed]

            .copy()

        )



    else:



        if len(sat_index) < 7:



            missing_sat = [

                d

                for d in needed

                if d not in sat_index.index

            ]



            raise ValueError(

                "IMERG context has fewer than seven completed "

                "available observations."

                + ", ".join(

                    d.strftime("%Y-%m-%d")

                    for d in missing_sat

                )

            )



        prior = (

            sat

            .sort_values("date")

            .tail(7)

            .set_index("date")

            .copy()

        )



    prior["precip_mm_day"] = pd.to_numeric(

        prior["precip_mm_day"],

        errors="coerce",

    )



    if (

        prior[

            "precip_mm_day"

        ]

        .isna()

        .any()

    ):



        raise ValueError(

            "IMERG precipitation contains missing "

            "values in the required history."

        )



    # --------------------------------------------------------

    # Quality ratio

    # --------------------------------------------------------



    cnt = pd.to_numeric(

        prior.get(

            "precipitation_cnt"

        ),

        errors="coerce",

    ).replace(

        0,

        np.nan,

    )



    cond = pd.to_numeric(

        prior.get(

            "precipitation_cnt_cond"

        ),

        errors="coerce",

    )



    prior["quality_ratio"] = (

        cond

        / cnt

    )



    # --------------------------------------------------------

    # Previous-day IMERG features

    # --------------------------------------------------------



    prev = prior.iloc[-1]



    live.loc[

        live.index[-1],

        "imerg_prev_day_mm",

    ] = float(

        prev[

            "precip_mm_day"

        ]

    )



    live.loc[

        live.index[-1],

        "imerg_3day_sum_mm",

    ] = float(

        prior[

            "precip_mm_day"

        ]

        .iloc[-3:]

        .sum()

    )



    live.loc[

        live.index[-1],

        "imerg_7day_sum_mm",

    ] = float(

        prior[

            "precip_mm_day"

        ]

        .sum()

    )



    live.loc[

        live.index[-1],

        "imerg_3day_max_mm",

    ] = float(

        prior[

            "precip_mm_day"

        ]

        .iloc[-3:]

        .max()

    )



    live.loc[

        live.index[-1],

        "imerg_7day_max_mm",

    ] = float(

        prior[

            "precip_mm_day"

        ]

        .max()

    )



    live.loc[

        live.index[-1],

        "imerg_prev_day_random_error",

    ] = _clean_scalar(

        prev.get(

            "randomError"

        )

    )



    live.loc[

        live.index[-1],

        "imerg_prev_day_quality_ratio",

    ] = _clean_scalar(

        prev.get(

            "quality_ratio"

        )

    )



    live.loc[

        live.index[-1],

        "imerg_prev_day_liquid_probability",

    ] = _clean_scalar(

        prev.get(

            "probabilityLiquidPrecipitation"

        )

    )



    live.loc[

        live.index[-1],

        "imerg_prev_day_count",

    ] = _clean_scalar(

        prev.get(

            "precipitation_cnt"

        )

    )



    live.loc[

        live.index[-1],

        "imerg_prev_day_cond_count",

    ] = _clean_scalar(

        prev.get(

            "precipitation_cnt_cond"

        )

    )



    # --------------------------------------------------------

    # Build model row

    # --------------------------------------------------------



    latest_model_row = (

        live.iloc[-1]

        .copy()

    )



    X_live = pd.DataFrame(

        [

            latest_model_row.to_dict()

        ]

    )



    missing_features = [

        c

        for c in feature_columns

        if c not in X_live.columns

    ]



    if missing_features:



        raise ValueError(

            "Could not construct frozen model features: "

            + ", ".join(

                missing_features

            )

        )



    X_live = (

        X_live[

            feature_columns

        ]

        .apply(

            pd.to_numeric,

            errors="coerce",

        )

    )



    if (

        X_live

        .isna()

        .any()

        .any()

    ):



        bad = (

            X_live.columns[

                X_live.iloc[0].isna()

            ]

            .tolist()

        )



        raise ValueError(

            "Frozen model input still contains "

            "NaN values: "

            + ", ".join(bad)

        )



    # --------------------------------------------------------

    # Rebuild local/UI row

    # --------------------------------------------------------



    local_row = (

        hourly_live

        .iloc[-1]

        .copy()

    )



    local_row[

        "rainfall_lag_1h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_1h"

        ]

    )



    local_row[

        "rainfall_lag_2h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_2h"

        ]

    )



    local_row[

        "rainfall_lag_3h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_3h"

        ]

    )



    local_row[

        "rainfall_lag_6h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_6h"

        ]

    )



    local_row[

        "rainfall_lag_12h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_12h"

        ]

    )



    local_row[

        "rainfall_lag_24h"

    ] = float(

        live.iloc[-1][

            "rainfall_lag_24h"

        ]

    )



    local_row[

        "rain_3h"

    ] = float(

        live.iloc[-1][

            "rain_3h"

        ]

    )



    local_row[

        "rain_6h"

    ] = float(

        live.iloc[-1][

            "rain_6h"

        ]

    )



    local_row[

        "rain_24h"

    ] = float(

        live.iloc[-1][

            "rain_24h"

        ]

    )



    return (

        X_live,

        local_row,

    )
