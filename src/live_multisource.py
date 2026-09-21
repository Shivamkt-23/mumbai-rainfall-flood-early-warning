"""
Live multisource feature builder for the Mumbai rainfall model.

Operational adapter for the frozen 2010-2025 ERA5 + IMERG Random Forest.

Important:
- The frozen model was trained with ERA5 hourly fields plus previous-day IMERG
  Final Daily V07 features.
- Live inference uses NASA IMERG Late Daily V07 because the Final Daily product
  is a research product with roughly 3.5 months of latency and cannot provide
  current-day satellite context.
- The live weather feed is converted to the same *feature schema* and its
  local timestamps are converted to UTC for the calendar features, matching
  the ERA5 training convention.
- IMERG features use only completed days strictly before the prediction hour.
- Current live deployment is an operational adapter, not a re-validation of
  the historical 2025 holdout metrics.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MUMBAI_LAT = 19.05
MUMBAI_LON = 73.05
MODEL_TIMEZONE = "Asia/Kolkata"

BASE_DIR = Path(__file__).resolve().parents[1]
LIVE_CACHE_PATH = BASE_DIR / "data" / "processed" / "imerg_live_late_cache.csv"
TEMP_DIR = BASE_DIR / "data" / "tmp_imerg_live"
HISTORICAL_FALLBACK_PATH = BASE_DIR / "data" / "processed" / "imerg_2010_2025_daily.csv"

# NASA GPM IMERG Late Daily V07 (same daily feature semantics, near-real-time latency).
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

DATE_RE = re.compile(r"3IMERG\.(\d{8})-")

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


def _clean_scalar(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value) or value <= -9000:
        return float("nan")
    return value


def _find_dataset(group, name: str):
    result = None

    def visitor(path, obj):
        nonlocal result
        if result is not None:
            return
        try:
            import h5py  # lazy import
            if isinstance(obj, h5py.Dataset) and path.rsplit("/", 1)[-1] == name:
                result = obj
        except Exception:
            return

    group.visititems(visitor)
    return result


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(np.asarray(arr, dtype=float) - value).argmin())


def _infer_axes(precipitation, lat_len: int, lon_len: int):
    shape = precipitation.shape
    if len(shape) != 3:
        raise RuntimeError(f"Unexpected IMERG precipitation shape: {shape}")

    lon_candidates = [i for i, n in enumerate(shape) if n == lon_len]
    lat_candidates = [i for i, n in enumerate(shape) if n == lat_len]
    if not lon_candidates or not lat_candidates:
        raise RuntimeError(
            f"Could not identify IMERG axes: shape={shape}, "
            f"lon_len={lon_len}, lat_len={lat_len}"
        )

    lon_axis = lon_candidates[0]
    lat_axis = lat_candidates[0]
    time_axis = next(i for i in range(3) if i not in (lon_axis, lat_axis))
    return time_axis, lon_axis, lat_axis


def _extract_imerg_file(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as h5:
        lat_ds = _find_dataset(h5, "lat")
        lon_ds = _find_dataset(h5, "lon")
        precip_ds = _find_dataset(h5, "precipitation")
        if lat_ds is None or lon_ds is None or precip_ds is None:
            raise RuntimeError(f"Required IMERG datasets missing in {path.name}")

        lat = np.asarray(lat_ds[:], dtype=float).reshape(-1)
        lon = np.asarray(lon_ds[:], dtype=float).reshape(-1)
        lat_idx = _nearest_index(lat, MUMBAI_LAT)
        lon_idx = _nearest_index(lon, MUMBAI_LON)

        time_axis, lon_axis, lat_axis = _infer_axes(
            precip_ds,
            lat_len=len(lat),
            lon_len=len(lon),
        )

        idx = [slice(None)] * 3
        idx[time_axis] = 0
        idx[lon_axis] = lon_idx
        idx[lat_axis] = lat_idx

        values = {
            "precip_mm_day": _clean_scalar(np.asarray(precip_ds[tuple(idx)]).reshape(-1)[0]),
            "lat": float(lat[lat_idx]),
            "lon": float(lon[lon_idx]),
        }

        for field in IMERG_FIELDS:
            ds = _find_dataset(h5, field)
            if ds is None:
                continue
            try:
                values[field] = _clean_scalar(np.asarray(ds[tuple(idx)]).reshape(-1)[0])
            except Exception:
                values[field] = np.nan

    match = DATE_RE.search(path.name)
    if not match:
        raise RuntimeError(f"Could not infer IMERG date from {path.name}")
    date_text = match.group(1)
    values["date"] = pd.to_datetime(date_text, format="%Y%m%d").normalize()
    return values


def _load_cache() -> pd.DataFrame:
    if not LIVE_CACHE_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_CACHE_PATH, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _save_cache(df: pd.DataFrame) -> None:
    LIVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").drop_duplicates("date").to_csv(LIVE_CACHE_PATH, index=False)


def _normalise_earthdata_token():
    """Return a raw Earthdata token without printing or exposing its value."""
    import os

    token = os.environ.get("EARTHDATA_TOKEN", "")
    token = token.strip().strip('"').strip("'")
    # Users sometimes paste the whole header value ("Bearer <token>")
    # into EARTHDATA_TOKEN. earthaccess expects the raw token only.
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        os.environ["EARTHDATA_TOKEN"] = token
    return token


def _login_earthdata():
    import os
    import earthaccess

    token = _normalise_earthdata_token()
    if token:
        return earthaccess.login(strategy="environment")

    # If there is no environment token, use an existing Windows _netrc/.netrc.
    # This keeps Streamlit non-interactive.
    return earthaccess.login(strategy="netrc")


def _search_imerg(start_date: pd.Timestamp, end_date: pd.Timestamp):
    _login_earthdata()
    import earthaccess

    return earthaccess.search_data(
        concept_id=IMERG_CONCEPT_ID,
        version=IMERG_VERSION,
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        cloud_hosted=False,
    )


def _download_granule_authenticated(granule, work_dir: Path) -> list[Path]:
    """Download a GES DISC granule with an authenticated HTTPS session.

    earthaccess documents get_requests_https_session() as the supported way to
    access restricted HTTPS data with an Earthdata bearer token. We use the
    granule's external GET-DATA URL and stream it ourselves, avoiding the
    redirect/auth handling that was producing HTTP 401 inside earthaccess.download.
    """
    import earthaccess

    _login_earthdata()
    work_dir.mkdir(parents=True, exist_ok=True)

    links = []
    try:
        links = granule.data_links(access="external")
    except TypeError:
        links = granule.data_links()

    urls = [
        str(url)
        for url in links
        if str(url).lower().endswith((".nc4", ".hdf5", ".h5", ".nc"))
    ]
    if not urls:
        urls = [str(url) for url in links if "data.gesdisc.earthdata.nasa.gov" in str(url)]
    if not urls:
        raise RuntimeError("No direct GES DISC data URL was found for the IMERG Late granule.")

    session = earthaccess.get_requests_https_session()
    paths: list[Path] = []

    for url in urls:
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "imerg_live.nc4"
        output = work_dir / filename

        response = session.get(url, stream=True, timeout=180, allow_redirects=True)
        if response.status_code == 401:
            raise RuntimeError(
                "GES DISC rejected the Earthdata credential (HTTP 401). "
                "The live downloader is now using earthaccess's authenticated HTTPS session. "
                "If this persists, the Earthdata token itself is invalid/expired or the GES DISC "
                "application authorization is not active for this account."
            )
        response.raise_for_status()

        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"IMERG download produced an empty file: {output}")
        paths.append(output)

    return paths


def _load_historical_fallback(days: int = 7) -> pd.DataFrame:
    """Load the latest archived IMERG history for demo continuity.

    This is a presentation/demo fallback only. It does not represent current
    satellite conditions when the live GES DISC download is unavailable.
    """
    if not HISTORICAL_FALLBACK_PATH.exists():
        raise RuntimeError(
            "Live IMERG is unavailable and the historical IMERG fallback file "
            f"does not exist: {HISTORICAL_FALLBACK_PATH}"
        )

    df = pd.read_csv(HISTORICAL_FALLBACK_PATH, parse_dates=["date"])
    if df.empty or "date" not in df.columns or "precip_mm_day" not in df.columns:
        raise RuntimeError("Historical IMERG fallback file is empty or malformed.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    if len(df) < days:
        raise RuntimeError(
            f"Historical IMERG fallback contains only {len(df)} days; need {days}."
        )

    result = df.tail(days).copy()
    result.attrs["source"] = "historical_fallback"
    result.attrs["source_note"] = (
        "Latest archived IMERG Final Daily history used because live IMERG Late "
        "Daily download was unavailable. This is demo continuity only."
    )
    return result


def fetch_imerg_live_context(target_utc: pd.Timestamp, history_days: int = 7) -> pd.DataFrame:
    """Fetch previous completed IMERG Late Daily V07 days for live inference.

    When GES DISC authentication/download is unavailable, fall back to the
    latest archived historical IMERG context so the dashboard remains usable
    for demonstrations. The returned attrs identify the fallback.
    """
    target_utc = pd.Timestamp(target_utc).normalize()
    end_date = target_utc - pd.Timedelta(days=1)
    start_date = target_utc - pd.Timedelta(days=history_days)
    required_dates = pd.date_range(start_date, end_date, freq="D")

    cache = _load_cache()
    if not cache.empty:
        cache_dates = set(pd.to_datetime(cache["date"]).dt.normalize())
        if all(d in cache_dates for d in required_dates):
            cache = cache.loc[cache["date"].isin(required_dates)].copy().sort_values("date")
            cache.attrs["source"] = "live_late_cache"
            return cache

    try:
        import earthaccess
    except ImportError:
        return _load_historical_fallback(history_days)

    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        granules = _search_imerg(start_date, end_date)
        if not granules:
            return _load_historical_fallback(history_days)

        extracted = []
        try:
            for granule in granules:
                work_dir = Path(tempfile.mkdtemp(prefix="imerg_live_", dir=TEMP_DIR))
                try:
                    files = _download_granule_authenticated(granule, work_dir)
                    for file_path in files:
                        path = Path(file_path)
                        if path.exists():
                            try:
                                extracted.append(_extract_imerg_file(path))
                            except Exception:
                                continue
                finally:
                    shutil.rmtree(work_dir, ignore_errors=True)
        finally:
            for child in list(TEMP_DIR.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)

        if extracted:
            fresh = pd.DataFrame(extracted)
            cache = pd.concat([cache, fresh], ignore_index=True) if not cache.empty else fresh
            cache = cache.drop_duplicates("date", keep="last").sort_values("date")
            _save_cache(cache)

        final = _load_cache()
        final_dates = (
            set(pd.to_datetime(final["date"]).dt.normalize())
            if not final.empty else set()
        )
        missing = [d for d in required_dates if d not in final_dates]
        if not missing:
            result = final.loc[final["date"].isin(required_dates)].copy().sort_values("date")
            result.attrs["source"] = "live_late"
            return result

        return _load_historical_fallback(history_days)

    except Exception:
        # Presentation-safe continuity: use archived IMERG rather than crashing
        # the entire Live Mumbai page when GES DISC authentication is unavailable.
        return _load_historical_fallback(history_days)

def _dewpoint_from_temperature_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    rh = rh_pct.clip(lower=1.0, upper=100.0)
    gamma = np.log(rh / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    return 243.04 * gamma / (17.625 - gamma)


def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        raise ValueError("Live weather data contains invalid datetime values.")

    if values.dt.tz is None:
        values = values.dt.tz_localize(MODEL_TIMEZONE).dt.tz_convert("UTC")
    else:
        values = values.dt.tz_convert("UTC")

    return values.dt.tz_localize(None)


def build_live_multisource_features(
    hourly_live: pd.DataFrame,
    imerg_daily: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the exact frozen-model feature schema for the latest live hour.

    Returns:
        X_live: one-row DataFrame in the frozen feature order.
        latest_local_row: latest weather row in the original/local time basis,
                          useful for the UI and risk engine.
    """
    required_live = [
        "datetime",
        "rainfall_mm",
        "temperature_c",
        "pressure_hpa",
        "u10_ms",
        "v10_ms",
    ]
    missing = [c for c in required_live if c not in hourly_live.columns]
    if missing:
        raise ValueError("Live weather feed is missing: " + ", ".join(missing))

    live = hourly_live.copy()
    original_datetime = pd.to_datetime(live["datetime"], errors="coerce")
    live["_model_datetime_utc"] = _ensure_datetime_utc(live["datetime"])
    live = live.sort_values("_model_datetime_utc").drop_duplicates("_model_datetime_utc").reset_index(drop=True)

    if len(live) < 25:
        raise ValueError(
            f"Need at least 25 hourly observations to construct the 24-hour lag features; "
            f"received {len(live)}."
        )

    if "dewpoint_c" not in live.columns:
        if "humidity_pct" not in live.columns:
            raise ValueError("Live weather feed needs dewpoint_c or humidity_pct.")
        live["dewpoint_c"] = _dewpoint_from_temperature_humidity(
            pd.to_numeric(live["temperature_c"], errors="coerce"),
            pd.to_numeric(live["humidity_pct"], errors="coerce"),
        )

    numeric_columns = [
        "rainfall_mm",
        "temperature_c",
        "dewpoint_c",
        "pressure_hpa",
        "u10_ms",
        "v10_ms",
    ]
    for column in numeric_columns:
        live[column] = pd.to_numeric(live[column], errors="coerce")

    if live[numeric_columns].iloc[-1].isna().any():
        bad = live[numeric_columns].iloc[-1][lambda s: s.isna()].index.tolist()
        raise ValueError("Latest live weather row has missing values: " + ", ".join(bad))

    for lag in [1, 2, 3, 6, 12, 24]:
        live[f"rainfall_lag_{lag}h"] = live["rainfall_mm"].shift(lag)

    # Rain totals use observed/current rainfall plus the immediately preceding hours.
    live["rain_3h"] = live["rainfall_mm"].rolling(3, min_periods=3).sum()
    live["rain_6h"] = live["rainfall_mm"].rolling(6, min_periods=6).sum()
    live["rain_24h"] = live["rainfall_mm"].rolling(24, min_periods=24).sum()

    model_time = pd.Timestamp(live.iloc[-1]["_model_datetime_utc"])
    live["year"] = live["_model_datetime_utc"].dt.year
    live["month"] = live["_model_datetime_utc"].dt.month
    live["day"] = live["_model_datetime_utc"].dt.day
    live["hour"] = live["_model_datetime_utc"].dt.hour
    live["day_of_year"] = live["_model_datetime_utc"].dt.dayofyear
    live["hour_sin"] = np.sin(2 * np.pi * live["hour"] / 24.0)
    live["hour_cos"] = np.cos(2 * np.pi * live["hour"] / 24.0)
    live["doy_sin"] = np.sin(2 * np.pi * live["day_of_year"] / 365.25)
    live["doy_cos"] = np.cos(2 * np.pi * live["day_of_year"] / 365.25)

    sat = imerg_daily.copy()
    sat["date"] = pd.to_datetime(sat["date"], errors="coerce").dt.normalize()
    sat = sat.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    needed = pd.date_range(
        model_time.normalize() - pd.Timedelta(days=7),
        model_time.normalize() - pd.Timedelta(days=1),
        freq="D",
    )
    sat_index = sat.set_index("date")

    if all(d in sat_index.index for d in needed):
        prior = sat_index.loc[needed].copy()
    else:
        # Demo fallback: use the latest seven archived IMERG observations.
        # The UI labels this as stale/historical context so it is not
        # represented as current satellite observation.
        if len(sat_index) < 7:
            missing_sat = [d for d in needed if d not in sat_index.index]
            raise ValueError(
                "IMERG context missing required completed days and no fallback history is available: "
                + ", ".join(d.strftime("%Y-%m-%d") for d in missing_sat)
            )
        prior = sat.sort_values("date").tail(7).set_index("date").copy()
    prior["precip_mm_day"] = pd.to_numeric(prior["precip_mm_day"], errors="coerce")
    if prior["precip_mm_day"].isna().any():
        raise ValueError("IMERG precipitation contains missing values in the required history.")

    cnt = pd.to_numeric(prior.get("precipitation_cnt"), errors="coerce").replace(0, np.nan)
    cond = pd.to_numeric(prior.get("precipitation_cnt_cond"), errors="coerce")
    prior["quality_ratio"] = cond / cnt

    prev = prior.iloc[-1]
    live.loc[live.index[-1], "imerg_prev_day_mm"] = float(prev["precip_mm_day"])
    live.loc[live.index[-1], "imerg_3day_sum_mm"] = float(prior["precip_mm_day"].iloc[-3:].sum())
    live.loc[live.index[-1], "imerg_7day_sum_mm"] = float(prior["precip_mm_day"].sum())
    live.loc[live.index[-1], "imerg_3day_max_mm"] = float(prior["precip_mm_day"].iloc[-3:].max())
    live.loc[live.index[-1], "imerg_7day_max_mm"] = float(prior["precip_mm_day"].max())
    live.loc[live.index[-1], "imerg_prev_day_random_error"] = _clean_scalar(prev.get("randomError"))
    live.loc[live.index[-1], "imerg_prev_day_quality_ratio"] = _clean_scalar(prev.get("quality_ratio"))
    live.loc[live.index[-1], "imerg_prev_day_liquid_probability"] = _clean_scalar(prev.get("probabilityLiquidPrecipitation"))
    live.loc[live.index[-1], "imerg_prev_day_count"] = _clean_scalar(prev.get("precipitation_cnt"))
    live.loc[live.index[-1], "imerg_prev_day_cond_count"] = _clean_scalar(prev.get("precipitation_cnt_cond"))

    latest_model_row = live.iloc[-1].copy()
    X_live = pd.DataFrame([latest_model_row.to_dict()])

    missing_features = [c for c in feature_columns if c not in X_live.columns]
    if missing_features:
        raise ValueError("Could not construct frozen model features: " + ", ".join(missing_features))

    X_live = X_live[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X_live.isna().any().any():
        bad = X_live.columns[X_live.iloc[0].isna()].tolist()
        raise ValueError("Frozen model input still contains NaN values: " + ", ".join(bad))

    # Rebuild the UI row in its original/local timezone so existing risk/UI code can use it.
    local_row = hourly_live.iloc[-1].copy()
    local_row["rainfall_lag_1h"] = float(live.iloc[-1]["rainfall_lag_1h"])
    local_row["rainfall_lag_2h"] = float(live.iloc[-1]["rainfall_lag_2h"])
    local_row["rainfall_lag_3h"] = float(live.iloc[-1]["rainfall_lag_3h"])
    local_row["rainfall_lag_6h"] = float(live.iloc[-1]["rainfall_lag_6h"])
    local_row["rainfall_lag_12h"] = float(live.iloc[-1]["rainfall_lag_12h"])
    local_row["rainfall_lag_24h"] = float(live.iloc[-1]["rainfall_lag_24h"])
    local_row["rain_3h"] = float(live.iloc[-1]["rain_3h"])
    local_row["rain_6h"] = float(live.iloc[-1]["rain_6h"])
    local_row["rain_24h"] = float(live.iloc[-1]["rain_24h"])

    return X_live, local_row
