# ============================================================
# MUMBAI RAINFALL & FLOOD EARLY WARNING SYSTEM
# Complete Streamlit Application
# ============================================================

from pathlib import Path
import json
import base64
from io import BytesIO

import requests


import joblib
from src.sharded_rf import ShardedRandomForestRegressor
import xgboost as xgb
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from streamlit_autorefresh import st_autorefresh

from src.flood_risk import calculate_flood_risk
from src.spatial_weather import (
    fetch_all_locations_with_history,
    MUMBAI_LOCATIONS,
)

from src.live_multisource import (
    build_live_multisource_features,
    fetch_imerg_live_context,
    fetch_imerg_live_contexts,
)
from src.nwp_forecast import fetch_mumbai_nwp_forecast
from src.live_weather import fetch_live_mumbai_weather


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mumbai Flood Early Warning",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN APP
       -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 88% 12%,
                rgba(168, 85, 247, 0.06),
                transparent 24%
            ),
            #0b0d14;
    }


    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }


    /* --------------------------------------------------------
       HEADINGS
       -------------------------------------------------------- */

    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.04em;
    }

    h2 {
        font-weight: 750 !important;
        letter-spacing: -0.02em;
    }

    h3 {
        font-weight: 750 !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #141722 0%,
                #10121b 100%
            );

        border-right:
            1px solid rgba(255, 255, 255, 0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }


    div[role="radiogroup"] label {
        padding:
            0.55rem
            0.7rem;

        border-radius:
            10px;

        margin-bottom:
            0.22rem;

        transition:
            0.2s ease;
    }


    div[role="radiogroup"] label:hover {
        background:
            rgba(255, 255, 255, 0.06);
    }


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        border-radius:
            10px;

        min-height:
            2.8rem;

        font-weight:
            650;

        border:
            1px solid
            rgba(255, 255, 255, 0.08);

        transition:
            all
            0.2s
            ease;
    }


    .stButton > button:hover {
        transform:
            translateY(-1px);

        box-shadow:
            0
            8px
            25px
            rgba(0, 0, 0, 0.25);
    }


    /* --------------------------------------------------------
       METRIC CARDS
       -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.045),
                rgba(255, 255, 255, 0.015)
            );

        border:
            1px solid
            rgba(255, 255, 255, 0.08);

        border-radius:
            14px;

        padding:
            1rem
            1.1rem;

        min-height:
            105px;

        box-shadow:
            0
            10px
            30px
            rgba(0, 0, 0, 0.12);
    }


    div[data-testid="stMetricLabel"] {
        font-size:
            0.84rem;

        opacity:
            0.72;
    }


    div[data-testid="stMetricValue"] {
        font-weight:
            750;
    }


    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius:
            12px;

        border-width:
            1px;
    }


    /* --------------------------------------------------------
       DATAFRAME
       -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius:
            12px;

        overflow:
            hidden;
    }


    /* --------------------------------------------------------
       CHARTS
       -------------------------------------------------------- */

    div[data-testid="stVegaLiteChart"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stArrowVegaLiteChart"] {

        border-radius:
            14px;

        overflow:
            hidden;
    }


    /* --------------------------------------------------------
       DIVIDERS
       -------------------------------------------------------- */

    hr {
        border-color:
            rgba(255, 255, 255, 0.08);

        margin-top:
            1.7rem;

        margin-bottom:
            1.7rem;
    }


    /* --------------------------------------------------------
       EXPANDER
       -------------------------------------------------------- */

    details {
        border-radius:
            12px !important;

        border:
            1px solid
            rgba(255, 255, 255, 0.08) !important;

        background:
            rgba(255, 255, 255, 0.015);
    }


    /* --------------------------------------------------------
       HERO
       -------------------------------------------------------- */

    .hero {

        background:
            linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.14),
                rgba(168, 85, 247, 0.10)
            );

        border:
            1px solid
            rgba(255, 255, 255, 0.08);

        border-radius:
            18px;

        padding:
            1.5rem
            1.6rem;

        margin-bottom:
            1.4rem;

        box-shadow:
            0
            16px
            45px
            rgba(0, 0, 0, 0.15);
    }


    .hero-title {

        font-size:
            1.75rem;

        font-weight:
            800;

        letter-spacing:
            -0.03em;
    }


    .hero-subtitle {

        opacity:
            0.7;

        font-size:
            0.92rem;

        margin-top:
            0.25rem;
    }


    /* --------------------------------------------------------
       SYSTEM STATUS PILL
       -------------------------------------------------------- */

    .pill {

        display:
            inline-flex;

        align-items:
            center;

        gap:
            0.45rem;

        margin-top:
            0.8rem;

        padding:
            0.35rem
            0.7rem;

        border-radius:
            999px;

        background:
            rgba(34, 197, 94, 0.12);

        border:
            1px solid
            rgba(34, 197, 94, 0.20);

        color:
            #86efac;

        font-size:
            0.8rem;

        font-weight:
            650;
    }


    .dot {

        width:
            8px;

        height:
            8px;

        border-radius:
            50%;

        background:
            #22c55e;

        box-shadow:
            0
            0
            10px
            rgba(34, 197, 94, 0.65);
    }


    /* --------------------------------------------------------
       SIDEBAR LABEL
       -------------------------------------------------------- */

    .section-label {

        font-size:
            0.74rem;

        text-transform:
            uppercase;

        letter-spacing:
            0.08em;

        opacity:
            0.5;

        margin:
            0.35rem
            0
            0.45rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "mumbai_era5_ml_ready_cleaned.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

RAIN_MODEL_PATH = (
    MODEL_DIR
    / "rainfall_model.pkl"
)

HEAVY_MODEL_PATH = (
    MODEL_DIR
    / "heavy_rain_classifier.pkl"
)

FEATURES_PATH = (
    MODEL_DIR
    / "rainfall_features.pkl"
)

METRICS_PATH = (
    MODEL_DIR
    / "training_metrics.json"
)

DEM_PATH = (
    BASE_DIR
    / "data"
    / "dem"
    / "mumbai_dem.tif"
)


# ============================================================
# FINAL 2010–2025 MULTISOURCE MODEL ARTIFACTS
# ============================================================

FINAL_MODEL_BUNDLE_PATH = (
    MODEL_DIR
    / "final_multisource_model_bundle.pkl"
)

FINAL_MODEL_XGB_PATH = (
    MODEL_DIR
    / "final_multisource_xgboost.json"
)

FINAL_MODEL_FEATURES_PATH = (
    MODEL_DIR
    / "final_multisource_feature_columns.json"
)

FINAL_MODEL_METADATA_PATH = (
    MODEL_DIR
    / "final_model_metadata.json"
)

FINAL_MODEL_TRAINING_METRICS_PATH = (
    MODEL_DIR
    / "final_training_metrics.json"
)

FINAL_MODEL_PREDICTIONS_PATH = (
    MODEL_DIR
    / "final_multisource_test_predictions.csv"
)

FINAL_MODEL_IMPORTANCE_PATH = (
    MODEL_DIR
    / "final_multisource_feature_importance.csv"
)

FINAL_MULTISOURCE_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "mumbai_era5_imerg_multisource_2010_2025.csv"
)


# ============================================================
# CALIBRATED V2 TWO-STAGE MODEL ARTIFACTS
# ============================================================

V2_BASE_MODEL_PATH = (
    MODEL_DIR
    / "v2_two_stage_base_model_2010_2024.pkl"
)

V2_FUSION_RF_PATH = (
    MODEL_DIR
    / "v2_two_stage_fusion_rf.pkl"
)

V2_CALIBRATION_PATH = (
    MODEL_DIR
    / "v2_calibration.json"
)

V2_CALIBRATED_PREDICTIONS_PATH = (
    MODEL_DIR
    / "v2_calibrated_2025_test_predictions.csv"
)

V2_CALIBRATED_METRICS_PATH = (
    MODEL_DIR
    / "v2_calibrated_metrics.json"
)

V2_BASE_SHARD_DIR = (
    BASE_DIR
    / "v2_shards"
    / "base"
)

V2_FUSION_SHARD_DIR = (
    BASE_DIR
    / "v2_shards"
    / "fusion"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_HEAVY_THRESHOLD = 5.0

DEFAULT_PROBABILITY_THRESHOLD = 0.70

# Final model target
TARGET = "next_hour_rain"

TEST_START_DATE = pd.Timestamp(
    "2024-01-01"
)


# ============================================================
# DEM / TERRAIN LAYER
# ============================================================

def add_dem_elevation(spatial: pd.DataFrame) -> pd.DataFrame:
    """Attach SRTM elevation as terrain context only.

    Elevation is not added to the frozen ERA5 + IMERG model feature vector.
    It is displayed only as additional spatial context.
    """

    result = spatial.copy()

    if rasterio is None:
        result["elevation_m"] = np.nan
        result["elevation_source"] = "Rasterio unavailable"
        return result

    if not DEM_PATH.exists():
        result["elevation_m"] = np.nan
        result["elevation_source"] = "DEM unavailable"
        return result

    elevations = []
    sources = []

    try:
        with rasterio.open(DEM_PATH) as dem:
            for _, row in result.iterrows():
                try:
                    sample = next(
                        dem.sample(
                            [(float(row["longitude"]), float(row["latitude"]))],
                            masked=True,
                        )
                    )[0]

                    if np.ma.is_masked(sample):
                        elevations.append(np.nan)
                        sources.append("DEM NoData")
                        continue

                    value = float(sample)

                    if (
                        not np.isfinite(value)
                        or (
                            dem.nodata is not None
                            and np.isclose(value, float(dem.nodata))
                        )
                    ):
                        elevations.append(np.nan)
                        sources.append("DEM NoData")
                    else:
                        elevations.append(value)
                        sources.append("SRTM 1 Arc-Second")

                except Exception:
                    elevations.append(np.nan)
                    sources.append("DEM sampling error")

    except Exception:
        result["elevation_m"] = np.nan
        result["elevation_source"] = "DEM read error"
        return result

    result["elevation_m"] = elevations
    result["elevation_source"] = sources
    return result


@st.cache_data(show_spinner=False)
def build_dem_overlay():

    if rasterio is None:
        return None, None, None, None

    if not DEM_PATH.exists():
        return None, None, None, None

    try:

        with rasterio.open(DEM_PATH) as dem:

            # Focus the raster around Mumbai + Thane.
            west = 72.75
            south = 18.90
            east = 73.10
            north = 19.35

            window = rasterio.windows.from_bounds(
                west,
                south,
                east,
                north,
                transform=dem.transform,
            )

            data = dem.read(
                1,
                window=window,
                out_shape=(180, 180),
                resampling=rasterio.enums.Resampling.bilinear,
                masked=True,
            )

            values = data.astype("float32").filled(np.nan)

            finite = np.isfinite(values)

            if not finite.any():
                return None, None, None, None

            vmin = float(
                np.nanpercentile(
                    values,
                    2,
                )
            )

            vmax = float(
                np.nanpercentile(
                    values,
                    98,
                )
            )

            if vmax <= vmin:
                vmax = vmin + 1.0

            normalized = (
                values - vmin
            ) / (
                vmax - vmin
            )

            normalized = np.clip(
                normalized,
                0.0,
                1.0,
            )

            # Terrain-style gradient.
            red = (
                55
                + 185 * normalized
            )

            green = (
                155
                - 55 * normalized
            )

            blue = (
                80
                - 45 * normalized
            )

            rgba = np.zeros(
                (
                    values.shape[0],
                    values.shape[1],
                    4,
                ),
                dtype=np.uint8,
            )

            rgba[:, :, 0] = np.where(
                finite,
                red,
                0,
            ).astype(np.uint8)

            rgba[:, :, 1] = np.where(
                finite,
                green,
                0,
            ).astype(np.uint8)

            rgba[:, :, 2] = np.where(
                finite,
                blue,
                0,
            ).astype(np.uint8)

            rgba[:, :, 3] = np.where(
                finite,
                85,
                0,
            ).astype(np.uint8)

            image = Image.fromarray(
                rgba,
                mode="RGBA",
            )

            buffer = BytesIO()

            image.save(
                buffer,
                format="PNG",
            )

            image_uri = (
                "data:image/png;base64,"
                + base64.b64encode(
                    buffer.getvalue()
                ).decode(
                    "utf-8"
                )
            )

            return (
                image_uri,
                [
                    west,
                    south,
                    east,
                    north,
                ],
                vmin,
                vmax,
            )

    except Exception:
        return None, None, None, None


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    rainfall_model = joblib.load(
        RAIN_MODEL_PATH
    )

    heavy_classifier = joblib.load(
        HEAVY_MODEL_PATH
    )

    features = joblib.load(
        FEATURES_PATH
    )

    return (
        rainfall_model,
        heavy_classifier,
        features,
    )


@st.cache_resource
def load_calibrated_v2_model():

    if not V2_BASE_SHARD_DIR.exists():
        raise FileNotFoundError(
            f"Missing V2 base shard directory: {V2_BASE_SHARD_DIR}"
        )

    if not V2_FUSION_SHARD_DIR.exists():
        raise FileNotFoundError(
            f"Missing V2 fusion shard directory: {V2_FUSION_SHARD_DIR}"
        )

    if not V2_CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2 calibration file: {V2_CALIBRATION_PATH}"
        )

    base_model = ShardedRandomForestRegressor(V2_BASE_SHARD_DIR)
    fusion_model = ShardedRandomForestRegressor(V2_FUSION_SHARD_DIR)

    with open(
        V2_CALIBRATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(file)

    base_features = list(base_model.feature_names_in_)
    fusion_features = list(fusion_model.feature_names_in_)

    alpha = float(
        calibration[
            "nwp_correction_alpha"
        ]
    )

    return (
        base_model,
        fusion_model,
        base_features,
        fusion_features,
        alpha,
        calibration,
    )


@st.cache_data(ttl=10 * 60, show_spinner=False)
def predict_v2_batch(stage, inputs):
    """Predict a V2 stage in one batch and cache the result for reruns.

    The sharded RF loads each shard during predict(); batching all spatial
    locations means each model's shards are loaded once instead of once per
    location. The cache also avoids repeating the inference when Streamlit
    reruns because of page/navigation interactions.
    """
    if stage not in {"base", "fusion"}:
        raise ValueError(f"Unknown V2 prediction stage: {stage}")

    (
        base_model,
        fusion_model,
        _,
        _,
        _,
        _,
    ) = load_calibrated_v2_model()

    model = base_model if stage == "base" else fusion_model
    values = model.predict(inputs)
    return np.asarray(values, dtype=float)


@st.cache_resource
def load_final_multisource_model():

    # Prefer the exact Random Forest bundle when it can be loaded.
    # Some local copies can become unreadable/incompatible even though the
    # file exists; in that case fail over to the tracked XGBoost backup so
    # historical/evaluation pages remain usable instead of crashing with a
    # low-level pickle KeyError.
    rf_load_error = None

    if FINAL_MODEL_BUNDLE_PATH.exists():
        try:
            with open(
                FINAL_MODEL_BUNDLE_PATH,
                "rb",
            ) as file:
                bundle = joblib.load(file)
            model_type = "Random Forest"
        except Exception as error:
            rf_load_error = error
            bundle = None
            model_type = None
    else:
        bundle = None
        model_type = None

    if bundle is None:
        if not FINAL_MODEL_XGB_PATH.exists():
            if rf_load_error is not None:
                raise RuntimeError(
                    "The local final Random Forest artifact could not be loaded "
                    f"({type(rf_load_error).__name__}), and the tracked XGBoost "
                    "backup model is also missing."
                ) from rf_load_error
            raise FileNotFoundError(
                "No final multisource inference model is available."
            )

        cloud_model = xgb.XGBRegressor()
        cloud_model.load_model(str(FINAL_MODEL_XGB_PATH))
        bundle = {
            "model": cloud_model,
            "feature_columns": None,
            "model_type": "XGBoost backup (cloud/local fallback)",
        }
        model_type = "XGBoost backup (cloud/local fallback)"

    with open(
        FINAL_MODEL_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    with open(
        FINAL_MODEL_FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        feature_info = json.load(file)

    if bundle.get("feature_columns") is None:
        bundle["feature_columns"] = feature_info.get(
            "feature_columns",
            [],
        )

    bundle["model_type"] = model_type

    return (
        bundle,
        metadata,
        feature_info,
    )


@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def get_cached_imerg_context(target_utc_text: str):
    target_utc = pd.Timestamp(target_utc_text)
    return fetch_imerg_live_context(target_utc, history_days=7)

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def get_cached_imerg_contexts(target_utc_text: str):
    target_utc = pd.Timestamp(target_utc_text)

    return fetch_imerg_live_contexts(
        target_utc=target_utc,
        locations=MUMBAI_LOCATIONS,
        history_days=7,
    )

def load_final_multisource_dataset():

    data = pd.read_csv(
        FINAL_MULTISOURCE_DATA_PATH,
        parse_dates=[
            "datetime",
            "date",
        ],
    )

    return (
        data
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    if not METRICS_PATH.exists():

        return {}

    try:

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        DATA_PATH
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=["datetime"]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Create rainfall lag features if missing
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        6,
        12,
        24,
    ]:

        column = (
            f"rainfall_lag_{lag}h"
        )

        if column not in data.columns:

            data[column] = (
                data["rainfall_mm"]
                .shift(lag)
            )

    return data


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        rainfall_model,
        heavy_classifier,
        features,
    ) = load_models()

    metadata = load_metadata()

    df = load_dataset()

except Exception as error:

    st.error(
        "The ML system could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL SETTINGS
# ============================================================

HEAVY_RAIN_THRESHOLD = float(
    metadata
    .get(
        "target",
        {}
    )
    .get(
        "heavy_rain_threshold_mm",
        DEFAULT_HEAVY_THRESHOLD,
    )
)


PROBABILITY_THRESHOLD = float(
    metadata
    .get(
        "heavy_rain_classification",
        {}
    )
    .get(
        "selected_probability_threshold",
        DEFAULT_PROBABILITY_THRESHOLD,
    )
)


# ============================================================
# HELPER:
# CREATE MODEL INPUT
# ============================================================

def make_model_input(
    row,
    required_features,
):

    values = {}

    missing = []

    for feature in required_features:

        if feature not in row.index:

            missing.append(
                feature
            )

        else:

            values[
                feature
            ] = row[feature]

    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [values],
        columns=required_features,
    )


# ============================================================
# HELPER:
# LIVE V2 NWP FEATURES
# ============================================================

def build_live_v2_nwp_features(
    nwp_forecast,
):
    """Convert live ECMWF forecast into the exact V2 NWP schema."""

    if nwp_forecast is None or nwp_forecast.empty:
        raise ValueError(
            "ECMWF NWP forecast is empty."
        )

    first = nwp_forecast.iloc[0]

    def first_available(
        candidates,
        default=np.nan,
    ):
        for column in candidates:
            if column in first.index:
                value = pd.to_numeric(
                    first[column],
                    errors="coerce",
                )
                if pd.notna(value):
                    return float(value)

        return float(default)

    temperature = first_available(
        [
            "forecast_temperature_c",
            "temperature_2m",
        ]
    )

    humidity = first_available(
        [
            "forecast_humidity_pct",
            "relative_humidity_2m",
        ]
    )

    dewpoint = first_available(
        [
            "forecast_dewpoint_c",
            "dewpoint_2m",
            "dew_point_2m",
        ]
    )

    pressure = first_available(
        [
            "forecast_pressure_hpa",
            "surface_pressure",
        ]
    )

    precipitation = first_available(
        [
            "forecast_precipitation_mm",
            "precipitation",
        ]
    )

    wind_speed = first_available(
        [
            "forecast_wind_speed_ms",
            "wind_speed_10m",
        ]
    )

    wind_u = first_available(
        [
            "forecast_wind_u_ms",
            "wind_u_10m",
        ]
    )

    wind_v = first_available(
        [
            "forecast_wind_v_ms",
            "wind_v_10m",
        ]
    )

    if (
        not np.isfinite(wind_u)
        or not np.isfinite(wind_v)
    ):
        wind_direction = first_available(
            [
                "forecast_wind_direction_deg",
                "wind_direction_10m",
            ]
        )

        if (
            np.isfinite(wind_speed)
            and np.isfinite(wind_direction)
        ):
            direction_rad = np.deg2rad(
                wind_direction
            )

            wind_u = (
                -wind_speed
                * np.sin(direction_rad)
            )

            wind_v = (
                -wind_speed
                * np.cos(direction_rad)
            )

    precip_next_3h = first_available(
        [
            "forecast_next_3h_mm",
        ]
    )

    precip_next_6h = first_available(
        [
            "forecast_next_6h_mm",
        ]
    )

    precip_next_24h = first_available(
        [
            "forecast_next_24h_mm",
        ]
    )

    result = pd.DataFrame(
        [
            {
                "nwp_temperature_1h": temperature,
                "nwp_humidity_1h": humidity,
                "nwp_dewpoint_1h": dewpoint,
                "nwp_pressure_1h": pressure,
                "nwp_precipitation_1h": precipitation,
                "nwp_wind_speed_1h": wind_speed,
                "nwp_wind_u_1h": wind_u,
                "nwp_wind_v_1h": wind_v,
                "nwp_precip_next_3h": precip_next_3h,
                "nwp_precip_next_6h": precip_next_6h,
                "nwp_precip_next_24h": precip_next_24h,
            }
        ]
    )

    return result


# ============================================================
# HELPER:
# RISK CALCULATION
# ============================================================

def calculate_risk_from_row(
    row,
    predicted_rain,
    heavy_probability,
):

    return calculate_flood_risk(

        predicted_rain_mm=float(
            predicted_rain
        ),

        heavy_probability=float(
            heavy_probability
        ),

        current_rain_mm=float(
            row.get(
                "rainfall_mm",
                0.0
            )
        ),

        rain_3h_mm=float(
            row.get(
                "rain_3h",
                0.0
            )
        ),

        rain_6h_mm=float(
            row.get(
                "rain_6h",
                0.0
            )
        ),

        rain_24h_mm=float(
            row.get(
                "rain_24h",
                0.0
            )
        ),
    )


# ============================================================
# HELPER:
# RISK ALERT
# ============================================================

def risk_alert(level):

    if level == "LOW":

        return "🟢", st.success

    if level == "MODERATE":

        return "🟡", st.warning

    if level == "HIGH":

        return "🟠", st.error

    return "🔴", st.error


# ============================================================
# FINAL MULTISOURCE TEST PREDICTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_final_test_predictions():

    # Use the frozen prediction artifact when it is available.
    if FINAL_MODEL_PREDICTIONS_PATH.exists():
        data = pd.read_csv(
            FINAL_MODEL_PREDICTIONS_PATH,
            parse_dates=["datetime"],
        )

        required = [
            "datetime",
            "actual_next_hour_rain",
            "rf_multisource_prediction",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Final prediction file is missing columns: {missing}"
            )

    else:
        # Presentation-safe fallback: rebuild the same 2025 holdout prediction
        # table from the tracked final inference artifact. This keeps the
        # Heavy-Rain Events page functional when the large CSV artifact was not
        # copied into a deployment bundle. The exact RF bundle is preferred
        # locally; the tracked XGBoost backup is used automatically on cloud.
        if not FINAL_MULTISOURCE_DATA_PATH.exists():
            raise FileNotFoundError(
                "Final multisource test predictions are unavailable and the "
                "historical multisource dataset is also missing."
            )

        final_bundle, final_metadata, final_feature_info = (
            load_final_multisource_model()
        )

        data_source = load_final_multisource_dataset()
        chronology = final_metadata.get(
            "chronological_evaluation",
            {},
        )
        test_start = pd.Timestamp(
            chronology.get("test_start", "2025-01-01")
        )
        test_end = pd.Timestamp(
            chronology.get("test_end", "2025-09-30 23:59:59")
        )

        test = data_source[
            (data_source["datetime"] >= test_start)
            & (data_source["datetime"] <= test_end)
        ].copy()

        feature_columns = list(
            final_bundle.get("feature_columns")
            or final_feature_info.get("feature_columns", [])
        )

        if not feature_columns:
            raise ValueError(
                "Final multisource feature columns are unavailable; cannot "
                "rebuild the 2025 holdout prediction table."
            )

        missing_features = [
            column
            for column in feature_columns
            if column not in test.columns
        ]

        if missing_features:
            raise ValueError(
                "Final multisource dataset is missing model features: "
                + ", ".join(missing_features)
            )

        X_test = test[feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if X_test.isna().any().any():
            missing_counts = {
                column: int(X_test[column].isna().sum())
                for column in feature_columns
                if X_test[column].isna().any()
            }
            raise ValueError(
                f"Final holdout features contain missing/non-numeric values: "
                f"{missing_counts}"
            )

        if "next_hour_rain" not in test.columns:
            raise ValueError(
                "Final multisource dataset does not contain the target "
                "column 'next_hour_rain'."
            )

        model = final_bundle["model"]
        prediction = np.maximum(
            np.asarray(model.predict(X_test), dtype=float),
            0.0,
        )

        data = pd.DataFrame(
            {
                "datetime": test["datetime"].to_numpy(),
                "actual_next_hour_rain": pd.to_numeric(
                    test["next_hour_rain"],
                    errors="coerce",
                ).to_numpy(),
                "rf_multisource_prediction": prediction,
            }
        )

        data.attrs["prediction_source"] = final_bundle.get(
            "model_type",
            "deployed final model",
        )

    data = data.sort_values("datetime").reset_index(drop=True)

    data["next_hour_rain"] = pd.to_numeric(
        data["actual_next_hour_rain"],
        errors="coerce",
    )

    data["predicted_rain_mm"] = np.clip(
        pd.to_numeric(
            data["rf_multisource_prediction"],
            errors="coerce",
        ),
        0.0,
        None,
    )

    data["absolute_error_mm"] = (
        data["next_hour_rain"]
        - data["predicted_rain_mm"]
    ).abs()

    data["actual_heavy"] = (
        data["next_hour_rain"] >= HEAVY_RAIN_THRESHOLD
    ).astype(int)

    # The final multisource model is a regression model, not the old
    # probability classifier. Heavy-rain detection is therefore derived from
    # the predicted rainfall crossing the configured threshold.
    data["predicted_heavy"] = (
        data["predicted_rain_mm"] >= HEAVY_RAIN_THRESHOLD
    ).astype(int)

    return data


def build_heavy_rain_episode_summary(data):
    """Group consecutive heavy-rain hours into continuous episodes.

    An episode is a run of heavy-rain observations (>= threshold) with no
    gap greater than one hour between consecutive observations. This keeps
    the dashboard from counting every heavy hour in a continuous storm as a
    separate event.
    """

    d = (
        data.copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    one_hour = pd.Timedelta(hours=1)
    prev_time = d["datetime"].shift(1)
    gap = d["datetime"] - prev_time

    actual_mask = d["actual_heavy"].eq(1)
    actual_start = actual_mask & (
        ~actual_mask.shift(1, fill_value=False)
        | (gap > one_hour)
    )
    d["actual_episode_id"] = np.where(
        actual_mask,
        actual_start.cumsum(),
        np.nan,
    )

    false_alarm_mask = (
        d["actual_heavy"].eq(0)
        & d["predicted_heavy"].eq(1)
    )
    false_start = false_alarm_mask & (
        ~false_alarm_mask.shift(1, fill_value=False)
        | (gap > one_hour)
    )
    d["false_alarm_episode_id"] = np.where(
        false_alarm_mask,
        false_start.cumsum(),
        np.nan,
    )

    actual_episode_rows = []

    actual_only = d[d["actual_episode_id"].notna()].copy()

    for episode_id, group in actual_only.groupby("actual_episode_id", sort=True):
        detected_hours = int(group["predicted_heavy"].sum())

        actual_episode_rows.append(
            {
                "Episode": int(episode_id),
                "Start": group["datetime"].min(),
                "End": group["datetime"].max() + one_hour,
                "Duration (hours)": len(group),
                "Actual max rain (mm/h)": group["next_hour_rain"].max(),
                "Detected heavy hours": detected_hours,
                "Max predicted rain (mm/h)": group["predicted_rain_mm"].max(),
                "Status": "Detected" if detected_hours > 0 else "Missed",
            }
        )

    actual_episodes = pd.DataFrame(actual_episode_rows)

    if not actual_episodes.empty:
        actual_episodes["Start"] = pd.to_datetime(actual_episodes["Start"])
        actual_episodes["End"] = pd.to_datetime(actual_episodes["End"])

    false_alarm_rows = []
    false_only = d[d["false_alarm_episode_id"].notna()].copy()

    for episode_id, group in false_only.groupby("false_alarm_episode_id", sort=True):
        false_alarm_rows.append(
            {
                "Episode": int(episode_id),
                "Start": group["datetime"].min(),
                "End": group["datetime"].max() + one_hour,
                "Duration (hours)": len(group),
                "Max actual rain (mm/h)": group["next_hour_rain"].max(),
                "Max predicted rain (mm/h)": group["predicted_rain_mm"].max(),
            }
        )

    false_alarm_episodes = pd.DataFrame(false_alarm_rows)

    if not false_alarm_episodes.empty:
        false_alarm_episodes["Start"] = pd.to_datetime(false_alarm_episodes["Start"])
        false_alarm_episodes["End"] = pd.to_datetime(false_alarm_episodes["End"])

    detected_episode_count = int(
        (actual_episodes["Status"] == "Detected").sum()
    ) if not actual_episodes.empty else 0

    missed_episode_count = int(
        (actual_episodes["Status"] == "Missed").sum()
    ) if not actual_episodes.empty else 0

    actual_episode_count = len(actual_episodes)
    false_alarm_episode_count = len(false_alarm_episodes)

    episode_detection_rate = (
        detected_episode_count / actual_episode_count
        if actual_episode_count
        else 0.0
    )

    missed_episodes = actual_episodes[
        actual_episodes["Status"] == "Missed"
    ].copy()

    detected_episodes = actual_episodes[
        actual_episodes["Status"] == "Detected"
    ].copy()

    return {
        "data": d,
        "actual_episodes": actual_episodes,
        "detected_episodes": detected_episodes,
        "missed_episodes": missed_episodes,
        "false_alarm_episodes": false_alarm_episodes,
        "actual_episode_count": actual_episode_count,
        "detected_episode_count": detected_episode_count,
        "missed_episode_count": missed_episode_count,
        "false_alarm_episode_count": false_alarm_episode_count,
        "episode_detection_rate": episode_detection_rate,
    }


# ============================================================
# HELPER:
# TEST PREDICTIONS
# ============================================================

def create_test_predictions():

    required = (
        list(features)
        + [
            "datetime",
            "next_hour_rain",
        ]
    )

    available = [
        column
        for column in required
        if column in df.columns
    ]

    test = (
        df[
            available
        ]
        .copy()
    )

    test = (
        test
        .dropna(
            subset=[
                feature
                for feature
                in features
                if feature in test.columns
            ]
        )
    )

    test = test[
        test["datetime"]
        >= TEST_START_DATE
    ].copy()

    test = (
        test
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if test.empty:

        return test

    X = test[
        features
    ]

    y = test[
        "next_hour_rain"
    ]

    predicted = np.maximum(
        rainfall_model.predict(X),
        0.0,
    )

    probability = np.clip(
        heavy_classifier
        .predict_proba(X)[:, 1],
        0.0,
        1.0,
    )

    test[
        "predicted_rain_mm"
    ] = predicted

    test[
        "heavy_probability"
    ] = probability

    test[
        "predicted_heavy"
    ] = (
        probability
        >= PROBABILITY_THRESHOLD
    ).astype(int)

    test[
        "actual_heavy"
    ] = (
        y
        >= HEAVY_RAIN_THRESHOLD
    ).astype(int)

    test[
        "absolute_error_mm"
    ] = np.abs(
        y
        - predicted
    )

    return test


test_predictions = (
    create_test_predictions()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.html(
    """
    <div style="padding: .4rem 0 1.1rem;">
        <div style="
            font-size: 1.35rem;
            font-weight: 800;
        ">
            🌧️ Mumbai Flood AI
        </div>

        <div style="
            font-size: .78rem;
            opacity: .6;
            margin-top: .2rem;
        ">
            Early Warning Dashboard
        </div>
    </div>
    """
)


st.sidebar.markdown(
    """
    <div class="section-label">
        Monitoring Views
    </div>
    """,
    unsafe_allow_html=True,
)


page = st.sidebar.radio(
    "",
    [
        "🔴 Live Mumbai",
        "🗺️ Mumbai Risk Map",
        "📊 Historical ERA5 Demo",
        "🧪 Custom Weather Input",
        "📈 V2 Model Evaluation",
        "⚠️ Heavy-Rain Events",
        "🧠 V1 Historical Baseline",
    ],
)


st.sidebar.divider()


st.sidebar.html(
    """
    <div class="section-label">
        Monitoring Views
    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-title">
            🌧️ Mumbai Rainfall & Flood Early Warning System
        </div>

        <div class="hero-subtitle">
            multisource rainfall forecasting •
            heavy-rain detection •
            spatial flood-risk assessment
        </div>

        <span class="pill">
            <span class="dot"></span>
            System Online
        </span>
    </div>
    """
)



# ============================================================
# HELPER:
# LOCATION-SPECIFIC ECMWF NWP FOR V2 SPATIAL INFERENCE
# ============================================================

def _prepare_location_nwp_forecast(
    payload,
    forecast_hours=48,
):
    """Normalize one ECMWF API response into the exact V2 NWP schema."""

    if forecast_hours < 24:
        forecast_hours = 24

    if forecast_hours > 72:
        forecast_hours = 72

    hourly = payload.get("hourly")

    if not hourly:
        raise ValueError(
            "ECMWF response did not contain hourly forecast data."
        )

    forecast = pd.DataFrame(hourly)
    forecast["datetime"] = pd.to_datetime(forecast["time"])
    forecast = forecast.sort_values("datetime").reset_index(drop=True)

    forecast["forecast_temperature_c"] = pd.to_numeric(
        forecast["temperature_2m"], errors="coerce"
    )
    forecast["forecast_humidity_pct"] = pd.to_numeric(
        forecast["relative_humidity_2m"], errors="coerce"
    )
    forecast["forecast_dewpoint_c"] = pd.to_numeric(
        forecast["dew_point_2m"], errors="coerce"
    )
    forecast["forecast_pressure_hpa"] = pd.to_numeric(
        forecast["surface_pressure"], errors="coerce"
    )
    forecast["forecast_precipitation_mm"] = pd.to_numeric(
        forecast["precipitation"], errors="coerce"
    ).fillna(0.0)
    forecast["forecast_wind_speed_ms"] = pd.to_numeric(
        forecast["wind_speed_10m"], errors="coerce"
    )
    forecast["forecast_wind_direction_deg"] = pd.to_numeric(
        forecast["wind_direction_10m"], errors="coerce"
    )

    direction_rad = np.deg2rad(
        forecast["forecast_wind_direction_deg"]
    )
    forecast["forecast_wind_u_ms"] = (
        -forecast["forecast_wind_speed_ms"]
        * np.sin(direction_rad)
    )
    forecast["forecast_wind_v_ms"] = (
        -forecast["forecast_wind_speed_ms"]
        * np.cos(direction_rad)
    )

    now_local = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).tz_localize(None).floor("h")

    forecast = forecast[
        (forecast["datetime"] >= now_local)
        & (
            forecast["datetime"]
            < now_local + pd.Timedelta(hours=forecast_hours)
        )
    ].copy()

    if forecast.empty:
        raise ValueError(
            "No future ECMWF forecast rows were returned."
        )

    precip = forecast["forecast_precipitation_mm"].reset_index(drop=True)
    next_1h = float(precip.head(1).sum())
    next_3h = float(precip.head(3).sum())
    next_6h = float(precip.head(6).sum())
    next_24h = float(precip.head(24).sum())

    forecast["forecast_next_1h_mm"] = next_1h
    forecast["forecast_next_3h_mm"] = next_3h
    forecast["forecast_next_6h_mm"] = next_6h
    forecast["forecast_next_24h_mm"] = next_24h

    return forecast.reset_index(drop=True)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_location_nwp_forecast(
    latitude,
    longitude,
    forecast_hours=48,
):
    """Fetch location-specific ECMWF HRES input features for V2 fusion."""

    if forecast_hours < 24:
        forecast_hours = 24

    if forecast_hours > 72:
        forecast_hours = 72

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        "https://api.open-meteo.com/v1/ecmwf",
        params=params,
        timeout=(5, 30),
    )
    response.raise_for_status()

    return _prepare_location_nwp_forecast(
        response.json(),
        forecast_hours=forecast_hours,
    )


@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_spatial_nwp_forecasts(
    location_specs,
    forecast_hours=48,
):
    """Fetch all Mumbai ECMWF forecasts in one API request.

    Open-Meteo supports comma-separated latitude/longitude coordinates and
    returns one JSON structure per requested location. This avoids the
    nine-request/thread-pool pattern that made the Risk Map fragile.
    """

    if not location_specs:
        return {}

    latitudes = ",".join(
        f"{float(spec[1]):.6f}"
        for spec in location_specs
    )
    longitudes = ",".join(
        f"{float(spec[2]):.6f}"
        for spec in location_specs
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/ecmwf",
            params=params,
            timeout=(5, 20),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    payloads = payload if isinstance(payload, list) else [payload]

    if len(payloads) != len(location_specs):
        return {}

    results = {}

    for spec, location_payload in zip(
        location_specs,
        payloads,
    ):
        try:
            results[spec[0]] = _prepare_location_nwp_forecast(
                location_payload,
                forecast_hours=forecast_hours,
            )
        except Exception:
            continue

    return results


# ============================================================
# PAGE 1
# LIVE MUMBAI
# ============================================================

if page == "🔴 Live Mumbai":

    # --------------------------------------------------------
    # Automatic refresh
    # --------------------------------------------------------

    st_autorefresh(
        interval=10 * 60 * 1000,
        key="live_mumbai_refresh",
    )

    st.header(
        "🔴 Live Mumbai Weather & Early Warning"
    )

    st.caption(
        "Live monitoring with automatic weather refresh every 10 minutes. "
        "The final next-hour rainfall prediction uses the calibrated V2 "
        "ERA5 + IMERG base model with ECMWF NWP residual correction. "
        "The raw ECMWF forecast below is shown as the NWP input guidance."
    )


    # --------------------------------------------------------
    # Manual refresh
    # --------------------------------------------------------

    if st.button(
        "🔄 Refresh Live Weather Now",
        type="primary",
        use_container_width=True,
    ):

        st.rerun()


    # ========================================================
    # LIVE WEATHER
    # ========================================================

    try:

        live_data = (
            fetch_live_mumbai_weather()
        )

        hourly_live = (
            live_data[
                "hourly"
            ]
            .copy()
        )

    except Exception as error:

        st.error(
            "Unable to retrieve live Mumbai weather."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Latest usable observation
    # --------------------------------------------------------

    now_local = (
        pd.Timestamp
        .now(
            tz="Asia/Kolkata"
        )
        .tz_localize(None)
    )


    available_now = (
        hourly_live[
            hourly_live[
                "datetime"
            ]
            <= now_local
        ]
    )


    if available_now.empty:

        latest_live = (
            hourly_live
            .iloc[-1]
            .copy()
        )

    else:

        latest_live = (
            available_now
            .iloc[-1]
            .copy()
        )


    latest_time = pd.Timestamp(
        latest_live[
            "datetime"
        ]
    )


    st.info(
        "Latest available Mumbai observation: "
        f"{latest_time.strftime('%d %B %Y, %H:%M')} IST"
    )


    # ========================================================
    # CURRENT LIVE WEATHER
    # ========================================================

    st.subheader(
        "Current Mumbai Weather"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Rainfall",
            f"{float(latest_live['rainfall_mm']):.2f} mm",
        )


    with c2:

        st.metric(
            "Temperature",
            f"{float(latest_live['temperature_c']):.1f} °C",
        )


    with c3:

        st.metric(
            "Humidity",
            f"{float(latest_live['humidity_pct']):.1f}%",
        )


    with c4:

        st.metric(
            "Pressure",
            f"{float(latest_live['pressure_hpa']):.1f} hPa",
        )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Wind speed",
            f"{float(latest_live['wind_speed_ms']):.2f} m/s",
        )


    with c2:

        st.metric(
            "U10",
            f"{float(latest_live['u10_ms']):.2f} m/s",
        )


    with c3:

        st.metric(
            "V10",
            f"{float(latest_live['v10_ms']):.2f} m/s",
        )


    # ========================================================
    # RECENT RAINFALL
    # ========================================================

    st.subheader(
        "Recent Rainfall"
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Last 3 hours",
            f"{float(latest_live['rain_3h']):.2f} mm",
        )


    with c2:

        st.metric(
            "Last 6 hours",
            f"{float(latest_live['rain_6h']):.2f} mm",
        )


    with c3:

        st.metric(
            "Last 24 hours",
            f"{float(latest_live['rain_24h']):.2f} mm",
        )


    # ========================================================
    # ECMWF NWP FORECAST
    # ========================================================

    st.divider()

    try:

        nwp_result = fetch_mumbai_nwp_forecast(
            forecast_hours=48
        )

        nwp_forecast = (
            nwp_result[
                "forecast"
            ]
            .copy()
        )

    except Exception as error:

        nwp_result = None
        nwp_forecast = None

        st.warning(
            "ECMWF NWP forecast is temporarily unavailable."
        )

        st.caption(
            f"NWP error: {error}"
        )


    if (
        nwp_forecast is not None
        and not nwp_forecast.empty
    ):

        st.subheader(
            "ECMWF IFS HRES — NWP Input to Calibrated V2"
        )

        st.caption(
            "ECMWF IFS HRES 9 km via Open-Meteo • "
            "48-hour hourly forecast • "
            f"Fetched "
            f"{nwp_result['fetched_at'].strftime('%d %B %Y, %H:%M')} IST"
        )


        first_nwp = (
            nwp_forecast
            .iloc[0]
        )


        nwp_col1, nwp_col2, nwp_col3, nwp_col4 = (
            st.columns(4)
        )


        with nwp_col1:

            st.metric(
                "Next 1 hour",
                f"{float(first_nwp['forecast_next_1h_mm']):.1f} mm",
            )


        with nwp_col2:

            st.metric(
                "Next 3 hours",
                f"{float(first_nwp['forecast_next_3h_mm']):.1f} mm",
            )


        with nwp_col3:

            st.metric(
                "Next 6 hours",
                f"{float(first_nwp['forecast_next_6h_mm']):.1f} mm",
            )


        with nwp_col4:

            st.metric(
                "Next 24 hours",
                f"{float(first_nwp['forecast_next_24h_mm']):.1f} mm",
            )


        # ----------------------------------------------------
        # Peak rainfall
        # ----------------------------------------------------

        peak_rain = float(
            nwp_forecast[
                "forecast_precipitation_mm"
            ]
            .max()
        )


        peak_index = (
            nwp_forecast[
                "forecast_precipitation_mm"
            ]
            .idxmax()
        )


        peak_time = pd.Timestamp(
            nwp_forecast.loc[
                peak_index,
                "datetime",
            ]
        )


        st.info(
            "ECMWF peak hourly rainfall forecast: "
            f"{peak_rain:.1f} mm at "
            f"{peak_time.strftime('%d %B %Y, %H:%M')} IST"
        )


        # ----------------------------------------------------
        # 24-hour forecast chart
        # ----------------------------------------------------

        st.subheader(
            "ECMWF Forecast Rainfall — Next 24 Hours"
        )


        nwp_chart = (
            nwp_forecast[
                [
                    "datetime",
                    "forecast_precipitation_mm",
                ]
            ]
            .head(24)
            .set_index(
                "datetime"
            )
            .rename(
                columns={
                    "forecast_precipitation_mm":
                        "Forecast rainfall (mm/h)"
                }
            )
        )


        st.line_chart(
            nwp_chart,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # Forecast conditions
        # ----------------------------------------------------

        st.subheader(
            "ECMWF Forecast Conditions — Next 24 Hours"
        )


        forecast_table = (
            nwp_forecast[
                [
                    "datetime",
                    "forecast_temperature_c",
                    "forecast_humidity_pct",
                    "forecast_pressure_hpa",
                    "forecast_wind_speed_ms",
                    "forecast_precipitation_mm",
                ]
            ]
            .head(24)
            .copy()
        )


        forecast_table = (
            forecast_table
            .rename(
                columns={

                    "datetime":
                        "Time",

                    "forecast_temperature_c":
                        "Temperature (°C)",

                    "forecast_humidity_pct":
                        "Humidity (%)",

                    "forecast_pressure_hpa":
                        "Pressure (hPa)",

                    "forecast_wind_speed_ms":
                        "Wind (m/s)",

                    "forecast_precipitation_mm":
                        "Rain (mm/h)",
                }
            )
        )


        forecast_table["Time"] = (
            pd.to_datetime(
                forecast_table["Time"]
            )
            .dt.strftime(
                "%d %b %H:%M"
            )
        )


        st.dataframe(
            forecast_table,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "NWP source: ECMWF IFS HRES 9 km via Open-Meteo. "
            "These raw NWP values are fed into the V2 fusion RF; "
            "they are not the final rainfall warning by themselves."
        )


    # ========================================================
    # CALIBRATED V2 MODEL — BASE RF + ECMWF NWP
    # ========================================================

    st.divider()

    try:

        (
            v2_base_model,
            v2_fusion_model,
            v2_base_features,
            v2_fusion_features,
            v2_alpha,
            v2_calibration,
        ) = load_calibrated_v2_model()

        # ----------------------------------------------------
        # Convert local observation time to UTC
        # ----------------------------------------------------

        probe_time = pd.Timestamp(
            latest_time
        )

        if probe_time.tzinfo is None:

            probe_time = (
                probe_time
                .tz_localize(
                    "Asia/Kolkata"
                )
            )

        probe_time_utc = (
            probe_time
            .tz_convert(
                "UTC"
            )
            .tz_localize(None)
        )

        # ----------------------------------------------------
        # Cached IMERG context
        # ----------------------------------------------------

        imerg_live = (
            get_cached_imerg_context(
                probe_time_utc
                .normalize()
                .isoformat()
            )
        )

        # ----------------------------------------------------
        # Build V2 base-model feature vector
        # ----------------------------------------------------

        live_input, latest_live = (
            build_live_multisource_features(
                hourly_live=hourly_live,
                imerg_daily=imerg_live,
                feature_columns=v2_base_features,
            )
        )

        # ----------------------------------------------------
        # Stage 1 — Base RF prediction
        # ----------------------------------------------------

        base_prediction = max(
            0.0,
            float(
                v2_base_model.predict(
                    live_input
                )[0]
            ),
        )

        # ----------------------------------------------------
        # Stage 2 — ECMWF NWP residual correction
        # ----------------------------------------------------

        live_nwp = build_live_v2_nwp_features(
            nwp_forecast
        )

        live_nwp["base_prediction"] = (
            base_prediction
        )

        fusion_input = live_nwp[
            v2_fusion_features
        ]

        nwp_correction = float(
            v2_fusion_model.predict(
                fusion_input
            )[0]
        )

        applied_nwp_correction = (
            v2_alpha
            * nwp_correction
        )

        predicted_rain = max(
            0.0,
            base_prediction
            + applied_nwp_correction
        )

        heavy_warning = (
            predicted_rain
            >= HEAVY_RAIN_THRESHOLD
        )

        heavy_signal = (
            1.0
            if heavy_warning
            else 0.0
        )

        # ----------------------------------------------------
        # Satellite metadata
        # ----------------------------------------------------

        satellite_latest_date = (
            pd.to_datetime(
                imerg_live[
                    "date"
                ]
            )
            .max()
            .normalize()
        )


        model_date = (
            probe_time_utc
            .normalize()
        )


        satellite_age_days = int(
            (
                model_date
                - satellite_latest_date
            ).days
        )
        # ========================================================
        # NASA IMERG SATELLITE INPUT — JUDGE DEMO
        # ========================================================

        st.divider()

        st.subheader(
            "🛰️ NASA IMERG Satellite Input"
        )

        demo_satellite_source = imerg_live.attrs.get(
            "source",
            "unknown",
        )

        # --------------------------------------------------------
        # Live / fallback status
        # --------------------------------------------------------

        if demo_satellite_source == "historical_fallback":

            st.warning(
                "🟡 NASA IMERG live access is unavailable. "
                "The model is using archived IMERG history as a fallback."
            )

        else:

            st.success(
                "🟢 NASA GPM IMERG Late Daily V07 is providing "
                "the satellite rainfall context used by the model."
            )

        # --------------------------------------------------------
        # Satellite summary
        # --------------------------------------------------------

        satellite_history = (
            imerg_live.copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        completed_days = len(satellite_history)

        latest_completed_day = (
            pd.to_datetime(
                satellite_history["date"]
            )
            .max()
        )

        total_7day_rain = float(
            pd.to_numeric(
                satellite_history["precip_mm_day"],
                errors="coerce",
            ).sum()
        )

        previous_day_rain = float(
            pd.to_numeric(
                satellite_history["precip_mm_day"],
                errors="coerce",
            ).iloc[-1]
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Completed IMERG days",
                completed_days,
            )

        with s2:

            st.metric(
                "Latest satellite day",
                latest_completed_day.strftime(
                    "%d %b %Y"
                ),
            )

        with s3:

            st.metric(
                "7-day satellite rainfall",
                f"{total_7day_rain:.2f} mm",
            )

        with s4:

            st.metric(
                "Previous-day rainfall",
                f"{previous_day_rain:.2f} mm",
            )

        st.caption(
            "The model uses completed IMERG days strictly before "
            "the prediction time. The latest available satellite "
            "day is therefore not necessarily the current calendar day."
        )

        # --------------------------------------------------------
        # Raw 7-day NASA IMERG history
        # --------------------------------------------------------

        st.subheader(
            "📡 NASA IMERG — Previous 7 Completed Days"
        )

        history_display = satellite_history.copy()

        history_display["date"] = pd.to_datetime(
            history_display["date"]
        )

        history_display["precip_mm_day"] = pd.to_numeric(
            history_display["precip_mm_day"],
            errors="coerce",
        ).round(2)

        # Satellite quality ratio
        if (
            "precipitation_cnt" in history_display.columns
            and "precipitation_cnt_cond" in history_display.columns
        ):

            count = pd.to_numeric(
                history_display["precipitation_cnt"],
                errors="coerce",
            ).replace(0, np.nan)

            conditional_count = pd.to_numeric(
                history_display["precipitation_cnt_cond"],
                errors="coerce",
            )

            history_display["quality_ratio"] = (
                conditional_count / count
            ).clip(
                lower=0,
                upper=1,
            )

        else:

            history_display["quality_ratio"] = np.nan

        history_columns = [
            "date",
            "precip_mm_day",
            "quality_ratio",
            "randomError",
            "probabilityLiquidPrecipitation",
        ]

        available_history_columns = [
            column
            for column in history_columns
            if column in history_display.columns
        ]

        history_table = (
            history_display[
                available_history_columns
            ]
            .copy()
            .rename(
                columns={
                    "date": "Satellite Date",
                    "precip_mm_day": "Rainfall (mm)",
                    "quality_ratio": "Quality Ratio",
                    "randomError": "Random Error",
                    "probabilityLiquidPrecipitation":
                        "Liquid Precipitation Probability",
                }
            )
        )

        if "Quality Ratio" in history_table.columns:

            history_table["Quality Ratio"] = (
                history_table["Quality Ratio"]
                .round(3)
            )

        if "Random Error" in history_table.columns:

            history_table["Random Error"] = (
                pd.to_numeric(
                    history_table["Random Error"],
                    errors="coerce",
                )
                .round(2)
            )

        if "Liquid Precipitation Probability" in history_table.columns:

            history_table["Liquid Precipitation Probability"] = (
                pd.to_numeric(
                    history_table[
                        "Liquid Precipitation Probability"
                    ],
                    errors="coerce",
                )
                .round(3)
            )

        history_table["Satellite Date"] = (
            pd.to_datetime(
                history_table["Satellite Date"]
            )
            .dt.strftime(
                "%d %b %Y"
            )
        )

        st.dataframe(
            history_table,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------
        # 7-day satellite rainfall chart
        # --------------------------------------------------------

        st.subheader(
            "📈 IMERG Satellite Rainfall History"
        )

        imerg_chart = (
            satellite_history[
                [
                    "date",
                    "precip_mm_day",
                ]
            ]
            .copy()
            .set_index("date")
            .rename(
                columns={
                    "precip_mm_day":
                        "NASA IMERG rainfall (mm/day)"
                }
            )
        )

        st.line_chart(
            imerg_chart,
            use_container_width=True,
        )

        # --------------------------------------------------------
        # EXACT IMERG FEATURES SENT TO THE MODEL
        # --------------------------------------------------------

        st.subheader(
            "🧠 IMERG Features Used by the ML Model"
        )

        st.caption(
            "These are the satellite-derived features contained "
            "in the exact model input vector sent to the rainfall model."
        )

        imerg_model_features = [
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

        available_model_features = [
            feature
            for feature in imerg_model_features
            if feature in live_input.columns
        ]

        imerg_feature_table = pd.DataFrame(
            {
                "Model Feature": available_model_features,
                "Value": [
                    pd.to_numeric(
                        live_input.iloc[0][feature],
                        errors="coerce",
                    )
                    for feature in available_model_features
                ],
            }
        )

        imerg_feature_table["Value"] = (
            imerg_feature_table["Value"]
            .round(4)
        )

        st.dataframe(
            imerg_feature_table,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------
        # Demonstrate the full linkage
        # --------------------------------------------------------

        st.markdown(
            """
            <div style="
                padding: 1rem 1.2rem;
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,.10);
                background: rgba(255,255,255,.025);
                margin-top: .8rem;
            ">
                <b>NASA IMERG → 7-day satellite history → engineered
                satellite features → ML model input → next-hour rainfall prediction</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        linkage_col1, linkage_col2, linkage_col3 = st.columns(3)

        with linkage_col1:

            st.metric(
                "NASA IMERG context",
                f"{completed_days} completed days",
            )

        with linkage_col2:

            st.metric(
                "IMERG 7-day accumulation",
                f"{total_7day_rain:.2f} mm",
            )

        with linkage_col3:

            st.metric(
                "ML next-hour prediction",
                f"{predicted_rain:.2f} mm",
            )
            


    except Exception as error:

        st.error(
            "Unable to create the live multisource prediction."
        )

        st.exception(error)

        st.info(
            "The live page requires the final multisource feature pipeline "
            "and at least 25 hourly weather observations. "
            "The IMERG module includes the archived fallback used when "
            "NASA live access is unavailable."
        )

        st.stop()


    # ========================================================
    # CALIBRATED V2 STATUS
    # ========================================================

    status_col1, status_col2, status_col3 = (
        st.columns(3)
    )

    satellite_source = (
        imerg_live.attrs.get(
            "source",
            "live_late",
        )
    )

    with status_col1:

        st.success(
            "🧠 Calibrated V2 RF + ECMWF NWP"
        )

    with status_col2:

        st.info(
            f"⚙️ NWP correction × {v2_alpha:.2f}"
        )

    with status_col3:

        if satellite_source == "historical_fallback":

            st.warning(
                "🛰️ Archived IMERG context • "
                f"{satellite_age_days} day(s) old"
            )

        elif satellite_age_days <= 2:

            st.success(
                "🛰️ Live IMERG context • "
                f"{satellite_age_days} day(s) old"
            )

        else:

            st.warning(
                "🛰️ IMERG context • "
                f"{satellite_age_days} day(s) old"
            )

    st.caption(
        "Live prediction pipeline: V2 Base RF → ECMWF NWP residual "
        f"correction × {v2_alpha:.2f} → calibrated next-hour rainfall. "
        "NASA IMERG provides satellite rainfall context."
    )


    # ========================================================
    # V2 PREDICTION BREAKDOWN
    # ========================================================

    with st.expander("🧠 V2 Prediction Breakdown"):

        breakdown = pd.DataFrame({
            "Component": [
                "Base RF prediction",
                "Raw NWP correction",
                "Calibration factor",
                "Applied NWP correction",
                "Final calibrated V2 prediction",
            ],
            "Value": [
                base_prediction,
                nwp_correction,
                v2_alpha,
                applied_nwp_correction,
                predicted_rain,
            ],
            "Unit": [
                "mm",
                "mm",
                "alpha",
                "mm",
                "mm",
            ],
        })

        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # FLOOD RISK
    # ========================================================

    risk = calculate_risk_from_row(
        latest_live,
        predicted_rain,
        heavy_signal,
    )


    risk_level = (
        risk[
            "risk_level"
        ]
    )


    st.divider()


    icon, alert_function = (
        risk_alert(
            risk_level
        )
    )


    alert_function(
        f"{icon} CURRENT FLOOD RISK: "
        f"{risk_level}"
    )


    # ========================================================
    # MODEL WARNING STATUS
    # ========================================================

    st.subheader(
        "Current Warning Status"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "ML next-hour rainfall",
            f"{predicted_rain:.2f} mm",
        )


    with c2:

        st.metric(
            "Heavy-rain model signal",
            "⚠️ YES"
            if heavy_warning
            else "✅ NO",
        )


    with c3:

        st.metric(
            "Flood-risk score",
            f"{risk['score']:.0f}",
        )


    with c4:

        if heavy_warning:

            st.error(
                "⚠️ HEAVY-RAIN WARNING"
            )

        else:

            st.success(
                "✅ NO HEAVY-RAIN WARNING"
            )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    if risk_level == "CRITICAL":

        st.error(
            f"🚨 {risk['action']}"
        )

    elif risk_level == "HIGH":

        st.error(
            f"⚠️ {risk['action']}"
        )

    elif risk_level == "MODERATE":

        st.warning(
            risk["action"]
        )

    else:

        st.info(
            risk["action"]
        )


    # ========================================================
    # HISTORICAL LIVE RAINFALL CHART
    # ========================================================

    st.subheader(
        "Mumbai Rainfall — Last 24 Hours"
    )


    chart_data = (
        hourly_live[
            hourly_live[
                "datetime"
            ]
            <= latest_time
        ]
        .tail(24)
        [
            [
                "datetime",
                "rainfall_mm",
            ]
        ]
        .set_index(
            "datetime"
        )
        .rename(
            columns={
                "rainfall_mm":
                    "Hourly rainfall (mm)"
            }
        )
    )


    st.line_chart(
        chart_data,
        use_container_width=True,
    )


    # ========================================================
    # RISK FACTORS
    # ========================================================

    st.divider()


    st.subheader(
        "Risk Factors"
    )


    for factor in risk[
        "factors"
    ]:

        st.write(
            f"• {factor}"
        )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.divider()


    st.subheader(
        "System Status"
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.success(
            "🟢 Weather API: Connected"
        )


    with c2:

        st.success(
            "🟢 ML Model: Running"
        )


    with c3:

        st.success(
            "🟢 Risk Engine: Running"
        )


    st.caption(
        "Live sources: Open-Meteo weather + NASA IMERG + ECMWF IFS HRES. "
        "The Live Mumbai prediction uses the calibrated V2 Base RF + ECMWF NWP "
        "residual fusion pipeline. The V1 historical baseline is retained separately "
        "for reference and evaluation."
    )




# ============================================================
# PAGE 2
# MUMBAI SPATIAL RISK MAP
# ============================================================

elif page == "🗺️ Mumbai Risk Map":

    st.header("🗺️ Mumbai Spatial Flood-Risk Map")

    st.info(
        "Each location uses the same calibrated V2 rainfall pipeline as the Live "
        "Mumbai page: ERA5/Open-Meteo + location-specific IMERG → V2 Base RF → "
        "location-specific ECMWF NWP residual RF → calibration alpha 0.50. "
        "The spatial flood-risk layer remains a rule-based prototype."
    )

    if st.button("🔄 Refresh Mumbai Risk Map", type="primary", use_container_width=True):
        st.rerun()

    try:
        (
            v2_base_model,
            v2_fusion_model,
            v2_base_features,
            v2_fusion_features,
            v2_alpha,
            v2_calibration,
        ) = load_calibrated_v2_model()

        spatial_model_type = (
            "Calibrated V2 RF + ECMWF NWP"
        )
    except Exception as error:
        st.error(
            "Unable to load the calibrated V2 spatial inference models."
        )
        st.exception(error)
        st.stop()

    try:
        spatial_records = fetch_all_locations_with_history()
    except Exception as error:
        st.error("Unable to retrieve spatial Mumbai weather history.")
        st.exception(error)
        st.stop()

    valid_records = [r for r in spatial_records if r.get("hourly") is not None]
    if not valid_records:
        st.error("No spatial locations returned enough hourly weather history for multisource inference.")
        st.stop()

    observed_times = [pd.Timestamp(r["datetime"]) for r in valid_records if r.get("datetime") is not None]
    if not observed_times:
        st.error("No valid spatial observation timestamps were returned.")
        st.stop()

    latest_spatial_time = max(observed_times)
    if latest_spatial_time.tzinfo is None:
        latest_spatial_time = latest_spatial_time.tz_localize("Asia/Kolkata")
    spatial_probe_utc = latest_spatial_time.tz_convert("UTC").tz_localize(None)

    try:
        imerg_live_contexts = get_cached_imerg_contexts(
            spatial_probe_utc.normalize().isoformat()
        )
    except Exception as error:
        st.error(
            "Unable to create location-specific IMERG contexts "
            "for the spatial model."
        )
        st.exception(error)
        st.stop()


    satellite_sources = {
        name: context.attrs.get("source", "unknown")
        for name, context in imerg_live_contexts.items()
    }

    live_satellite_count = sum(
        1
        for source in satellite_sources.values()
        if source != "historical_fallback"
    )

    fallback_satellite_count = sum(
        1
        for source in satellite_sources.values()
        if source == "historical_fallback"
    )

    satellite_dates = []

    for context in imerg_live_contexts.values():
        if (
            context is not None
            and not context.empty
            and "date" in context.columns
        ):
            satellite_dates.append(
                pd.to_datetime(context["date"]).max().normalize()
            )

    if satellite_dates:
        satellite_latest_date = min(satellite_dates)
        satellite_age_days = int(
            (
                spatial_probe_utc.normalize()
                - satellite_latest_date
            ).days
        )
    else:
        satellite_age_days = None


    s1, s2, s3 = st.columns(3)

    with s1:
        st.success(
            f"🧠 Spatial model: {spatial_model_type}"
        )

    with s2:
        if fallback_satellite_count == 0:
            st.success(
                f"🛰️ IMERG: live Late Daily "
                f"({live_satellite_count}/9 locations)"
            )
        elif live_satellite_count > 0:
            st.warning(
                f"🛰️ IMERG: mixed live/fallback "
                f"({live_satellite_count}/9 live)"
            )
        else:
            st.warning(
                "🛰️ IMERG: archived fallback context"
            )

    with s3:
        if satellite_age_days is not None:
            st.info(
                f"Satellite context age: "
                f"{satellite_age_days} day(s)"
            )
        else:
            st.info(
                "Satellite context age: unavailable"
            )

    processed_records = []
    errors = []

    # Fetch every location's ECMWF forecast in one request.
    # Open-Meteo supports comma-separated coordinates and returns one
    # response object per location. This removes the nine-request pattern.
    nwp_records = []
    for record in spatial_records:
        location_name = record.get("name", "Unknown")
        hourly = record.get("hourly")
        if hourly is None or hourly.empty:
            continue

        imerg_live = imerg_live_contexts.get(location_name)
        if imerg_live is None or imerg_live.empty:
            continue

        nwp_records.append(record)

    nwp_location_specs = tuple(
        (
            record.get("name", "Unknown"),
            float(record["latitude"]),
            float(record["longitude"]),
        )
        for record in nwp_records
    )

    nwp_results = fetch_spatial_nwp_forecasts(
        nwp_location_specs,
        forecast_hours=48,
    )
    nwp_fallback_locations = []

    # ------------------------------------------------------------
    # 1) Build all live base-model rows first.
    #    The sharded base RF is then predicted ONCE for all locations.
    # ------------------------------------------------------------
    batch_items = []
    base_inputs = []

    for record in spatial_records:
        location_name = record.get("name", "Unknown")
        hourly = record.get("hourly")

        if hourly is None or hourly.empty:
            errors.append(f"{location_name}: no hourly history returned")
            continue

        imerg_live = imerg_live_contexts.get(location_name)

        if imerg_live is None or imerg_live.empty:
            errors.append(
                f"{location_name}: no location-specific "
                "IMERG context returned"
            )
            continue

        try:
            X_live, latest_live = build_live_multisource_features(
                hourly_live=hourly,
                imerg_daily=imerg_live,
                feature_columns=v2_base_features,
            )

            batch_items.append({
                "record": record,
                "location_name": location_name,
                "latest_live": latest_live,
                "X_live": X_live,
            })
            base_inputs.append(X_live)
        except Exception as error:
            errors.append(f"{location_name}: {error}")

    if base_inputs:
        try:
            base_batch = pd.concat(base_inputs, ignore_index=True)
            base_values = predict_v2_batch("base", base_batch)

            if len(base_values) != len(batch_items):
                raise ValueError(
                    "V2 base model returned an unexpected number of predictions."
                )

            for item, value in zip(batch_items, base_values):
                item["base_prediction"] = max(0.0, float(value))
        except Exception as error:
            # Preserve the old fail-soft behavior if a batch prediction
            # unexpectedly fails: retry location-by-location.
            for item in batch_items:
                location_name = item["location_name"]
                try:
                    value = v2_base_model.predict(item["X_live"])[0]
                    item["base_prediction"] = max(0.0, float(value))
                except Exception as location_error:
                    errors.append(
                        f"{location_name}: V2 base prediction failed "
                        f"({location_error})"
                    )
                    item["base_prediction"] = None

            batch_items = [
                item for item in batch_items
                if item.get("base_prediction") is not None
            ]

            if not batch_items:
                st.warning(
                    "The V2 base model batch prediction failed for all "
                    "spatial locations."
                )

    # ------------------------------------------------------------
    # 2) Build the NWP fusion rows.
    #    All successful NWP rows are then predicted ONCE.
    # ------------------------------------------------------------
    fusion_inputs = []
    fusion_items = []

    for item in batch_items:
        location_name = item["location_name"]
        location_nwp = nwp_results.get(location_name)

        if location_nwp is None or location_nwp.empty:
            item["location_nwp"] = None
            item["nwp_correction"] = 0.0
            item["applied_nwp_correction"] = 0.0
            item["predicted_rain"] = item["base_prediction"]
            item["nwp_next_1h"] = np.nan
            item["nwp_next_3h"] = np.nan
            item["nwp_next_6h"] = np.nan
            item["nwp_next_24h"] = np.nan
            item["nwp_status"] = "V2 base fallback"
            nwp_fallback_locations.append(location_name)
            continue

        try:
            nwp_features = build_live_v2_nwp_features(location_nwp)
            nwp_features["base_prediction"] = item["base_prediction"]

            fusion_input = nwp_features[v2_fusion_features]

            item["location_nwp"] = location_nwp
            fusion_inputs.append(fusion_input)
            fusion_items.append(item)
        except Exception as error:
            # Keep the location visible with the calibrated base prediction.
            item["location_nwp"] = None
            item["nwp_correction"] = 0.0
            item["applied_nwp_correction"] = 0.0
            item["predicted_rain"] = item["base_prediction"]
            item["nwp_next_1h"] = np.nan
            item["nwp_next_3h"] = np.nan
            item["nwp_next_6h"] = np.nan
            item["nwp_next_24h"] = np.nan
            item["nwp_status"] = "V2 base fallback"
            nwp_fallback_locations.append(location_name)
            errors.append(f"{location_name}: NWP feature error: {error}")

    if fusion_inputs:
        try:
            fusion_batch = pd.concat(fusion_inputs, ignore_index=True)
            correction_values = predict_v2_batch("fusion", fusion_batch)

            if len(correction_values) != len(fusion_items):
                raise ValueError(
                    "V2 fusion model returned an unexpected number of predictions."
                )

            for item, correction in zip(fusion_items, correction_values):
                location_nwp = item["location_nwp"]
                nwp_correction = float(correction)
                applied_nwp_correction = v2_alpha * nwp_correction

                item["nwp_correction"] = nwp_correction
                item["applied_nwp_correction"] = applied_nwp_correction
                item["predicted_rain"] = max(
                    0.0,
                    item["base_prediction"] + applied_nwp_correction,
                )
                item["nwp_next_1h"] = float(
                    location_nwp.iloc[0]["forecast_next_1h_mm"]
                )
                item["nwp_next_3h"] = float(
                    location_nwp.iloc[0]["forecast_next_3h_mm"]
                )
                item["nwp_next_6h"] = float(
                    location_nwp.iloc[0]["forecast_next_6h_mm"]
                )
                item["nwp_next_24h"] = float(
                    location_nwp.iloc[0]["forecast_next_24h_mm"]
                )
                item["nwp_status"] = "Live ECMWF"
        except Exception as error:
            # Extremely defensive fallback: retain each base prediction
            # instead of making one fusion-model failure break the whole map.
            for item in fusion_items:
                location_name = item["location_name"]
                item["location_nwp"] = None
                item["nwp_correction"] = 0.0
                item["applied_nwp_correction"] = 0.0
                item["predicted_rain"] = item["base_prediction"]
                item["nwp_next_1h"] = np.nan
                item["nwp_next_3h"] = np.nan
                item["nwp_next_6h"] = np.nan
                item["nwp_next_24h"] = np.nan
                item["nwp_status"] = "V2 base fallback"
                nwp_fallback_locations.append(location_name)

            errors.append(f"V2 fusion batch prediction failed: {error}")

    # ------------------------------------------------------------
    # 3) Calculate the risk layer and assemble the original output
    #    schema. No UI fields or model outputs are removed.
    # ------------------------------------------------------------
    for item in batch_items:
        location_name = item["location_name"]
        latest_live = item["latest_live"]
        predicted_rain = float(item["predicted_rain"])
        nwp_correction = float(item["nwp_correction"])
        applied_nwp_correction = float(item["applied_nwp_correction"])

        heavy_warning = predicted_rain >= HEAVY_RAIN_THRESHOLD
        heavy_signal = 1.0 if heavy_warning else 0.0

        risk_result = calculate_flood_risk(
            predicted_rain_mm=predicted_rain,
            heavy_probability=heavy_signal,
            current_rain_mm=float(latest_live.get("rainfall_mm", 0.0)),
            rain_3h_mm=float(latest_live.get("rain_3h", 0.0)),
            rain_6h_mm=float(latest_live.get("rain_6h", 0.0)),
            rain_24h_mm=float(latest_live.get("rain_24h", 0.0)),
        )

        record = item["record"]

        processed_records.append({
            "location": location_name,
            "latitude": float(record["latitude"]),
            "longitude": float(record["longitude"]),
            "datetime": latest_live.get(
                "datetime",
                record.get("datetime"),
            ),
            "rainfall_mm": float(latest_live.get("rainfall_mm", 0.0)),
            "rain_3h": float(latest_live.get("rain_3h", 0.0)),
            "rain_6h": float(latest_live.get("rain_6h", 0.0)),
            "rain_24h": float(latest_live.get("rain_24h", 0.0)),
            "predicted_rain_mm": predicted_rain,
            "base_prediction_mm": item["base_prediction"],
            "nwp_correction_mm": nwp_correction,
            "applied_nwp_correction_mm": applied_nwp_correction,
            "nwp_next_1h_mm": item["nwp_next_1h"],
            "nwp_next_3h_mm": item["nwp_next_3h"],
            "nwp_next_6h_mm": item["nwp_next_6h"],
            "nwp_next_24h_mm": item["nwp_next_24h"],
            "nwp_status": item["nwp_status"],
            "heavy_signal": heavy_signal,
            "heavy_warning": heavy_warning,
            "risk_level": risk_result["risk_level"],
            "risk_score": float(risk_result["score"]),
            "risk_action": risk_result["action"],
            "risk_factors": risk_result["factors"],
        })

    spatial = pd.DataFrame(processed_records)
    if spatial.empty:
        st.error("The final multisource model could not generate a prediction for any spatial location.")
        if errors:
            st.write(errors)
        st.stop()

    # DEM is a terrain-context layer only; it does not change model inference.
    spatial = add_dem_elevation(spatial)

    # --------------------------------------------------------
    # DEM VISUAL OVERLAY
    # --------------------------------------------------------

    (
        dem_image,
        dem_bounds,
        dem_min,
        dem_max,
    ) = build_dem_overlay()



    if errors:
        with st.expander(f"⚠️ {len(errors)} spatial location(s) skipped"):
            for message in errors:
                st.write(f"• {message}")

    if nwp_fallback_locations:
        st.warning(
            f"ECMWF temporarily unavailable for "
            f"{len(nwp_fallback_locations)} location(s); "
            "V2 base fallback used so the Risk Map remains available."
        )

    st.subheader("Mumbai-Wide Risk Summary")
    low = int((spatial["risk_level"] == "LOW").sum())
    moderate = int((spatial["risk_level"] == "MODERATE").sum())
    high = int((spatial["risk_level"] == "HIGH").sum())
    critical = int((spatial["risk_level"] == "CRITICAL").sum())
    warnings = int(spatial["heavy_warning"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("🟢 Low", low)
    with c2: st.metric("🟡 Moderate", moderate)
    with c3: st.metric("🟠 High", high)
    with c4: st.metric("🔴 Critical", critical)
    with c5: st.metric("⚠️ Heavy warnings", warnings)

    st.subheader("Live Mumbai Risk Map")

    def risk_color(level):
        if level == "LOW": return [34, 197, 94, 235]
        if level == "MODERATE": return [234, 179, 8, 235]
        if level == "HIGH": return [249, 115, 22, 240]
        return [239, 68, 68, 245]

    spatial["color"] = spatial["risk_level"].apply(risk_color)
    map_data = spatial[[
        "location", "latitude", "longitude", "risk_level", "risk_score",
        "predicted_rain_mm", "base_prediction_mm",
        "applied_nwp_correction_mm", "nwp_next_1h_mm", "nwp_status",
        "heavy_signal", "heavy_warning",
        "rainfall_mm", "rain_3h", "rain_6h", "rain_24h",
        "elevation_m", "elevation_source", "color",
    ]].copy()
    map_data["radius"] = (3500 + map_data["risk_score"].clip(0, 100) * 28).clip(3500, 6500)
    map_data["risk_score"] = map_data["risk_score"].round(2)
    map_data["predicted_rain_mm"] = map_data["predicted_rain_mm"].round(2)
    map_data["base_prediction_mm"] = map_data["base_prediction_mm"].round(2)
    map_data["applied_nwp_correction_mm"] = map_data["applied_nwp_correction_mm"].round(3)
    map_data["nwp_next_1h_mm"] = map_data["nwp_next_1h_mm"].round(2)
    for column in ["rainfall_mm", "rain_3h", "rain_6h", "rain_24h"]:
        map_data[column] = map_data[column].round(2)
    map_data["elevation_m"] = map_data["elevation_m"].round(1)

    layer = pdk.Layer(
        "ScatterplotLayer", data=map_data,
        get_position=["longitude", "latitude"], get_radius="radius",
        get_fill_color="color", get_line_color=[255, 255, 255, 225],
        line_width_min_pixels=2, pickable=True, auto_highlight=True,
    )
    labels = pdk.Layer(
        "TextLayer", data=map_data,
        get_position=["longitude", "latitude"], get_text="location",
        get_size=13, get_color=[235, 235, 245, 230],
        get_pixel_offset=[0, 40], pickable=False,
    )
    tooltip = {
        "html": (
            "<b>{location}</b><br/>"
            "Risk: <b>{risk_level}</b><br/>"
            "Risk score: {risk_score}<br/>"
            "Base RF prediction: {base_prediction_mm} mm<br/>"
            "Applied NWP correction: {applied_nwp_correction_mm} mm<br/>"
            "Final V2 next-hour rainfall: {predicted_rain_mm} mm<br/>"
            "Raw ECMWF next-hour input: {nwp_next_1h_mm} mm<br/>"
            "NWP status: {nwp_status}<br/>"
            "Heavy-rain signal: {heavy_signal}<br/>"
            "Heavy warning: {heavy_warning}<br/>"
            "Current rainfall: {rainfall_mm} mm<br/>"
            "Rain 3h: {rain_3h} mm<br/>"
            "Rain 6h: {rain_6h} mm<br/>"
            "Rain 24h: {rain_24h} mm<br/>"
            "Elevation: {elevation_m} m"
        ),
        "style": {"backgroundColor": "rgba(15,15,20,.97)", "color": "white"},
    }
    dem_layer = None

    deck = pdk.Deck(
    layers=(
        [dem_layer, layer, labels]
        if dem_layer is not None
        else [layer, labels]
    ),
    initial_view_state=pdk.ViewState(
        latitude=19.0760,
        longitude=72.8777,
        zoom=10.2,
        pitch=35
    ),
    tooltip=tooltip,
)
    try:
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as map_error:
        st.warning(f"Interactive PyDeck map could not render: {map_error}")
        st.map(
            map_data[["latitude", "longitude"]].rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )
        )

    st.markdown("**Risk Legend**")
    st.markdown(
        """
        🟢 LOW
        &nbsp;&nbsp;&nbsp;
        🟡 MODERATE
        &nbsp;&nbsp;&nbsp;
        🟠 HIGH
        &nbsp;&nbsp;&nbsp;
        🔴 CRITICAL
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Location-by-Location Risk")
    location_table = spatial[[
        "location", "risk_level", "risk_score", "predicted_rain_mm",
        "base_prediction_mm", "applied_nwp_correction_mm",
        "nwp_next_1h_mm", "heavy_warning", "rainfall_mm",
        "rain_3h", "rain_6h", "rain_24h",
        "elevation_m", "elevation_source",
    ]].copy().rename(columns={
        "location": "Location", "risk_level": "Risk", "risk_score": "Risk Score",
        "predicted_rain_mm": "Final V2 Next-hour Rain (mm)",
        "base_prediction_mm": "Base RF (mm)",
        "applied_nwp_correction_mm": "Applied NWP Correction (mm)",
        "nwp_next_1h_mm": "Raw ECMWF 1h (mm)",
        "heavy_warning": "Heavy Warning",
        "rainfall_mm": "Current Rain (mm)", "rain_3h": "Rain 3h (mm)",
        "rain_6h": "Rain 6h (mm)", "rain_24h": "Rain 24h (mm)",
        "elevation_m": "Elevation (m)", "elevation_source": "Elevation Source",
    })
    location_table["Risk Score"] = location_table["Risk Score"].round(2)
    location_table["Final V2 Next-hour Rain (mm)"] = location_table["Final V2 Next-hour Rain (mm)"].round(2)
    location_table["Base RF (mm)"] = location_table["Base RF (mm)"].round(2)
    location_table["Applied NWP Correction (mm)"] = location_table["Applied NWP Correction (mm)"].round(3)
    location_table["Raw ECMWF 1h (mm)"] = location_table["Raw ECMWF 1h (mm)"].round(2)
    location_table["Heavy Warning"] = location_table["Heavy Warning"].map(lambda v: "YES" if v else "NO")
    for column in ["Current Rain (mm)", "Rain 3h (mm)", "Rain 6h (mm)", "Rain 24h (mm)"]:
        location_table[column] = location_table[column].round(2)
    location_table = location_table.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    st.dataframe(location_table, use_container_width=True, hide_index=True)

    highest = spatial.sort_values("risk_score", ascending=False).iloc[0]
    st.subheader("Highest Current Risk")
    highest_level = highest["risk_level"]
    highest_score = float(highest["risk_score"])
    icon, alert_function = risk_alert(highest_level)
    alert_function(
        f"{icon} **{highest['location']}** has the highest prototype risk "
        f"score of **{highest_score:.0f} ({highest_level})**."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Final V2 next-hour rain", f"{float(highest['predicted_rain_mm']):.2f} mm")
    with c2: st.metric("Heavy-rain signal", "YES" if highest["heavy_warning"] else "NO")
    with c3: st.metric("Current rain", f"{float(highest['rainfall_mm']):.2f} mm")
    with c4: st.metric("24h rain", f"{float(highest['rain_24h']):.2f} mm")
    st.write(f"**Recommended action:** {highest['risk_action']}")
    st.subheader("Risk Factors")
    highest_factors = highest["risk_factors"]
    if isinstance(highest_factors, list):
        for factor in highest_factors: st.write(f"• {factor}")
    else:
        st.write(f"• {highest_factors}")
    st.divider()
    st.caption(
        "Spatial monitoring uses approximately sampled locations and is not a street-level "
        "flood sensor network. Each location uses the calibrated V2 rainfall pipeline with "
        "location-specific ECMWF NWP residual correction (alpha 0.50); the flood-risk layer "
        "remains a rule-based prototype."
    )

# PAGE 3
# HISTORICAL ERA5 DEMO
# ============================================================

elif page == "📊 Historical ERA5 Demo":

    st.header(
    "📊 Historical ERA5 Demo"
    )

    st.info(
    "Select a real historical observation and "
    "run the complete model pipeline."
    )


    historical = (

        df[
            df["datetime"]
            >= TEST_START_DATE
        ]

        .dropna(
            subset=[
                feature
                for feature
                in features
                if feature
                in df.columns
            ]
        )

        .copy()
    )


    if historical.empty:

        st.warning(
            "No historical records are available."
        )

        st.stop()


    event_type = st.selectbox(

        "Historical event type",

        [
            "All available records",
            "Heavy rainfall events",
            "Extreme rainfall events",
        ],
    )


    if event_type == (
        "Heavy rainfall events"
    ):

        filtered = (
            historical[
                historical[
                    "next_hour_rain"
                ]
                >= HEAVY_RAIN_THRESHOLD
            ]
        )

    elif event_type == (
        "Extreme rainfall events"
    ):

        filtered = (
            historical[
                historical[
                    "next_hour_rain"
                ]
                >= 10.0
            ]
        )

    else:

        filtered = historical


    if filtered.empty:

        st.warning(
            "No records match this filter."
        )

        st.stop()


    timestamps = (

        filtered[
            "datetime"
        ]

        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        .tolist()
    )


    selected_timestamp = (
        st.selectbox(

            "Select observation time",

            timestamps,

            index=
                len(timestamps) - 1,
        )
    )


    selected_datetime = (
        pd.Timestamp(
            selected_timestamp
        )
    )


    selected_rows = (
        filtered[
            filtered[
                "datetime"
            ]
            == selected_datetime
        ]
    )


    if selected_rows.empty:

        st.error(
            "Selected observation not found."
        )

        st.stop()


    row = (
        selected_rows
        .iloc[0]
        .copy()
    )


    # --------------------------------------------------------
    # Selected observation
    # --------------------------------------------------------

    st.subheader(
        "Selected Observation"
    )


    st.info(
        selected_datetime.strftime(
            "%d %B %Y, %H:%M"
        )
    )


    # --------------------------------------------------------
    # Current weather
    # --------------------------------------------------------

    st.subheader(
        "Current Weather"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Rainfall",
            f"{float(row['rainfall_mm']):.2f} mm",
        )


    with c2:

        st.metric(
            "Temperature",
            f"{float(row['temperature_c']):.2f} °C",
        )


    with c3:

        st.metric(
            "Humidity",
            f"{float(row['humidity_pct']):.1f}%",
        )


    with c4:

        st.metric(
            "Pressure",
            f"{float(row['pressure_hpa']):.1f} hPa",
        )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Wind speed",
            f"{float(row['wind_speed_ms']):.2f} m/s",
        )


    with c2:

        st.metric(
            "U10",
            f"{float(row['u10_ms']):.2f} m/s",
        )


    with c3:

        st.metric(
            "V10",
            f"{float(row['v10_ms']):.2f} m/s",
        )


    # --------------------------------------------------------
    # Recent rainfall
    # --------------------------------------------------------

    st.subheader(
        "Recent Rainfall"
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Last 3 hours",
            f"{float(row['rain_3h']):.2f} mm",
        )


    with c2:

        st.metric(
            "Last 6 hours",
            f"{float(row['rain_6h']):.2f} mm",
        )


    with c3:

        st.metric(
            "Last 24 hours",
            f"{float(row['rain_24h']):.2f} mm",
        )


    # --------------------------------------------------------
    # History chart
    # --------------------------------------------------------

    st.subheader(
        "Rainfall — Previous 24 Hours"
    )


    history_chart = (

        df[
            df["datetime"]
            <= selected_datetime
        ]

        .tail(24)

        [
            [
                "datetime",
                "rainfall_mm",
            ]
        ]

        .set_index(
            "datetime"
        )

        .rename(
            columns={
                "rainfall_mm":
                    "Rainfall (mm)"
            }
        )
    )


    st.line_chart(
        history_chart,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    input_df = (
        make_model_input(
            row,
            features,
        )
    )


    predicted = max(
        0.0,
        float(
            rainfall_model
            .predict(
                input_df
            )[0]
        ),
    )


    probability = float(
        np.clip(
            heavy_classifier
            .predict_proba(
                input_df
            )[0, 1],
            0.0,
            1.0,
        )
    )


    warning = (
        probability
        >= PROBABILITY_THRESHOLD
    )


    risk = calculate_risk_from_row(
        row,
        predicted,
        probability,
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.divider()

    st.header(
        "🔮 ML Prediction"
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Predicted next-hour rainfall",
            f"{predicted:.2f} mm",
        )


    with c2:

        st.metric(
            "Heavy-rain probability",
            f"{probability:.1%}",
        )


    with c3:

        st.metric(
            "Flood-risk score",
            f"{risk['score']:.0f}",
        )


    if warning:

        st.error(
            "⚠️ Heavy rainfall warning"
        )

    else:

        st.info(
            "No heavy-rain warning"
        )


    st.write(
        f"**Recommended action:** "
        f"{risk['action']}"
    )


    st.subheader(
        "Risk Factors"
    )


    for factor in risk[
        "factors"
    ]:

        st.write(
            f"• {factor}"
        )


    # --------------------------------------------------------
    # Historical validation
    # --------------------------------------------------------

    st.divider()

    st.header(
        "📈 Historical Validation"
    )


    actual = float(
        row[
            "next_hour_rain"
        ]
    )


    prediction_error = (
        actual
        - predicted
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Actual next-hour rainfall",
            f"{actual:.2f} mm",
        )


    with c2:

        st.metric(
            "Prediction error",
            f"{prediction_error:+.2f} mm",
        )


    with c3:

        st.metric(
            "Absolute error",
            f"{abs(prediction_error):.2f} mm",
        )


# ============================================================
# PAGE 4
# CUSTOM WEATHER INPUT
# ============================================================

elif page == "🧪 Custom Weather Input":

    st.header(
        "🧪 Custom Weather Prediction"
    )

    st.info(
        "Enter weather conditions manually "
        "to simulate a next-hour forecast."
    )


    # --------------------------------------------------------
    # Main weather inputs
    # --------------------------------------------------------

    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        rainfall = st.number_input(
            "Current rainfall (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )


        temperature = st.number_input(
            "Temperature (°C)",
            value=28.0,
            step=0.1,
        )


        dewpoint = st.number_input(
            "Dew point (°C)",
            value=24.0,
            step=0.1,
        )


        pressure = st.number_input(
            "Pressure (hPa)",
            value=1005.0,
            step=0.1,
        )


    with c2:

        u10 = st.number_input(
            "U10 wind (m/s)",
            value=0.0,
            step=0.1,
        )


        v10 = st.number_input(
            "V10 wind (m/s)",
            value=0.0,
            step=0.1,
        )


        wind_speed = st.number_input(
            "Wind speed (m/s)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )


        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1,
        )


    with c3:

        rain_3h = st.number_input(
            "Rainfall last 3h (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )


        rain_6h = st.number_input(
            "Rainfall last 6h (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )


        rain_24h = st.number_input(
            "Rainfall last 24h (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )


    # --------------------------------------------------------
    # Forecast time
    # --------------------------------------------------------

    forecast_time = (
        st.datetime_input(
            "Forecast datetime",
            value=
                pd.Timestamp
                .now()
                .to_pydatetime(),
        )
    )


    month = int(
        forecast_time.month
    )


    hour = int(
        forecast_time.hour
    )


    day_of_year = int(
        forecast_time
        .timetuple()
        .tm_yday
    )


    hour_sin = np.sin(
        2
        * np.pi
        * hour
        / 24
    )


    hour_cos = np.cos(
        2
        * np.pi
        * hour
        / 24
    )


    doy_sin = np.sin(
        2
        * np.pi
        * day_of_year
        / 365.25
    )


    doy_cos = np.cos(
        2
        * np.pi
        * day_of_year
        / 365.25
    )


    # --------------------------------------------------------
    # Rainfall lags
    # --------------------------------------------------------

    st.subheader(
        "Recent Rainfall"
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        lag_1 = st.number_input(
            "Rainfall 1h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


        lag_2 = st.number_input(
            "Rainfall 2h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


    with c2:

        lag_3 = st.number_input(
            "Rainfall 3h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


        lag_6 = st.number_input(
            "Rainfall 6h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


    with c3:

        lag_12 = st.number_input(
            "Rainfall 12h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


        lag_24 = st.number_input(
            "Rainfall 24h ago (mm)",
            min_value=0.0,
            value=rainfall,
            step=0.1,
        )


    # --------------------------------------------------------
    # Custom dictionary
    # --------------------------------------------------------

    custom = {

        "rainfall_mm":
            rainfall,

        "temperature_c":
            temperature,

        "dewpoint_c":
            dewpoint,

        "pressure_hpa":
            pressure,

        "u10_ms":
            u10,

        "v10_ms":
            v10,

        "wind_speed_ms":
            wind_speed,

        "humidity_pct":
            humidity,

        "rain_3h":
            rain_3h,

        "rain_6h":
            rain_6h,

        "rain_24h":
            rain_24h,

        "rainfall_lag_1h":
            lag_1,

        "rainfall_lag_2h":
            lag_2,

        "rainfall_lag_3h":
            lag_3,

        "rainfall_lag_6h":
            lag_6,

        "rainfall_lag_12h":
            lag_12,

        "rainfall_lag_24h":
            lag_24,

        "month":
            month,

        "hour":
            hour,

        "day_of_year":
            day_of_year,

        "hour_sin":
            hour_sin,

        "hour_cos":
            hour_cos,

        "doy_sin":
            doy_sin,

        "doy_cos":
            doy_cos,
    }


    input_df = (
        pd.DataFrame(
            [custom]
        )
    )


    missing_custom = [

        feature

        for feature
        in features

        if feature
        not in input_df.columns
    ]


    if missing_custom:

        st.error(
            "The trained model expects features "
            "that are not available in the custom "
            f"input: {missing_custom}"
        )

        st.stop()


    input_df = (
        input_df[
            features
        ]
    )


    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Next Hour",
        type="primary",
        use_container_width=True,
    ):

        predicted = max(
            0.0,
            float(
                rainfall_model
                .predict(
                    input_df
                )[0]
            ),
        )


        probability = float(
            np.clip(
                heavy_classifier
                .predict_proba(
                    input_df
                )[0, 1],
                0.0,
                1.0,
            )
        )


        warning = (
            probability
            >= PROBABILITY_THRESHOLD
        )


        custom_row = pd.Series(
            custom
        )


        risk = calculate_risk_from_row(
            custom_row,
            predicted,
            probability,
        )


        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.divider()

        st.header(
            "Prediction Results"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(
                "Next-hour rainfall",
                f"{predicted:.2f} mm",
            )


        with c2:

            st.metric(
                "Heavy-rain probability",
                f"{probability:.1%}",
            )


        with c3:

            st.metric(
                "Flood-risk score",
                f"{risk['score']:.0f}",
            )


        icon, alert_function = (
            risk_alert(
                risk["risk_level"]
            )
        )


        alert_function(
            f"{icon} Risk: "
            f"{risk['risk_level']}"
        )


        if warning:

            st.error(
                "⚠️ Heavy rainfall warning"
            )

        else:

            st.info(
                "No heavy-rain warning"
            )


        st.write(
            f"**Recommended action:** "
            f"{risk['action']}"
        )


        st.subheader(
            "Risk Factors"
        )


        for factor in risk[
            "factors"
        ]:

            st.write(
                f"• {factor}"
            )


# ============================================================
# PAGE 5
# CALIBRATED V2 MODEL EVALUATION
# ============================================================

elif page == "📈 V2 Model Evaluation":

    st.header(
        "📈 Calibrated V2 Model Evaluation"
    )

    st.caption(
        "V2 two-stage rainfall forecasting: V2 Base Random Forest + ECMWF NWP "
        "residual fusion, followed by the validation-selected calibration factor. "
        "This page reports the untouched 2025 holdout and walk-forward validation results."
    )

    # --------------------------------------------------------
    # V2 artifacts
    # --------------------------------------------------------

    st.subheader("🧠 V2 Architecture")

    st.code(
        "ERA5 + NASA IMERG\n"
        "        ↓\n"
        "V2 Base Random Forest\n"
        "        ↓\n"
        "Base rainfall prediction\n"
        "        +\n"
        "ECMWF IFS HRES NWP\n"
        "        ↓\n"
        "NWP Residual Fusion Random Forest\n"
        "        ↓\n"
        "Raw NWP correction\n"
        "        ↓\n"
        "× calibration α = 0.50\n"
        "        ↓\n"
        "Final calibrated next-hour rainfall",
        language="text",
    )

    # Known, previously validated V2 results. Artifact files are loaded
    # below when present; these values keep the evaluation page informative
    # on cloud deployments where only tracked evaluation artifacts exist.
    fallback_results = {
        "base": {
            "mae": 0.186232,
            "rmse": 0.371330,
            "r2": 0.848032,
        },
        "full": {
            "mae": 0.186015,
            "rmse": 0.375026,
            "r2": 0.844991,
        },
        "calibrated": {
            "mae": 0.182612,
            "rmse": 0.368781,
            "r2": 0.850111,
        },
        "heavy": {
            "count": 22,
            "base_mae": 1.316923,
            "base_rmse": 1.986454,
            "full_mae": 1.259722,
            "full_rmse": 1.918167,
            "cal_mae": 1.283514,
            "cal_rmse": 1.944743,
        },
    }

    calibration = {
        "nwp_correction_alpha": 0.50,
    }

    if V2_CALIBRATION_PATH.exists():
        try:
            with open(
                V2_CALIBRATION_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                loaded_calibration = json.load(file)
            if isinstance(loaded_calibration, dict):
                calibration.update(loaded_calibration)
        except Exception as error:
            st.warning(
                f"V2 calibration JSON could not be read; using validated fallback values. {error}"
            )

    # Optional calibrated metrics artifact.
    if V2_CALIBRATED_METRICS_PATH.exists():
        try:
            with open(
                V2_CALIBRATED_METRICS_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                artifact_metrics = json.load(file)

            # Accept the common nested layouts produced by the artifact script.
            def _pick_metric(container, *keys, default):
                current = container
                for key in keys:
                    if not isinstance(current, dict) or key not in current:
                        return default
                    current = current[key]
                try:
                    return float(current)
                except (TypeError, ValueError):
                    return default

            fallback_results["base"]["mae"] = _pick_metric(
                artifact_metrics, "base", "MAE_mm", default=fallback_results["base"]["mae"]
            )
            fallback_results["base"]["rmse"] = _pick_metric(
                artifact_metrics, "base", "RMSE_mm", default=fallback_results["base"]["rmse"]
            )
            fallback_results["base"]["r2"] = _pick_metric(
                artifact_metrics, "base", "R2", default=fallback_results["base"]["r2"]
            )
            fallback_results["calibrated"]["mae"] = _pick_metric(
                artifact_metrics, "calibrated", "MAE_mm", default=fallback_results["calibrated"]["mae"]
            )
            fallback_results["calibrated"]["rmse"] = _pick_metric(
                artifact_metrics, "calibrated", "RMSE_mm", default=fallback_results["calibrated"]["rmse"]
            )
            fallback_results["calibrated"]["r2"] = _pick_metric(
                artifact_metrics, "calibrated", "R2", default=fallback_results["calibrated"]["r2"]
            )
        except Exception:
            pass

    alpha = float(
        calibration.get(
            "nwp_correction_alpha",
            0.50,
        )
    )

    # --------------------------------------------------------
    # Headline metrics
    # --------------------------------------------------------

    st.subheader("2025 Untouched Holdout")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Calibrated V2 MAE",
            f"{fallback_results['calibrated']['mae']:.4f} mm",
        )

    with c2:
        st.metric(
            "Calibrated V2 RMSE",
            f"{fallback_results['calibrated']['rmse']:.4f} mm",
        )

    with c3:
        st.metric(
            "Calibrated V2 R²",
            f"{fallback_results['calibrated']['r2']:.4f}",
        )

    with c4:
        st.metric(
            "Calibration α",
            f"{alpha:.2f}",
        )

    st.info(
        "The calibration factor α was selected using walk-forward validation and then "
        "applied unchanged to the untouched 2025 holdout. It was not chosen from the "
        "2025 holdout itself."
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.subheader("V2 Stage Comparison")

    comparison = pd.DataFrame(
        [
            {
                "Model": "V2 Base RF",
                "MAE (mm)": fallback_results["base"]["mae"],
                "RMSE (mm)": fallback_results["base"]["rmse"],
                "R²": fallback_results["base"]["r2"],
            },
            {
                "Model": "V2 Two-Stage RF (α=1.00)",
                "MAE (mm)": fallback_results["full"]["mae"],
                "RMSE (mm)": fallback_results["full"]["rmse"],
                "R²": fallback_results["full"]["r2"],
            },
            {
                "Model": f"V2 Calibrated RF (α={alpha:.2f})",
                "MAE (mm)": fallback_results["calibrated"]["mae"],
                "RMSE (mm)": fallback_results["calibrated"]["rmse"],
                "R²": fallback_results["calibrated"]["r2"],
            },
        ]
    )

    st.dataframe(
        comparison.round(6),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Improvement vs base
    # --------------------------------------------------------

    mae_improvement = (
        100.0
        * (
            fallback_results["base"]["mae"]
            - fallback_results["calibrated"]["mae"]
        )
        / fallback_results["base"]["mae"]
    )

    rmse_improvement = (
        100.0
        * (
            fallback_results["base"]["rmse"]
            - fallback_results["calibrated"]["rmse"]
        )
        / fallback_results["base"]["rmse"]
    )

    i1, i2 = st.columns(2)

    with i1:
        st.metric(
            "MAE change vs V2 Base",
            f"{mae_improvement:.3f}%",
        )

    with i2:
        st.metric(
            "RMSE change vs V2 Base",
            f"{rmse_improvement:.3f}%",
        )

    # --------------------------------------------------------
    # Walk-forward calibration evidence
    # --------------------------------------------------------

    st.subheader("🧪 Walk-Forward Validation Used for Calibration")

    validation_alpha_rows = pd.DataFrame(
        [
            {"α": 0.50, "Walk-forward MAE (mm)": 0.185404, "Walk-forward RMSE (mm)": 0.360349},
            {"α": 0.55, "Walk-forward MAE (mm)": 0.185348, "Walk-forward RMSE (mm)": 0.360422},
            {"α": 1.00, "Walk-forward MAE (mm)": 0.187770, "Walk-forward RMSE (mm)": 0.364638},
        ]
    )

    st.dataframe(
        validation_alpha_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Combined usable walk-forward validation covered 5,804 rows. "
        "The operational α=0.50 setting was selected using validation RMSE, "
        "then frozen before evaluating the 2025 holdout."
    )

    # --------------------------------------------------------
    # Heavy-rain subset
    # --------------------------------------------------------

    st.subheader("🌧️ Heavy-Rain Subset (Actual Rain ≥ 5 mm/h)")

    heavy_table = pd.DataFrame(
        [
            {
                "Model": "V2 Base RF",
                "MAE (mm)": fallback_results["heavy"]["base_mae"],
                "RMSE (mm)": fallback_results["heavy"]["base_rmse"],
                "Samples": fallback_results["heavy"]["count"],
            },
            {
                "Model": "V2 Two-Stage RF (α=1.00)",
                "MAE (mm)": fallback_results["heavy"]["full_mae"],
                "RMSE (mm)": fallback_results["heavy"]["full_rmse"],
                "Samples": fallback_results["heavy"]["count"],
            },
            {
                "Model": f"V2 Calibrated RF (α={alpha:.2f})",
                "MAE (mm)": fallback_results["heavy"]["cal_mae"],
                "RMSE (mm)": fallback_results["heavy"]["cal_rmse"],
                "Samples": fallback_results["heavy"]["count"],
            },
        ]
    )

    st.dataframe(
        heavy_table.round(6),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Only 22 2025 holdout observations met the ≥5 mm/h threshold, so heavy-rain "
        "metrics are a small-sample diagnostic rather than a broad event-level estimate."
    )

    # --------------------------------------------------------
    # Holdout prediction explorer
    # --------------------------------------------------------

    st.subheader("🔎 2025 Holdout Prediction Explorer")

    v2_predictions = None

    if V2_CALIBRATED_PREDICTIONS_PATH.exists():
        try:
            v2_predictions = pd.read_csv(
                V2_CALIBRATED_PREDICTIONS_PATH,
                parse_dates=["datetime"],
            )
        except Exception as error:
            st.warning(
                f"Unable to read the V2 calibrated prediction artifact: {error}"
            )

    if v2_predictions is None:
        st.info(
            "The V2 prediction artifact is not present in this deployment. "
            "The validated holdout metrics above remain available."
        )
    else:
        # Normalize common column names.
        actual_col = next(
            (
                col for col in [
                    "actual_next_hour_rain",
                    "actual",
                    "next_hour_rain",
                ]
                if col in v2_predictions.columns
            ),
            None,
        )
        base_col = next(
            (
                col for col in [
                    "base_prediction",
                    "v2_base_prediction",
                ]
                if col in v2_predictions.columns
            ),
            None,
        )
        calibrated_col = next(
            (
                col for col in [
                    "calibrated_prediction",
                    "calibrated_v2_prediction",
                    "v2_calibrated_prediction",
                ]
                if col in v2_predictions.columns
            ),
            None,
        )

        if actual_col is None or calibrated_col is None:
            st.warning(
                "The V2 prediction artifact does not contain the expected actual/calibrated prediction columns."
            )
        else:
            chart_cols = ["datetime", actual_col, calibrated_col]
            chart_labels = {
                actual_col: "Actual",
                calibrated_col: "Calibrated V2",
            }

            if base_col is not None:
                chart_cols.insert(2, base_col)
                chart_labels[base_col] = "V2 Base RF"

            plot_count = st.slider(
                "Number of holdout observations",
                min_value=100,
                max_value=min(1000, len(v2_predictions)),
                value=min(500, len(v2_predictions)),
                step=100,
                key="v2_eval_plot_count",
            )

            chart = (
                v2_predictions[chart_cols]
                .tail(plot_count)
                .set_index("datetime")
                .rename(columns=chart_labels)
            )

            st.line_chart(
                chart,
                use_container_width=True,
            )

            display_cols = [
                col for col in [
                    "datetime",
                    actual_col,
                    base_col,
                    calibrated_col,
                ]
                if col is not None and col in v2_predictions.columns
            ]

            st.dataframe(
                v2_predictions[display_cols].tail(50),
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # Integrity statement
    # --------------------------------------------------------

    with st.expander("🔐 Evaluation Integrity"):
        st.write(
            "V1 artifacts remain unchanged. V2 calibration was selected on walk-forward "
            "validation rather than the final 2025 holdout. The 2025 holdout is therefore "
            "used only for the final reported V2 evaluation."
        )


# ============================================================
# PAGE 6
# FINAL MULTISOURCE HEAVY-RAIN EVENTS
# ============================================================

elif page == "⚠️ Heavy-Rain Events":

    st.header("⚠️ Final Multisource Heavy-Rain Event Analysis")

    st.info(
        "This page uses the final multisource Random Forest predictions from the same "
        "2025 holdout used for the final model evaluation. A heavy-rain episode is a "
        f"continuous run of hourly observations at or above {HEAVY_RAIN_THRESHOLD:.1f} mm/hour. "
        "Consecutive heavy hours are grouped into one episode, so a multi-hour storm is "
        "not counted as many separate events."
    )

    try:
        final_test = load_final_test_predictions()
    except Exception as error:
        st.error("Unable to load final multisource test predictions.")
        st.exception(error)
        st.stop()

    episode_summary = build_heavy_rain_episode_summary(final_test)
    actual_episodes = episode_summary["actual_episodes"]
    detected_episodes = episode_summary["detected_episodes"]
    missed_episodes = episode_summary["missed_episodes"]
    false_alarm_episodes = episode_summary["false_alarm_episodes"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Actual heavy-rain episodes",
            episode_summary["actual_episode_count"],
        )
    with c2:
        st.metric(
            "Episodes detected",
            episode_summary["detected_episode_count"],
        )
    with c3:
        st.metric(
            "Episodes missed",
            episode_summary["missed_episode_count"],
        )
    with c4:
        st.metric(
            "False-alarm episodes",
            episode_summary["false_alarm_episode_count"],
        )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Episode detection rate",
            f"{episode_summary['episode_detection_rate']:.1%}",
        )
    with d2:
        st.metric(
            "Actual heavy hours",
            int(final_test["actual_heavy"].sum()),
        )
    with d3:
        st.metric(
            "Detected heavy hours",
            int(
                (
                    (final_test["actual_heavy"] == 1)
                    & (final_test["predicted_heavy"] == 1)
                ).sum()
            ),
        )

    st.caption(
        "Episode detection means at least one predicted heavy-rain hour overlaps the "
        "actual episode. Hour-level precision/recall/F1 are shown on the Model Evaluation page."
    )

    st.subheader("🌧️ Actual Heavy-Rain Episodes")

    if not actual_episodes.empty:
        actual_episode_table = actual_episodes.copy()
        actual_episode_table["Start"] = actual_episode_table["Start"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        actual_episode_table["End"] = actual_episode_table["End"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        st.dataframe(
            actual_episode_table.round(3),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No heavy-rain episodes occurred in this holdout.")

    with st.expander("View every heavy-rain hour"):
        heavy_hours = final_test[final_test["actual_heavy"] == 1][
            [
                "datetime",
                "next_hour_rain",
                "predicted_rain_mm",
                "predicted_heavy",
                "absolute_error_mm",
            ]
        ].rename(
            columns={
                "datetime": "Time",
                "next_hour_rain": "Actual Rain (mm/h)",
                "predicted_rain_mm": "Predicted Rain (mm/h)",
                "predicted_heavy": "Warning",
                "absolute_error_mm": "Absolute Error (mm)",
            }
        )
        st.dataframe(
            heavy_hours.round(3),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("❌ Missed Heavy-Rain Episodes")

    if not missed_episodes.empty:
        missed_table = missed_episodes.copy()
        missed_table["Start"] = missed_table["Start"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        missed_table["End"] = missed_table["End"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        st.dataframe(
            missed_table[
                [
                    "Episode",
                    "Start",
                    "End",
                    "Duration (hours)",
                    "Actual max rain (mm/h)",
                    "Max predicted rain (mm/h)",
                ]
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No heavy-rain episodes were completely missed in this holdout.")

    st.subheader("⚠️ False Heavy-Rain Alarm Episodes")

    if not false_alarm_episodes.empty:
        false_alarm_table = false_alarm_episodes.copy()
        false_alarm_table["Start"] = false_alarm_table["Start"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        false_alarm_table["End"] = false_alarm_table["End"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        st.dataframe(
            false_alarm_table.round(3),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No false heavy-rain alarm episodes occurred in this holdout.")


# ============================================================
# PAGE 7
# FINAL MULTISOURCE MODEL
# ============================================================

elif page == "🧠 V1 Historical Baseline":

    st.header(
        "🧠 V1 Historical Multisource Baseline"
    )

    st.caption(
        "Historical V1 baseline trained on 2010–2025 ERA5 + NASA GPM IMERG "
        "Final Daily V07 data. This is the reference model; the live system "
        "uses the calibrated V2 Base RF + ECMWF NWP fusion pipeline."
    )

    required_artifacts = [
        FINAL_MODEL_FEATURES_PATH,
        FINAL_MODEL_METADATA_PATH,
        FINAL_MODEL_TRAINING_METRICS_PATH,
        FINAL_MODEL_PREDICTIONS_PATH,
        FINAL_MODEL_IMPORTANCE_PATH,
        FINAL_MULTISOURCE_DATA_PATH,
    ]

    inference_artifact_available = (
        FINAL_MODEL_BUNDLE_PATH.exists()
        or FINAL_MODEL_XGB_PATH.exists()
    )

    if not inference_artifact_available:
        required_artifacts.append(FINAL_MODEL_XGB_PATH)

    missing_artifacts = [
        str(path)
        for path in required_artifacts
        if not path.exists()
    ]

    if missing_artifacts:
        # The two generated evaluation artifacts below are allowed to be rebuilt
        # from the tracked final model + holdout data at runtime. Other missing
        # artifacts are still fatal because the page cannot function without them.
        generated_at_runtime = {
            str(FINAL_MODEL_TRAINING_METRICS_PATH),
            str(FINAL_MODEL_PREDICTIONS_PATH),
        }
        fatal_missing = [
            path
            for path in missing_artifacts
            if path not in generated_at_runtime
        ]

        if fatal_missing:
            st.error(
                "One or more final multisource model artifacts are missing."
            )
            for path in fatal_missing:
                st.write(path)
            st.stop()

    try:
        (
            final_bundle,
            final_metadata,
            final_feature_info,
        ) = load_final_multisource_model()

        if FINAL_MODEL_TRAINING_METRICS_PATH.exists():
            with open(
                FINAL_MODEL_TRAINING_METRICS_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                final_metrics = json.load(file)
        else:
            # The detailed comparison JSON is optional for deployment.
            # Reuse the frozen holdout metrics embedded in final_metadata so
            # the historical page remains usable without fabricating values.
            fallback_overall = (
                final_metadata
                .get("metrics_2025_holdout", {})
                .get("random_forest", {})
                .get("overall", {})
            )
            fallback_heavy = (
                final_metadata
                .get("metrics_2025_holdout", {})
                .get("random_forest", {})
                .get("heavy", {})
            )
            final_metrics = {
                "models": {
                    "multisource_random_forest": {
                        "overall": fallback_overall,
                        "heavy": fallback_heavy,
                    }
                }
            }
            st.info(
                "The archived comparative metrics JSON is not bundled with this deployment; "
                "the frozen holdout metrics stored in the model metadata are being used instead."
            )

        final_predictions = load_final_test_predictions()

        final_importance = pd.read_csv(
            FINAL_MODEL_IMPORTANCE_PATH
        )

    except Exception as error:
        st.error(
            "Unable to load the final multisource model."
        )
        st.exception(error)
        st.stop()

    deployed_model_type = final_bundle.get(
        "model_type",
        "Random Forest",
    )
    if deployed_model_type.startswith("XGBoost"):
        st.info(
            "Frozen Random Forest evaluation is shown below. "
            "Cloud inference uses the tracked multisource XGBoost backup model "
            "because the exact Random Forest bundle is too large for GitHub."
        )
    else:
        st.success(
            "Final multisource Random Forest model loaded successfully."
        )

    model_metrics = (
        final_metadata
        .get("metrics_2025_holdout", {})
        .get("random_forest", {})
        .get("overall", {})
    )

    heavy_metrics = (
        final_metadata
        .get("metrics_2025_holdout", {})
        .get("random_forest", {})
        .get("heavy", {})
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "MAE",
            f"{model_metrics.get('MAE_mm', 0):.4f} mm",
        )

    with c2:
        st.metric(
            "RMSE",
            f"{model_metrics.get('RMSE_mm', 0):.4f} mm",
        )

    with c3:
        st.metric(
            "R²",
            f"{model_metrics.get('R2', 0):.4f}",
        )

    with c4:
        st.metric(
            "Heavy-rain MAE",
            f"{heavy_metrics.get('MAE_mm', 0):.4f} mm",
        )

    # --------------------------------------------------------
    # Chronological evaluation
    # --------------------------------------------------------

    st.subheader(
        "📊 Chronological Evaluation"
    )

    chronology = final_metadata.get(
        "chronological_evaluation",
        {},
    )

    e1, e2 = st.columns(2)

    with e1:
        st.write(
            "**Training period**"
        )
        st.write(
            f"{chronology.get('train_start', 'N/A')}"
            " → "
            f"{chronology.get('train_end', 'N/A')}"
        )
        st.write(
            f"Training observations: "
            f"{chronology.get('train_rows', 0):,}"
        )

    with e2:
        st.write(
            "**2025 holdout period**"
        )
        st.write(
            f"{chronology.get('test_start', 'N/A')}"
            " → "
            f"{chronology.get('test_end', 'N/A')}"
        )
        st.write(
            f"Test observations: "
            f"{chronology.get('test_rows', 0):,}"
        )

    gap_hours = chronology.get(
        "gap_hours_between_train_and_test",
        0,
    )

    if gap_hours:
        st.info(
            f"The available ERA5 record contains a "
            f"{gap_hours:,}-hour gap between the end of training "
            f"and the available 2025 holdout. The reported test "
            f"period is therefore the actual available 2025 period."
        )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader(
        "🌐 V1 Historical Multisource Architecture"
    )

    st.code(
        """
ERA5 hourly weather
    │
    ├── rainfall
    ├── temperature
    ├── dew point
    ├── pressure
    ├── u10 / v10 wind
    ├── rainfall lags
    └── rolling rainfall windows
             │
             ├──────────────────────┐
             │                      │
             │              NASA GPM IMERG
             │              Final Daily V07
             │
             │              previous-day rainfall
             │              3-day / 7-day rainfall
             │              quality + uncertainty
             │                      │
             └──────────────┬───────┘
                            ↓
                     34 model features
                            ↓
                 Random Forest Regressor
                            ↓
                 Next-hour rainfall
                            ↓
                   Flood-risk layer
        """,
        language="text",
    )

    st.info(
        "V1 uses IMERG as a historical daily auxiliary source. "
        "Only previously completed daily IMERG periods are used for "
        "the model features, avoiding temporal leakage. This page is the "
        "historical baseline and is not the live production V2 inference path."
    )

    if deployed_model_type.startswith("XGBoost"):
        st.caption(
            "Deployment note: the frozen Random Forest remains the reference model for "
            "the reported holdout metrics; this cloud build uses the smaller XGBoost "
            "backup artifact for interactive inference."
        )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    st.subheader(
        "🧩 V1 Frozen Model Features"
    )

    feature_columns = final_feature_info.get(
        "feature_columns",
        [],
    )

    st.write(
        f"Total model features: **{len(feature_columns)}**"
    )

    with st.expander(
        "View feature list"
    ):
        feature_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Source": [
                    "NASA IMERG"
                    if feature.startswith("imerg_")
                    else "ERA5"
                    for feature in feature_columns
                ],
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader(
        "📈 Feature Importance"
    )

    if not final_importance.empty:

        cols = [
            col
            for col in [
                "feature",
                "rf_importance",
                "xgb_importance",
            ]
            if col in final_importance.columns
        ]

        st.dataframe(
            final_importance[cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Historical prediction explorer
    # --------------------------------------------------------

    st.subheader(
        "🔎 V1 Historical Prediction Explorer"
    )

    st.caption(
        "Select a historical date and hour and run the frozen "
        "34-feature Random Forest model on that observation."
    )

    try:
        historical = load_final_multisource_dataset()
    except Exception as error:
        st.error(
            "Unable to load the final multisource dataset."
        )
        st.exception(error)
        st.stop()

    min_date = historical["datetime"].dt.date.min()
    max_date = historical["datetime"].dt.date.max()

    selected_date = st.date_input(
        "Historical date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="multisource_date",
    )

    day_rows = historical[
        historical["datetime"].dt.date
        == selected_date
    ].copy()

    if day_rows.empty:
        st.warning(
            "No hourly record is available for that date."
        )
    else:

        hours = day_rows["datetime"].dt.strftime(
            "%H:%M"
        ).tolist()

        selected_hour = st.selectbox(
            "Hour",
            hours,
            index=len(hours) - 1,
            key="multisource_hour",
        )

        selected_dt = pd.Timestamp(
            f"{selected_date} {selected_hour}"
        )

        selected_row = historical[
            historical["datetime"]
            == selected_dt
        ]

        if selected_row.empty:
            st.warning(
                "Selected timestamp was not found."
            )
        else:

            row = selected_row.iloc[0]

            X_selected = pd.DataFrame(
                [
                    [
                        pd.to_numeric(
                            row.get(
                                feature,
                                np.nan,
                            ),
                            errors="coerce",
                        )
                        for feature in feature_columns
                    ]
                ],
                columns=feature_columns,
            )

            if X_selected.isna().any().any():

                bad = X_selected.columns[
                    X_selected.isna().iloc[0]
                ].tolist()

                st.warning(
                    "Selected observation has missing model "
                    f"features: {bad}"
                )

            else:

                predicted = max(
                    0.0,
                    float(
                        final_bundle["model"].predict(
                            X_selected
                        )[0]
                    ),
                )

                actual = float(
                    row[TARGET]
                )

                p1, p2, p3 = st.columns(3)

                with p1:
                    st.metric(
                        "Predicted next-hour rainfall",
                        f"{predicted:.2f} mm",
                    )

                with p2:
                    st.metric(
                        "Actual next-hour rainfall",
                        f"{actual:.2f} mm",
                    )

                with p3:
                    st.metric(
                        "Absolute error",
                        f"{abs(actual - predicted):.2f} mm",
                    )

                # Satellite context
                satellite_cols = [
                    "imerg_prev_day_mm",
                    "imerg_3day_sum_mm",
                    "imerg_7day_sum_mm",
                    "imerg_3day_max_mm",
                    "imerg_7day_max_mm",
                    "imerg_prev_day_random_error",
                    "imerg_prev_day_quality_ratio",
                    "imerg_prev_day_liquid_probability",
                ]

                available_satellite = [
                    col
                    for col in satellite_cols
                    if col in row.index
                ]

                st.write(
                    "**NASA IMERG historical context**"
                )

                satellite_df = pd.DataFrame(
                    {
                        "Feature": available_satellite,
                        "Value": [
                            row[col]
                            for col in available_satellite
                        ],
                    }
                )

                st.dataframe(
                    satellite_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # --------------------------------------------------------
    # 2025 holdout predictions
    # --------------------------------------------------------

    st.subheader(
        "🧪 2025 Holdout Predictions"
    )

    if not final_predictions.empty:

        display_predictions = final_predictions.copy()

        display_predictions[
            "absolute_error"
        ] = (
            display_predictions[
                "actual_next_hour_rain"
            ]
            - display_predictions[
                "rf_multisource_prediction"
            ]
        ).abs()

        st.dataframe(
            display_predictions.tail(50),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.subheader(
        "⚖️ V1 Model Comparison"
    )

    comparison_rows = []

    for model_name, model_info in (
        final_metrics
        .get("models", {})
        .items()
    ):

        overall = model_info.get(
            "overall",
            {},
        )

        comparison_rows.append(
            {
                "Model": model_name.replace(
                    "_",
                    " ",
                ).title(),
                "MAE (mm)": overall.get(
                    "MAE_mm",
                    np.nan,
                ),
                "RMSE (mm)": overall.get(
                    "RMSE_mm",
                    np.nan,
                ),
                "R²": overall.get(
                    "R2",
                    np.nan,
                ),
            }
        )

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(
                comparison_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


with st.expander(
    "ℹ️ Model & System Information"
):

    st.write(
        "V1 historical baseline: Random Forest Regressor trained on ERA5 + IMERG"
    )


    st.write(
        "V2 live rainfall model: Base Random Forest + ECMWF NWP residual Random Forest, calibrated with alpha 0.50"
    )


    st.write(
        f"Heavy-rain threshold: "
        f"{HEAVY_RAIN_THRESHOLD:.1f} mm/hour"
    )


    st.write(
        f"Probability cutoff: "
        f"{PROBABILITY_THRESHOLD:.2f}"
    )


    st.write(
        f"Model features: "
        f"{len(features)}"
    )


    st.write(
        "Forecast horizon: 1 hour"
    )


    st.write(
        "V1 historical baseline evaluation is documented on the V1 Historical Baseline page; Live Mumbai and the spatial map use the calibrated V2 + ECMWF NWP pipeline."
    )


    st.write(
        "Live weather: Open-Meteo"
    )


    st.write(
        "Flood-risk layer: rule-based prototype"
    )
