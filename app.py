# ============================================================
# MUMBAI RAINFALL & FLOOD EARLY WARNING SYSTEM
# Complete Streamlit Application
# ============================================================

from pathlib import Path
import json

import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

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
from src.live_weather import fetch_live_mumbai_weather
from src.spatial_weather import fetch_all_locations
from src.live_multisource import (
    build_live_multisource_features,
    fetch_imerg_live_context,
)


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
def load_final_multisource_model():

    # The exact Random Forest bundle is intentionally kept out of GitHub
    # because it is ~2.68 GB. For cloud deployment, use the tracked
    # multisource XGBoost backup model for inference while retaining the
    # frozen Random Forest metrics/predictions for evaluation.
    if FINAL_MODEL_BUNDLE_PATH.exists():
        with open(
            FINAL_MODEL_BUNDLE_PATH,
            "rb",
        ) as file:
            bundle = joblib.load(file)
        model_type = "Random Forest"
    elif FINAL_MODEL_XGB_PATH.exists():
        cloud_model = xgb.XGBRegressor()
        cloud_model.load_model(str(FINAL_MODEL_XGB_PATH))
        bundle = {
            "model": cloud_model,
            "feature_columns": None,
            "model_type": "XGBoost backup (cloud inference)",
        }
        model_type = "XGBoost backup (cloud inference)"
    else:
        raise FileNotFoundError(
            "No final multisource inference model is available."
        )

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


@st.cache_data
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

@st.cache_data
def load_final_test_predictions():

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

    # The frozen final model is a regression model, not the old
    # probability classifier. Heavy-rain detection is therefore
    # derived from the predicted rainfall crossing the 5 mm threshold.
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
                for feature in features
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
        "📈 Model Evaluation",
        "⚠️ Heavy-Rain Events",
        "🧠 Final Multisource Model",
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
        "Predictions use the final ERA5 + IMERG multisource model."
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


    # --------------------------------------------------------
    # Fetch live weather
    # --------------------------------------------------------

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
    # Latest row
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


    # --------------------------------------------------------
    # --------------------------------------------------------
    # FINAL MULTISOURCE MODEL INPUT
    # --------------------------------------------------------

    try:

        final_bundle, final_metadata, final_feature_info = (
            load_final_multisource_model()
        )

        final_model = final_bundle["model"]
        final_feature_columns = final_bundle["feature_columns"]

        # The frozen model uses previous completed IMERG days only.
        # Fetch/cache the seven completed UTC days immediately preceding
        # the current model timestamp.
        probe_time = pd.Timestamp(latest_time)
        if probe_time.tzinfo is None:
            probe_time = probe_time.tz_localize("Asia/Kolkata")
        probe_time_utc = probe_time.tz_convert("UTC").tz_localize(None)

        @st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
        def get_live_imerg_context(target_utc_text: str):
            target_utc = pd.Timestamp(target_utc_text)
            return fetch_imerg_live_context(target_utc, history_days=7)

        imerg_live = get_live_imerg_context(
            probe_time_utc.normalize().isoformat()
        )

        live_input, latest_live = build_live_multisource_features(
            hourly_live=hourly_live,
            imerg_daily=imerg_live,
            feature_columns=final_feature_columns,
        )

        predicted_rain = max(
            0.0,
            float(final_model.predict(live_input)[0]),
        )

        # The frozen final model is a regressor, not a probability classifier.
        # Therefore this is a binary heavy-rain signal, not a calibrated probability.
        heavy_warning = predicted_rain >= HEAVY_RAIN_THRESHOLD
        heavy_signal = 1.0 if heavy_warning else 0.0

        satellite_latest_date = pd.to_datetime(
            imerg_live["date"]
        ).max().normalize()
        model_date = probe_time_utc.normalize()
        satellite_age_days = int(
            (model_date - satellite_latest_date).days
        )

    except Exception as error:

        st.error(
            "Unable to create the live multisource prediction."
        )

        st.exception(error)

        st.info(
            "The live page requires the final RF feature pipeline and at least 25 hourly weather observations. "
            "The IMERG module also includes a presentation-safe archived-data fallback when NASA live access is unavailable."
        )

        st.stop()


    # --------------------------------------------------------
    # Multisource status
    # --------------------------------------------------------

    status_col1, status_col2, status_col3 = st.columns(3)

    satellite_source = imerg_live.attrs.get("source", "live_late")

    with status_col1:
        live_model_label = final_bundle.get(
            "model_type",
            "Random Forest",
        )
        if satellite_source == "historical_fallback":
            st.warning(
                f"🧠 {live_model_label} • archived IMERG fallback"
            )
        else:
            st.success(
                f"🧠 {live_model_label} + IMERG Late live fusion active"
            )

    with status_col2:
        if satellite_source == "historical_fallback":
            st.warning(
                "🛰️ Live IMERG unavailable • using archived context through "
                f"{satellite_latest_date.strftime('%d %b %Y')}"
            )
        else:
            st.info(
                "🛰️ IMERG Late completed day: "
                f"{satellite_latest_date.strftime('%d %b %Y')}"
            )

    with status_col3:
        if satellite_source == "historical_fallback":
            st.warning(
                f"Archived satellite context age: {satellite_age_days} day(s)"
            )
        elif satellite_age_days <= 2:
            st.success(
                f"Satellite context age: {satellite_age_days} day(s)"
            )
        else:
            st.warning(
                f"Satellite context age: {satellite_age_days} day(s)"
            )

    st.caption(
        "The live page uses the frozen 34-feature ERA5 + IMERG feature schema. "
        "The historical model was trained with IMERG Final Daily V07. When available, "
        "live inference uses IMERG Late Daily V07; otherwise the app falls back to the "
        "latest archived IMERG history for demonstration continuity."
    )


    # Risk
    # --------------------------------------------------------

    heavy_warning = (
        heavy_signal >= 1.0
    )

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


    # --------------------------------------------------------
    # Risk banner
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Warning status
    # --------------------------------------------------------

    st.subheader(
        "Current Warning Status"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Next-hour rainfall",
            f"{predicted_rain:.2f} mm",
        )

    with c2:

        st.metric(
            "Heavy-rain model signal",
            "⚠️ YES" if heavy_warning else "✅ NO",
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


    # --------------------------------------------------------
    # Recommended action
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    st.divider()

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


    # --------------------------------------------------------
    # Rainfall chart
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Risk factors
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # System status
    # --------------------------------------------------------

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
        "Live source: Open-Meteo. "
        "The ML model was trained using historical "
        "ERA5-derived data. Live integration is a "
        "prototype and is not separately validated "
        "as an operational forecast."
    )


# ============================================================
# PAGE 2
# MUMBAI SPATIAL RISK MAP
# ============================================================

elif page == "🗺️ Mumbai Risk Map":

    st.header(
        "🗺️ Mumbai Spatial Flood-Risk Map"
    )

    st.info(
        "Live weather is evaluated independently "
        "across nine Mumbai monitoring locations using "
        "live weather, the trained rainfall model, "
        "heavy-rain classifier, and the flood-risk engine."
    )


    # --------------------------------------------------------
    # Refresh
    # --------------------------------------------------------

    if st.button(
        "🔄 Refresh Mumbai Risk Map",
        type="primary",
        use_container_width=True,
    ):

        st.rerun()


    # --------------------------------------------------------
    # Fetch spatial weather
    # --------------------------------------------------------

    try:

        spatial_records = fetch_all_locations()

        spatial = pd.DataFrame(
            spatial_records
        )

        # ----------------------------------------------------
        # Normalize location column name
        # ----------------------------------------------------

        if (
            "location" not in spatial.columns
            and "name" in spatial.columns
        ):

            spatial = spatial.rename(
                columns={
                    "name": "location"
                }
            )

    except Exception as error:

        st.error(
            "Unable to retrieve spatial Mumbai weather."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Empty check
    # --------------------------------------------------------

    if spatial.empty:

        st.error(
            "No spatial weather data was returned."
        )

        st.stop()


    # --------------------------------------------------------
    # Required spatial columns
    # --------------------------------------------------------

    required_columns = [

        "location",
        "latitude",
        "longitude",

        "rainfall_mm",
        "temperature_c",
        "dewpoint_c",
        "pressure_hpa",

        "humidity_pct",

        "wind_speed_ms",
        "u10_ms",
        "v10_ms",

        "rain_3h",
        "rain_6h",
        "rain_24h",

        "rainfall_lag_1h",
        "rainfall_lag_2h",
        "rainfall_lag_3h",
        "rainfall_lag_6h",
        "rainfall_lag_12h",
        "rainfall_lag_24h",

        "hour",
        "month",
        "day_of_year",
    ]


    missing_spatial_columns = [

        column
        for column in required_columns
        if column not in spatial.columns

    ]


    if missing_spatial_columns:

        st.error(
            "Required spatial weather columns are missing:"
        )

        st.write(
            missing_spatial_columns
        )

        st.stop()


    # --------------------------------------------------------
    # Verify model features
    # --------------------------------------------------------

    missing_model_features = [

        feature
        for feature in features
        if feature not in spatial.columns

    ]


    if missing_model_features:

        st.error(
            "Required model features are missing "
            "from the spatial weather data."
        )

        st.write(
            missing_model_features
        )

        st.stop()


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X_spatial = (
        spatial[
            features
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    for column in features:

        X_spatial[
            column
        ] = pd.to_numeric(
            X_spatial[
                column
            ],
            errors="coerce",
        )


    X_spatial = (
        X_spatial
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )


    # --------------------------------------------------------
    # Rainfall prediction
    # --------------------------------------------------------

    try:

        spatial[
            "predicted_rain_mm"
        ] = np.maximum(
            rainfall_model.predict(
                X_spatial
            ),
            0.0,
        )

    except Exception as error:

        st.error(
            "Rainfall model prediction failed."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Heavy probability
    # --------------------------------------------------------

    try:

        spatial[
            "heavy_probability"
        ] = np.clip(
            heavy_classifier
            .predict_proba(
                X_spatial
            )[:, 1],
            0.0,
            1.0,
        )

    except Exception as error:

        st.error(
            "Heavy-rain classifier prediction failed."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Heavy warning
    # --------------------------------------------------------

    spatial[
        "heavy_warning"
    ] = (
        spatial[
            "heavy_probability"
        ]
        >= PROBABILITY_THRESHOLD
    )


    # --------------------------------------------------------
    # Flood-risk engine
    # --------------------------------------------------------

    risk_results = []


    for _, row in spatial.iterrows():

        try:

            result = calculate_flood_risk(

                predicted_rain_mm=float(
                    row[
                        "predicted_rain_mm"
                    ]
                ),

                heavy_probability=float(
                    row[
                        "heavy_probability"
                    ]
                ),

                current_rain_mm=float(
                    row[
                        "rainfall_mm"
                    ]
                ),

                rain_3h_mm=float(
                    row[
                        "rain_3h"
                    ]
                ),

                rain_6h_mm=float(
                    row[
                        "rain_6h"
                    ]
                ),

                rain_24h_mm=float(
                    row[
                        "rain_24h"
                    ]
                ),
            )

        except Exception as error:

            result = {

                "risk_level":
                    "LOW",

                "score":
                    0.0,

                "action":
                    "Risk calculation unavailable.",

                "factors":
                    [
                        f"Risk-engine error: {error}"
                    ],
            }


        risk_results.append(
            result
        )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    spatial[
        "risk_level"
    ] = [

        result[
            "risk_level"
        ]

        for result
        in risk_results

    ]


    spatial[
        "risk_score"
    ] = [

        float(
            result[
                "score"
            ]
        )

        for result
        in risk_results

    ]


    spatial[
        "risk_action"
    ] = [

        result[
            "action"
        ]

        for result
        in risk_results

    ]


    spatial[
        "risk_factors"
    ] = [

        result[
            "factors"
        ]

        for result
        in risk_results

    ]


    # ========================================================
    # MUMBAI-WIDE RISK SUMMARY
    # ========================================================

    st.subheader(
        "Mumbai-Wide Risk Summary"
    )


    low = int(
        (
            spatial[
                "risk_level"
            ]
            == "LOW"
        ).sum()
    )


    moderate = int(
        (
            spatial[
                "risk_level"
            ]
            == "MODERATE"
        ).sum()
    )


    high = int(
        (
            spatial[
                "risk_level"
            ]
            == "HIGH"
        ).sum()
    )


    critical = int(
        (
            spatial[
                "risk_level"
            ]
            == "CRITICAL"
        ).sum()
    )


    warnings = int(
        spatial[
            "heavy_warning"
        ].sum()
    )


    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )


    with c1:

        st.metric(
            "🟢 Low",
            low,
        )


    with c2:

        st.metric(
            "🟡 Moderate",
            moderate,
        )


    with c3:

        st.metric(
            "🟠 High",
            high,
        )


    with c4:

        st.metric(
            "🔴 Critical",
            critical,
        )


    with c5:

        st.metric(
            "⚠️ Heavy warnings",
            warnings,
        )


    # ========================================================
    # MAP
    # ========================================================

    st.subheader(
        "Live Mumbai Risk Map"
    )


    # --------------------------------------------------------
    # Map colors
    # --------------------------------------------------------

    def risk_color(level):

        if level == "LOW":

            return [
                34,
                197,
                94,
                235,
            ]

        if level == "MODERATE":

            return [
                234,
                179,
                8,
                235,
            ]

        if level == "HIGH":

            return [
                249,
                115,
                22,
                240,
            ]

        return [
            239,
            68,
            68,
            245,
        ]


    spatial[
        "color"
    ] = (
        spatial[
            "risk_level"
        ]
        .apply(
            risk_color
        )
    )


    # --------------------------------------------------------
    # Map dataframe
    # --------------------------------------------------------

    map_data = (
        spatial[
            [
                "location",
                "latitude",
                "longitude",

                "risk_level",
                "risk_score",

                "predicted_rain_mm",

                "heavy_probability",
                "heavy_warning",

                "rainfall_mm",
                "rain_3h",
                "rain_6h",
                "rain_24h",

                "color",
            ]
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Marker size
    # --------------------------------------------------------

    map_data[
        "radius"
    ] = (
        3500
        +
        map_data[
            "risk_score"
        ]
        .clip(
            0,
            100
        )
        * 28
    )


    map_data[
        "radius"
    ] = (
        map_data[
            "radius"
        ]
        .clip(
            3500,
            6500
        )
    )


    # --------------------------------------------------------
    # Rounding
    # --------------------------------------------------------

    map_data[
        "risk_score"
    ] = (
        map_data[
            "risk_score"
        ]
        .round(2)
    )


    map_data[
        "predicted_rain_mm"
    ] = (
        map_data[
            "predicted_rain_mm"
        ]
        .round(2)
    )


    map_data[
        "heavy_probability"
    ] = (
        map_data[
            "heavy_probability"
        ]
        .round(3)
    )


    for column in [

        "rainfall_mm",
        "rain_3h",
        "rain_6h",
        "rain_24h",

    ]:

        map_data[
            column
        ] = (
            map_data[
                column
            ]
            .round(2)
        )


    # --------------------------------------------------------
    # Marker layer
    # --------------------------------------------------------

    layer = pdk.Layer(

        "ScatterplotLayer",

        data=map_data,

        get_position=[
            "longitude",
            "latitude",
        ],

        get_radius="radius",

        get_fill_color="color",

        get_line_color=[
            255,
            255,
            255,
            225,
        ],

        line_width_min_pixels=2,

        pickable=True,

        auto_highlight=True,
    )


    # --------------------------------------------------------
    # Location labels
    # --------------------------------------------------------

    labels = pdk.Layer(

        "TextLayer",

        data=map_data,

        get_position=[
            "longitude",
            "latitude",
        ],

        get_text="location",

        get_size=13,

        get_color=[
            235,
            235,
            245,
            230,
        ],

        get_pixel_offset=[
            0,
            40,
        ],

        pickable=False,
    )


    # --------------------------------------------------------
    # Tooltip
    # --------------------------------------------------------

    tooltip = {

        "html":

            (
                "<b>{location}</b><br/>"

                "Risk: "
                "<b>{risk_level}</b><br/>"

                "Risk score: "
                "{risk_score}<br/>"

                "Next-hour rainfall: "
                "{predicted_rain_mm} mm<br/>"

                "Heavy probability: "
                "{heavy_probability}<br/>"

                "Heavy warning: "
                "{heavy_warning}<br/>"

                "Current rainfall: "
                "{rainfall_mm} mm<br/>"

                "Rain 3h: "
                "{rain_3h} mm<br/>"

                "Rain 6h: "
                "{rain_6h} mm<br/>"

                "Rain 24h: "
                "{rain_24h} mm"
            ),

        "style": {

            "backgroundColor":
                "rgba(15,15,20,.97)",

            "color":
                "white",
        },
    }


    # --------------------------------------------------------
    # View state
    # --------------------------------------------------------

    view_state = pdk.ViewState(

        latitude=19.0760,

        longitude=72.8777,

        zoom=10.2,

        pitch=35,
    )


    deck = pdk.Deck(

        layers=[
            layer,
            labels,
        ],

        initial_view_state=
            view_state,

        tooltip=
            tooltip,
    )


    st.pydeck_chart(
        deck,
        use_container_width=True,
    )


    # ========================================================
    # LEGEND
    # ========================================================

    st.markdown(
        "**Risk Legend**"
    )

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


    # ========================================================
    # LOCATION TABLE
    # ========================================================

    st.subheader(
        "Location-by-Location Risk"
    )


    location_table = (
        spatial[
            [
                "location",

                "risk_level",

                "risk_score",

                "predicted_rain_mm",

                "heavy_probability",

                "heavy_warning",

                "rainfall_mm",

                "rain_3h",

                "rain_6h",

                "rain_24h",
            ]
        ]
        .copy()
    )


    location_table = (
        location_table
        .rename(
            columns={

                "location":
                    "Location",

                "risk_level":
                    "Risk",

                "risk_score":
                    "Risk Score",

                "predicted_rain_mm":
                    "Next-hour Rain (mm)",

                "heavy_probability":
                    "Heavy Probability",

                "heavy_warning":
                    "Heavy Warning",

                "rainfall_mm":
                    "Current Rain (mm)",

                "rain_3h":
                    "Rain 3h (mm)",

                "rain_6h":
                    "Rain 6h (mm)",

                "rain_24h":
                    "Rain 24h (mm)",
            }
        )
    )


    location_table[
        "Risk Score"
    ] = (
        location_table[
            "Risk Score"
        ]
        .round(2)
    )


    location_table[
        "Next-hour Rain (mm)"
    ] = (
        location_table[
            "Next-hour Rain (mm)"
        ]
        .round(2)
    )


    location_table[
        "Heavy Probability"
    ] = (
        location_table[
            "Heavy Probability"
        ]
        .map(
            lambda value:
                f"{float(value):.1%}"
        )
    )


    location_table[
        "Heavy Warning"
    ] = (
        location_table[
            "Heavy Warning"
        ]
        .map(
            lambda value:
                "YES"
                if value
                else "NO"
        )
    )


    for column in [

        "Current Rain (mm)",
        "Rain 3h (mm)",
        "Rain 6h (mm)",
        "Rain 24h (mm)",

    ]:

        location_table[
            column
        ] = (
            location_table[
                column
            ]
            .round(2)
        )


    location_table = (
        location_table
        .sort_values(
            "Risk Score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    st.dataframe(

        location_table,

        use_container_width=True,

        hide_index=True,
    )


    # ========================================================
    # HIGHEST CURRENT RISK
    # ========================================================

    highest = (

        spatial

        .sort_values(
            "risk_score",
            ascending=False,
        )

        .iloc[0]
    )


    st.subheader(
        "Highest Current Risk"
    )


    highest_level = (
        highest[
            "risk_level"
        ]
    )


    highest_score = float(
        highest[
            "risk_score"
        ]
    )


    icon, alert_function = (
        risk_alert(
            highest_level
        )
    )


    alert_function(

        f"{icon} **{highest['location']}** "
        f"has the highest prototype risk "
        f"score of **{highest_score:.0f} "
        f"({highest_level})**."
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Next-hour rain",
            f"{float(highest['predicted_rain_mm']):.2f} mm",
        )


    with c2:

        st.metric(
            "Heavy probability",
            f"{float(highest['heavy_probability']):.1%}",
        )


    with c3:

        st.metric(
            "Current rain",
            f"{float(highest['rainfall_mm']):.2f} mm",
        )


    with c4:

        st.metric(
            "24h rain",
            f"{float(highest['rain_24h']):.2f} mm",
        )


    st.write(
        f"**Recommended action:** "
        f"{highest['risk_action']}"
    )


    st.subheader(
        "Risk Factors"
    )


    highest_factors = (
        highest[
            "risk_factors"
        ]
    )


    if isinstance(
        highest_factors,
        list,
    ):

        for factor in highest_factors:

            st.write(
                f"• {factor}"
            )

    else:

        st.write(
            f"• {highest_factors}"
        )


    st.divider()


    st.caption(

        "Spatial monitoring uses approximately "
        "sampled locations and is not a street-level "
        "flood sensor network. Live weather comes "
        "from Open-Meteo. The rainfall model was "
        "trained on historical ERA5-derived data, "
        "while the flood-risk layer is a rule-based "
        "prototype."
    )


# ============================================================
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
# FINAL MULTISOURCE MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.header(
        "📈 Final Multisource Model Evaluation — 2025 Holdout"
    )

    st.info(
        "This page now uses the frozen 2010–2025 ERA5 + NASA GPM IMERG "
        "Random Forest model and its 2025 holdout predictions."
    )

    try:
        final_test = load_final_test_predictions()
    except Exception as error:
        st.error("Unable to load final multisource test predictions.")
        st.exception(error)
        st.stop()

    actual = final_test["next_hour_rain"]
    predicted = final_test["predicted_rain_mm"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    actual_heavy = final_test["actual_heavy"]
    predicted_heavy = final_test["predicted_heavy"]

    precision = precision_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )
    recall = recall_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )
    f1 = f1_score(
        actual_heavy,
        predicted_heavy,
        zero_division=0,
    )

    st.subheader("Rainfall Regression")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Test observations", f"{len(final_test):,}")

    with c2:
        st.metric("MAE", f"{mae:.4f} mm")

    with c3:
        st.metric("RMSE", f"{rmse:.4f} mm")

    with c4:
        st.metric("R²", f"{r2:.4f}")

    st.caption(
        "2025 holdout: "
        f"{final_test['datetime'].min()} → {final_test['datetime'].max()}"
    )

    st.subheader("Actual vs Predicted Rainfall")

    plot_count = st.slider(
        "Number of observations",
        min_value=100,
        max_value=min(1000, len(final_test)),
        value=min(500, len(final_test)),
        step=100,
        key="final_eval_plot_count",
    )

    chart_data = (
        final_test[["datetime", "next_hour_rain", "predicted_rain_mm"]]
        .tail(plot_count)
        .set_index("datetime")
        .rename(
            columns={
                "next_hour_rain": "Actual",
                "predicted_rain_mm": "Predicted",
            }
        )
    )

    st.line_chart(chart_data, use_container_width=True)

    st.subheader("Heavy-Rain Hour Detection")

    st.info(
        f"Heavy rain is defined as ≥ {HEAVY_RAIN_THRESHOLD:.1f} mm/hour. "
        "The metrics in this section are hour-level observations, not storm/event counts. "
        "The frozen final model is a rainfall regressor, so heavy-rain detection is "
        "derived by thresholding its predicted rainfall; there is no separate probability "
        "classifier in this final bundle."
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric("Actual heavy hours", int(actual_heavy.sum()))
    with h2:
        st.metric("Detected heavy hours", int(((actual_heavy == 1) & (predicted_heavy == 1)).sum()))
    with h3:
        st.metric("Missed heavy hours", int(((actual_heavy == 1) & (predicted_heavy == 0)).sum()))
    with h4:
        st.metric("False-alarm hours", int(((actual_heavy == 0) & (predicted_heavy == 1)).sum()))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Hourly precision", f"{precision:.3f}")
    with m2:
        st.metric("Hourly recall", f"{recall:.3f}")
    with m3:
        st.metric("Hourly F1-score", f"{f1:.3f}")

    cm = confusion_matrix(actual_heavy, predicted_heavy, labels=[0, 1])

    st.subheader("Heavy-Rain Hour Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Heavy"],
        columns=["Predicted Normal", "Predicted Heavy"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Prediction Error Distribution")

    error_data = (
        final_test[["datetime", "absolute_error_mm"]]
        .tail(plot_count)
        .set_index("datetime")
        .rename(columns={"absolute_error_mm": "Absolute error (mm)"})
    )

    st.line_chart(error_data, use_container_width=True)


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

elif page == "🧠 Final Multisource Model":

    st.header(
        "🧠 Final Multisource Rainfall Model"
    )

    st.caption(
        "Frozen 2010–2025 ERA5 + NASA GPM IMERG Final Daily V07 "
        "multisource model."
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
        st.error(
            "One or more final multisource model artifacts are missing."
        )
        for path in missing_artifacts:
            st.write(path)
        st.stop()

    try:
        (
            final_bundle,
            final_metadata,
            final_feature_info,
        ) = load_final_multisource_model()

        with open(
            FINAL_MODEL_TRAINING_METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            final_metrics = json.load(file)

        final_predictions = pd.read_csv(
            FINAL_MODEL_PREDICTIONS_PATH,
            parse_dates=["datetime"],
        )

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
        "🌐 Multisource Architecture"
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
        "IMERG is a daily historical auxiliary source here. "
        "Only previously completed daily IMERG periods are used "
        "for the model features, avoiding temporal leakage. "
        "This page demonstrates the historical multisource model; "
        "it is not a claim of real-time satellite nowcasting."
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
        "🧩 Frozen Model Features"
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
        "🔎 Historical Prediction Explorer"
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
        "⚖️ Final Model Comparison"
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
        "Rainfall model: Random Forest Regressor"
    )


    st.write(
        "Heavy-rain model: Random Forest Classifier"
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
        "Legacy model test period shown on monitoring pages; final multisource evaluation is documented on the Final Multisource Model page."
    )


    st.write(
        "Live weather: Open-Meteo"
    )


    st.write(
        "Flood-risk layer: rule-based prototype"
    )