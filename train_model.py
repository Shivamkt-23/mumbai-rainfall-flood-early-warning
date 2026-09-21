# ============================================================
# MUMBAI RAINFALL + HEAVY RAIN PREDICTION
# Complete ML Training Pipeline with Threshold Tuning
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# ============================================================
# 0. CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/mumbai_era5_ml_ready_cleaned.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Final test begins here
SPLIT_DATE = "2024-01-01"

# Prediction target
TARGET_COLUMN = "next_hour_rain"

# Heavy rainfall definition
HEAVY_RAIN_THRESHOLD_MM = 5.0

# Additional extreme-rain evaluation
EXTREME_RAIN_THRESHOLD_MM = 10.0

# Reproducibility
RANDOM_STATE = 42

# Historical rainfall lags
LAG_HOURS = [
    1,
    2,
    3,
    6,
    12,
    24,
]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("1. LOADING DATA")
print("=" * 60)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure the CSV is inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully."
)

print(
    f"Original shape: {df.shape}"
)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("2. DATA VALIDATION")
print("=" * 60)

if "datetime" not in df.columns:

    raise ValueError(
        "The dataset does not contain a 'datetime' column."
    )

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        "was not found in the dataset."
    )

if "rainfall_mm" not in df.columns:

    raise ValueError(
        "The dataset does not contain "
        "'rainfall_mm'."
    )


# Convert datetime

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)


# Remove invalid datetime rows

invalid_datetime = df["datetime"].isna().sum()

if invalid_datetime > 0:

    print(
        f"Removing {invalid_datetime} "
        "rows with invalid datetime."
    )

    df = df.dropna(
        subset=["datetime"]
    )


# Sort chronologically

df = (
    df.sort_values("datetime")
      .reset_index(drop=True)
)


print(
    "Chronological sorting complete."
)


# Duplicate timestamps

duplicate_datetime = (
    df["datetime"].duplicated().sum()
)

print(
    f"Duplicate datetime rows: "
    f"{duplicate_datetime}"
)


# ============================================================
# 3. CREATE RAINFALL LAG FEATURES
# ============================================================

print("\n" + "=" * 60)
print("3. CREATING RAINFALL LAG FEATURES")
print("=" * 60)

for lag in LAG_HOURS:

    column_name = (
        f"rainfall_lag_{lag}h"
    )

    df[column_name] = (
        df["rainfall_mm"].shift(lag)
    )

    print(
        f"Created: {column_name}"
    )


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("4. DEFINING FEATURES")
print("=" * 60)

FEATURES = [

    # Current rainfall
    "rainfall_mm",

    # Atmospheric variables
    "temperature_c",
    "dewpoint_c",
    "pressure_hpa",

    # Wind
    "u10_ms",
    "v10_ms",
    "wind_speed_ms",

    # Humidity
    "humidity_pct",

    # Existing rainfall accumulations
    "rain_3h",
    "rain_6h",
    "rain_24h",

    # Historical rainfall
    "rainfall_lag_1h",
    "rainfall_lag_2h",
    "rainfall_lag_3h",
    "rainfall_lag_6h",
    "rainfall_lag_12h",
    "rainfall_lag_24h",

    # Time features
    "month",
    "hour",
    "day_of_year",

    # Cyclic encodings
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "These required features are missing:\n"
        + "\n".join(missing_features)
    )


print(
    f"Total features: {len(FEATURES)}"
)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("5. CLEANING TRAINING DATA")
print("=" * 60)

required_columns = (
    FEATURES
    + [
        TARGET_COLUMN,
        "datetime",
    ]
)

rows_before_drop = len(df)

df = (
    df.dropna(
        subset=required_columns
    )
    .reset_index(drop=True)
)

rows_after_drop = len(df)

print(
    f"Rows before cleanup: "
    f"{rows_before_drop}"
)

print(
    f"Rows after cleanup:  "
    f"{rows_after_drop}"
)

print(
    f"Rows removed: "
    f"{rows_before_drop - rows_after_drop}"
)


if len(df) == 0:

    raise ValueError(
        "No rows remain after data cleaning."
    )


# ============================================================
# 6. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("6. TARGET ANALYSIS")
print("=" * 60)

print(
    df[TARGET_COLUMN].describe()
)


negative_target = (
    df[TARGET_COLUMN] < 0
).sum()


if negative_target > 0:

    print(
        f"WARNING: {negative_target} "
        "negative rainfall target values found."
    )

else:

    print(
        "No negative target rainfall values."
    )


# ============================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("7. TIME-BASED TRAIN / TEST SPLIT")
print("=" * 60)

split_timestamp = pd.Timestamp(
    SPLIT_DATE
)

train = (
    df[
        df["datetime"] < split_timestamp
    ]
    .copy()
)

test = (
    df[
        df["datetime"] >= split_timestamp
    ]
    .copy()
)


if len(train) == 0:

    raise ValueError(
        "Training set is empty."
    )


if len(test) == 0:

    raise ValueError(
        "Testing set is empty."
    )


X_train = train[FEATURES]

y_train = train[TARGET_COLUMN]

X_test = test[FEATURES]

y_test = test[TARGET_COLUMN]


print(
    f"Training rows: {len(train)}"
)

print(
    f"Testing rows:  {len(test)}"
)


print(
    "\nTraining period:"
)

print(
    train["datetime"].min(),
    "to",
    train["datetime"].max()
)


print(
    "\nTesting period:"
)

print(
    test["datetime"].min(),
    "to",
    test["datetime"].max()
)


# ============================================================
# 8. BASELINE — CURRENT RAINFALL
# ============================================================

print("\n" + "=" * 60)
print("8. BASELINE — PERSISTENCE")
print("=" * 60)

baseline_prediction = (
    X_test["rainfall_mm"]
    .to_numpy()
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)


baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)


baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)


print(
    f"Baseline MAE : "
    f"{baseline_mae:.6f} mm"
)

print(
    f"Baseline RMSE: "
    f"{baseline_rmse:.6f} mm"
)

print(
    f"Baseline R²  : "
    f"{baseline_r2:.6f}"
)


# ============================================================
# 9. RANDOM FOREST REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("9. RANDOM FOREST REGRESSION")
print("=" * 60)

rf_regressor = RandomForestRegressor(

    n_estimators=300,

    max_depth=20,

    min_samples_split=2,

    min_samples_leaf=2,

    max_features=1.0,

    bootstrap=True,

    n_jobs=-1,

    random_state=RANDOM_STATE,
)


print(
    "Training Random Forest regression model..."
)


rf_regressor.fit(
    X_train,
    y_train
)


print(
    "Random Forest regression training complete."
)


# ============================================================
# 10. REGRESSION PREDICTION
# ============================================================

rf_prediction = (
    rf_regressor.predict(X_test)
)

# Rainfall cannot be negative

rf_prediction = np.maximum(
    rf_prediction,
    0
)


# ============================================================
# 11. REGRESSION EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("10. RANDOM FOREST REGRESSION RESULTS")
print("=" * 60)

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_prediction
    )
)


rf_r2 = r2_score(
    y_test,
    rf_prediction
)


print(
    f"MAE : {rf_mae:.6f} mm"
)

print(
    f"RMSE: {rf_rmse:.6f} mm"
)

print(
    f"R²  : {rf_r2:.6f}"
)


# ============================================================
# 12. REGRESSION FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("11. REGRESSION FEATURE IMPORTANCE")
print("=" * 60)

regression_importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        rf_regressor
        .feature_importances_,
})


regression_importance = (
    regression_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    regression_importance
    .to_string(index=False)
)


# ============================================================
# 13. CREATE HEAVY-RAIN TARGET
# ============================================================

print("\n" + "=" * 60)
print("12. HEAVY RAIN CLASSIFICATION")
print("=" * 60)

print(
    f"Heavy-rain threshold: "
    f"{HEAVY_RAIN_THRESHOLD_MM} mm"
)


y_train_heavy = (
    y_train >= HEAVY_RAIN_THRESHOLD_MM
).astype(int)


y_test_heavy = (
    y_test >= HEAVY_RAIN_THRESHOLD_MM
).astype(int)


train_heavy_count = (
    int(y_train_heavy.sum())
)


test_heavy_count = (
    int(y_test_heavy.sum())
)


train_normal_count = (
    int((y_train_heavy == 0).sum())
)


test_normal_count = (
    int((y_test_heavy == 0).sum())
)


print(
    f"Training heavy-rain events: "
    f"{train_heavy_count}"
)

print(
    f"Training normal events: "
    f"{train_normal_count}"
)

print(
    f"Testing heavy-rain events: "
    f"{test_heavy_count}"
)

print(
    f"Testing normal events: "
    f"{test_normal_count}"
)


if train_heavy_count == 0:

    raise ValueError(
        "No heavy-rain events exist "
        "in the training set."
    )


# ============================================================
# 14. THRESHOLD-SELECTION VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("13. CHRONOLOGICAL VALIDATION SET")
print("=" * 60)

# Use the last 20% of the training period
# to choose the warning probability threshold.

validation_split = int(
    len(train) * 0.80
)


train_core = (
    train
    .iloc[:validation_split]
    .copy()
)


validation = (
    train
    .iloc[validation_split:]
    .copy()
)


X_train_core = (
    train_core[FEATURES]
)


y_train_core = (
    train_core[TARGET_COLUMN]
)


X_validation = (
    validation[FEATURES]
)


y_validation = (
    validation[TARGET_COLUMN]
)


y_train_core_heavy = (
    y_train_core
    >= HEAVY_RAIN_THRESHOLD_MM
).astype(int)


y_validation_heavy = (
    y_validation
    >= HEAVY_RAIN_THRESHOLD_MM
).astype(int)


print(
    f"Core training rows: "
    f"{len(train_core)}"
)


print(
    f"Validation rows: "
    f"{len(validation)}"
)


print(
    "\nCore training period:"
)


print(
    train_core["datetime"].min(),
    "to",
    train_core["datetime"].max()
)


print(
    "\nValidation period:"
)


print(
    validation["datetime"].min(),
    "to",
    validation["datetime"].max()
)


# ============================================================
# 15. TRAIN TEMPORARY CLASSIFIER FOR THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 60)
print("14. TRAINING THRESHOLD-SELECTION MODEL")
print("=" * 60)

threshold_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=20,

    min_samples_split=2,

    min_samples_leaf=2,

    max_features=1.0,

    bootstrap=True,

    class_weight="balanced",

    n_jobs=-1,

    random_state=RANDOM_STATE,
)


threshold_model.fit(
    X_train_core,
    y_train_core_heavy
)


print(
    "Threshold-selection model trained."
)


# Validation probabilities

validation_probability = (
    threshold_model
    .predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# 16. TEST WARNING THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("15. TESTING WARNING THRESHOLDS")
print("=" * 60)


thresholds = np.arange(
    0.20,
    0.71,
    0.05
)


threshold_results = []


for threshold in thresholds:

    validation_prediction = (
        validation_probability
        >= threshold
    ).astype(int)


    validation_precision = (
        precision_score(
            y_validation_heavy,
            validation_prediction,
            zero_division=0
        )
    )


    validation_recall = (
        recall_score(
            y_validation_heavy,
            validation_prediction,
            zero_division=0
        )
    )


    validation_f1 = (
        f1_score(
            y_validation_heavy,
            validation_prediction,
            zero_division=0
        )
    )


    threshold_results.append({

        "threshold":
            float(threshold),

        "precision":
            float(validation_precision),

        "recall":
            float(validation_recall),

        "f1":
            float(validation_f1),

    })


threshold_table = pd.DataFrame(
    threshold_results
)


print(
    threshold_table
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 17. SELECT WARNING THRESHOLD
# ============================================================

print("\n" + "=" * 60)
print("16. SELECTING WARNING THRESHOLD")
print("=" * 60)


# We prioritize recall because this is
# an early-warning system.
#
# Requirement:
# recall >= 0.70 if possible.
#
# Among thresholds satisfying that condition,
# select the one with the best precision.
#
# If none reach 0.70 recall,
# choose the best F1 score.

RECALL_TARGET = 0.70


acceptable_thresholds = (
    threshold_table[
        threshold_table["recall"]
        >= RECALL_TARGET
    ]
)


if len(acceptable_thresholds) > 0:

    selected_row = (
        acceptable_thresholds
        .sort_values(
            ["precision", "f1"],
            ascending=False
        )
        .iloc[0]
    )

else:

    selected_row = (
        threshold_table
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )


selected_threshold = float(
    selected_row["threshold"]
)


selected_validation_precision = float(
    selected_row["precision"]
)


selected_validation_recall = float(
    selected_row["recall"]
)


selected_validation_f1 = float(
    selected_row["f1"]
)


print(
    f"Selected threshold: "
    f"{selected_threshold:.2f}"
)


print(
    f"Validation precision: "
    f"{selected_validation_precision:.6f}"
)


print(
    f"Validation recall: "
    f"{selected_validation_recall:.6f}"
)


print(
    f"Validation F1: "
    f"{selected_validation_f1:.6f}"
)


# ============================================================
# 18. TRAIN FINAL HEAVY-RAIN CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("17. TRAINING FINAL HEAVY-RAIN CLASSIFIER")
print("=" * 60)


rf_classifier = RandomForestClassifier(

    n_estimators=300,

    max_depth=20,

    min_samples_split=2,

    min_samples_leaf=2,

    max_features=1.0,

    bootstrap=True,

    class_weight="balanced",

    n_jobs=-1,

    random_state=RANDOM_STATE,
)


print(
    "Training final classifier "
    "using the complete training period..."
)


rf_classifier.fit(
    X_train,
    y_train_heavy
)


print(
    "Final classifier training complete."
)


# ============================================================
# 19. FINAL TEST PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("18. FINAL TEST PREDICTIONS")
print("=" * 60)


heavy_probability = (
    rf_classifier
    .predict_proba(
        X_test
    )[:, 1]
)


# Apply selected threshold

heavy_prediction = (
    heavy_probability
    >= selected_threshold
).astype(int)


# ============================================================
# 20. FINAL CLASSIFICATION METRICS
# ============================================================

precision = precision_score(
    y_test_heavy,
    heavy_prediction,
    zero_division=0
)


recall = recall_score(
    y_test_heavy,
    heavy_prediction,
    zero_division=0
)


f1 = f1_score(
    y_test_heavy,
    heavy_prediction,
    zero_division=0
)


print("\n" + "=" * 60)
print("19. FINAL HEAVY-RAIN RESULTS")
print("=" * 60)


print(
    f"Selected threshold: "
    f"{selected_threshold:.2f}"
)


print(
    f"Precision: "
    f"{precision:.6f}"
)


print(
    f"Recall   : "
    f"{recall:.6f}"
)


print(
    f"F1-score : "
    f"{f1:.6f}"
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("20. CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test_heavy,
    heavy_prediction,
    labels=[0, 1]
)


print(
    "                    Predicted"
)

print(
    "                 Normal  Heavy"
)

print(
    f"Actual Normal   "
    f"{cm[0, 0]:7d} "
    f"{cm[0, 1]:6d}"
)

print(
    f"Actual Heavy    "
    f"{cm[1, 0]:7d} "
    f"{cm[1, 1]:6d}"
)


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("21. CLASSIFICATION REPORT")
print("=" * 60)


classification_report_text = (
    classification_report(
        y_test_heavy,
        heavy_prediction,
        labels=[0, 1],
        target_names=[
            "Normal Rain",
            "Heavy Rain",
        ],
        zero_division=0,
    )
)


print(
    classification_report_text
)


# ============================================================
# 23. HEAVY-RAIN EVENT INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("22. HEAVY-RAIN EVENT INSPECTION")
print("=" * 60)


heavy_mask = (
    y_test
    >= HEAVY_RAIN_THRESHOLD_MM
)


if heavy_mask.sum() > 0:

    heavy_results = pd.DataFrame({

        "datetime":
            test
            .loc[
                heavy_mask,
                "datetime"
            ]
            .values,

        "actual_rain_mm":
            y_test[
                heavy_mask
            ].values,

        "predicted_rain_mm":
            rf_prediction[
                heavy_mask
            ].flatten(),

        "heavy_probability":
            heavy_probability[
                heavy_mask
            ],

        "predicted_heavy":
            heavy_prediction[
                heavy_mask
            ],

    })


    heavy_results[
        "regression_error_mm"
    ] = (
        heavy_results[
            "actual_rain_mm"
        ]
        -
        heavy_results[
            "predicted_rain_mm"
        ]
    )


    print(
        heavy_results
        .to_string(index=False)
    )


    heavy_mae = mean_absolute_error(
        heavy_results[
            "actual_rain_mm"
        ],
        heavy_results[
            "predicted_rain_mm"
        ]
    )


    heavy_rmse = np.sqrt(
        mean_squared_error(
            heavy_results[
                "actual_rain_mm"
            ],
            heavy_results[
                "predicted_rain_mm"
            ]
        )
    )


    print(
        f"\nHeavy-rain MAE: "
        f"{heavy_mae:.6f} mm"
    )


    print(
        f"Heavy-rain RMSE: "
        f"{heavy_rmse:.6f} mm"
    )


else:

    print(
        "No heavy-rain events found "
        "in the test set."
    )


# ============================================================
# 24. EXTREME-RAIN EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("23. EXTREME-RAIN EVALUATION")
print("=" * 60)


extreme_mask = (
    y_test
    >= EXTREME_RAIN_THRESHOLD_MM
)


extreme_count = (
    int(extreme_mask.sum())
)


print(
    f"Extreme threshold: "
    f"{EXTREME_RAIN_THRESHOLD_MM} mm"
)


print(
    f"Extreme events in test set: "
    f"{extreme_count}"
)


if extreme_count > 0:

    extreme_actual = (
        y_test[
            extreme_mask
        ]
    )


    extreme_predicted = (
        rf_prediction[
            extreme_mask
        ]
    )


    extreme_mae = (
        mean_absolute_error(
            extreme_actual,
            extreme_predicted
        )
    )


    extreme_rmse = np.sqrt(
        mean_squared_error(
            extreme_actual,
            extreme_predicted
        )
    )


    print(
        f"Extreme-rain MAE : "
        f"{extreme_mae:.6f} mm"
    )


    print(
        f"Extreme-rain RMSE: "
        f"{extreme_rmse:.6f} mm"
    )


# ============================================================
# 25. CLASSIFIER FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("24. CLASSIFIER FEATURE IMPORTANCE")
print("=" * 60)


classifier_importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        rf_classifier
        .feature_importances_,
})


classifier_importance = (
    classifier_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    classifier_importance
    .to_string(index=False)
)


# ============================================================
# 26. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("25. MODEL COMPARISON")
print("=" * 60)


print(
    f"Baseline       | "
    f"MAE: {baseline_mae:.4f} | "
    f"RMSE: {baseline_rmse:.4f} | "
    f"R²: {baseline_r2:.4f}"
)


print(
    f"Random Forest  | "
    f"MAE: {rf_mae:.4f} | "
    f"RMSE: {rf_rmse:.4f} | "
    f"R²: {rf_r2:.4f}"
)


# ============================================================
# 27. CREATE TEST PREDICTION TABLE
# ============================================================

print("\n" + "=" * 60)
print("26. SAVING TEST PREDICTIONS")
print("=" * 60)


prediction_results = pd.DataFrame({

    "datetime":
        test["datetime"].values,

    "actual_rain_mm":
        y_test.values,

    "predicted_rain_mm":
        rf_prediction,

    "heavy_rain_probability":
        heavy_probability,

    "predicted_heavy_rain":
        heavy_prediction,

    "actual_heavy_rain":
        y_test_heavy.values,
})


prediction_results[
    "absolute_error_mm"
] = np.abs(
    prediction_results[
        "actual_rain_mm"
    ]
    -
    prediction_results[
        "predicted_rain_mm"
    ]
)


prediction_path = (
    MODEL_DIR
    / "test_predictions.csv"
)


prediction_results.to_csv(
    prediction_path,
    index=False
)


print(
    f"Saved: {prediction_path}"
)


# ============================================================
# 28. SAVE MODELS
# ============================================================

print("\n" + "=" * 60)
print("27. SAVING MODELS")
print("=" * 60)


rainfall_model_path = (
    MODEL_DIR
    / "rainfall_model.pkl"
)


heavy_classifier_path = (
    MODEL_DIR
    / "heavy_rain_classifier.pkl"
)


features_path = (
    MODEL_DIR
    / "rainfall_features.pkl"
)


joblib.dump(
    rf_regressor,
    rainfall_model_path
)


joblib.dump(
    rf_classifier,
    heavy_classifier_path
)


joblib.dump(
    FEATURES,
    features_path
)


print(
    f"Rainfall model saved: "
    f"{rainfall_model_path}"
)


print(
    f"Heavy-rain classifier saved: "
    f"{heavy_classifier_path}"
)


print(
    f"Feature list saved: "
    f"{features_path}"
)


# ============================================================
# 29. SAVE METADATA
# ============================================================

print("\n" + "=" * 60)
print("28. SAVING TRAINING METADATA")
print("=" * 60)


metrics = {

    "dataset": {

        "path":
            str(DATA_PATH),

        "original_rows":
            int(rows_before_drop),

        "final_rows":
            int(rows_after_drop),

        "rows_removed":
            int(
                rows_before_drop
                - rows_after_drop
            ),

        "feature_count":
            int(len(FEATURES)),
    },


    "target": {

        "column":
            TARGET_COLUMN,

        "description":
            "Rainfall in the next hour",

        "heavy_rain_threshold_mm":
            HEAVY_RAIN_THRESHOLD_MM,

        "extreme_rain_threshold_mm":
            EXTREME_RAIN_THRESHOLD_MM,
    },


    "split": {

        "split_date":
            SPLIT_DATE,

        "training_rows":
            int(len(train)),

        "testing_rows":
            int(len(test)),
    },


    "baseline": {

        "mae_mm":
            float(baseline_mae),

        "rmse_mm":
            float(baseline_rmse),

        "r2":
            float(baseline_r2),
    },


    "rainfall_regression": {

        "model":
            "RandomForestRegressor",

        "mae_mm":
            float(rf_mae),

        "rmse_mm":
            float(rf_rmse),

        "r2":
            float(rf_r2),

        "parameters":
            rf_regressor.get_params(),
    },


    "heavy_rain_classification": {

        "model":
            "RandomForestClassifier",

        "threshold_mm":
            HEAVY_RAIN_THRESHOLD_MM,

        "selected_probability_threshold":
            selected_threshold,

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "validation_precision":
            selected_validation_precision,

        "validation_recall":
            selected_validation_recall,

        "validation_f1":
            selected_validation_f1,

        "training_heavy_events":
            train_heavy_count,

        "testing_heavy_events":
            test_heavy_count,

        "confusion_matrix":
            cm.tolist(),

        "parameters":
            rf_classifier.get_params(),
    },


    "feature_engineering": {

        "lag_hours":
            LAG_HOURS,

        "features":
            FEATURES,
    },


    "reproducibility": {

        "random_state":
            RANDOM_STATE,
    },
}


metrics_path = (
    MODEL_DIR
    / "training_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"Training metadata saved: "
    f"{metrics_path}"
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("29. FINAL SUMMARY")
print("=" * 60)


print(
    "\nDataset:"
)

print(
    f"  Original rows : "
    f"{rows_before_drop}"
)

print(
    f"  Final rows   : "
    f"{rows_after_drop}"
)

print(
    f"  Features     : "
    f"{len(FEATURES)}"
)


print(
    "\nRainfall regression:"
)

print(
    f"  Baseline MAE      : "
    f"{baseline_mae:.4f} mm"
)

print(
    f"  Random Forest MAE : "
    f"{rf_mae:.4f} mm"
)

print(
    f"  Random Forest RMSE: "
    f"{rf_rmse:.4f} mm"
)

print(
    f"  Random Forest R²  : "
    f"{rf_r2:.4f}"
)


print(
    "\nHeavy-rain classifier:"
)

print(
    f"  Rain threshold    : "
    f"{HEAVY_RAIN_THRESHOLD_MM:.1f} mm"
)

print(
    f"  Probability cutoff: "
    f"{selected_threshold:.2f}"
)

print(
    f"  Precision         : "
    f"{precision:.4f}"
)

print(
    f"  Recall            : "
    f"{recall:.4f}"
)

print(
    f"  F1-score          : "
    f"{f1:.4f}"
)


print(
    "\nSaved files:"
)

print(
    f"  {rainfall_model_path}"
)

print(
    f"  {heavy_classifier_path}"
)

print(
    f"  {features_path}"
)

print(
    f"  {metrics_path}"
)

print(
    f"  {prediction_path}"
)


print(
    "\nTraining pipeline completed successfully."
)

print("=" * 60)