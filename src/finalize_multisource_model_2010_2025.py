#!/usr/bin/env python3
"""
finalize_multisource_model_2010_2025.py

Finalizes the already-trained 2010-2025 multisource models without
retraining them.

What this script verifies:
    - exact available train/test date ranges
    - no overlap between train and test
    - actual 2025 test window present in the dataset
    - saved RF/XGBoost artifacts can load
    - saved predictions agree with the saved models
    - exact feature order is preserved

Primary model for deployment:
    Random Forest (ERA5 + IMERG)

XGBoost remains saved as a comparison/backup model.

Outputs:
    models/final_multisource_model_bundle.pkl
    models/final_model_metadata.json
    models/final_multisource_feature_columns.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is required to validate the saved XGBoost artifact. "
        "Run: python -m pip install -U xgboost"
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "mumbai_era5_imerg_multisource_2010_2025.csv"
)

RF_PATH = BASE_DIR / "models" / "final_multisource_random_forest.pkl"
XGB_PATH = BASE_DIR / "models" / "final_multisource_xgboost.json"

PRED_PATH = (
    BASE_DIR
    / "models"
    / "final_multisource_test_predictions.csv"
)

METRICS_PATH = (
    BASE_DIR
    / "models"
    / "final_training_metrics.json"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR
    / "models"
    / "final_multisource_feature_importance.csv"
)

BUNDLE_PATH = (
    BASE_DIR
    / "models"
    / "final_multisource_model_bundle.pkl"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "final_model_metadata.json"
)

FEATURE_COLUMNS_PATH = (
    BASE_DIR
    / "models"
    / "final_multisource_feature_columns.json"
)

TARGET = "next_hour_rain"

NON_FEATURE_COLUMNS = {
    "datetime",
    "date",
    TARGET,
}

CONSTANT_COLUMNS = {
    "latitude_center",
    "longitude_center",
    "imerg_lat",
    "imerg_lon",
}


def banner(text: str) -> None:
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def prepare_features(df: pd.DataFrame) -> list[str]:
    cols = []

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col in CONSTANT_COLUMNS:
            continue
        cols.append(col)

    return cols


def metrics(y_true, y_pred) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom else float("nan")

    return {
        "MAE_mm": mae,
        "RMSE_mm": rmse,
        "R2": r2,
    }


def heavy_metrics(y_true, y_pred, threshold=5.0) -> dict:
    mask = y_true >= threshold

    if int(mask.sum()) == 0:
        return {
            "count": 0,
            "MAE_mm": None,
            "RMSE_mm": None,
        }

    return {
        "count": int(mask.sum()),
        "MAE_mm": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))),
        "RMSE_mm": float(
            np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))
        ),
    }


def main() -> None:
    banner("FINALIZE 2010-2025 MULTISOURCE MODEL")

    for path in [DATA_PATH, RF_PATH, XGB_PATH, PRED_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    # ------------------------------------------------------------
    # 1. Load exact final dataset
    # ------------------------------------------------------------
    banner("1. LOAD FINAL MULTISOURCE DATASET")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime", "date"],
    ).sort_values("datetime").reset_index(drop=True)

    feature_columns = prepare_features(df)

    X_all = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X_all.isna().any().any():
        bad = X_all.isna().sum()
        bad = bad[bad > 0].to_dict()
        raise RuntimeError(f"NaN/non-numeric model features found: {bad}")

    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise RuntimeError("Target contains NaN values.")

    # ------------------------------------------------------------
    # 2. Determine ACTUAL available 2025 test window
    # ------------------------------------------------------------
    banner("2. DETERMINE ACTUAL CHRONOLOGICAL SPLIT")

    test_mask = df["datetime"].dt.year == 2025
    train_mask = df["datetime"].dt.year < 2025

    if not test_mask.any():
        raise RuntimeError("No 2025 observations exist in the final dataset.")

    train_end = df.loc[train_mask, "datetime"].max()
    test_start = df.loc[test_mask, "datetime"].min()
    test_end = df.loc[test_mask, "datetime"].max()

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_end >= test_start:
        raise RuntimeError("Train/test temporal overlap detected.")

    print(f"Train rows : {len(train_df):,}")
    print(f"Train end  : {train_end}")
    print(f"Test rows  : {len(test_df):,}")
    print(f"Test start : {test_start}")
    print(f"Test end   : {test_end}")

    # Explicitly document the gap instead of hiding it.
    expected_after_train = train_end + pd.Timedelta(hours=1)
    gap_hours = int((test_start - expected_after_train).total_seconds() / 3600)

    print(f"Gap between train and test: {gap_hours:,} hours")

    # ------------------------------------------------------------
    # 3. Load saved model artifacts
    # ------------------------------------------------------------
    banner("3. LOAD SAVED MODEL ARTIFACTS")

    rf_model = joblib.load(RF_PATH)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(XGB_PATH))

    print(f"RF loaded : {type(rf_model).__name__}")
    print(f"XGB loaded: {type(xgb_model).__name__}")

    if hasattr(rf_model, "n_features_in_"):
        if int(rf_model.n_features_in_) != len(feature_columns):
            raise RuntimeError(
                "RF feature-count mismatch: "
                f"model={rf_model.n_features_in_}, "
                f"dataset={len(feature_columns)}"
            )

    # ------------------------------------------------------------
    # 4. Reproduce saved test predictions
    # ------------------------------------------------------------
    banner("4. VERIFY SAVED TEST PREDICTIONS")

    saved_pred = pd.read_csv(
        PRED_PATH,
        parse_dates=["datetime"],
    ).sort_values("datetime").reset_index(drop=True)

    expected_test_dates = test_df["datetime"].reset_index(drop=True)

    if len(saved_pred) != len(test_df):
        raise RuntimeError(
            "Saved prediction row count does not match actual 2025 test rows."
        )

    if not saved_pred["datetime"].equals(expected_test_dates):
        raise RuntimeError(
            "Saved prediction timestamps do not match the actual 2025 test window."
        )

    X_test = test_df[feature_columns].apply(pd.to_numeric)

    rf_now = np.clip(
        rf_model.predict(X_test),
        0,
        None,
    )

    xgb_now = np.clip(
        xgb_model.predict(X_test),
        0,
        None,
    )

    rf_saved = saved_pred["rf_multisource_prediction"].to_numpy()
    xgb_saved = saved_pred["xgb_multisource_prediction"].to_numpy()

    rf_diff = float(np.max(np.abs(rf_now - rf_saved)))
    xgb_diff = float(np.max(np.abs(xgb_now - xgb_saved)))

    print(f"Max RF prediction difference : {rf_diff:.12g}")
    print(f"Max XGB prediction difference: {xgb_diff:.12g}")

    if rf_diff > 1e-8:
        raise RuntimeError(
            "Saved RF predictions do not reproduce from the saved model."
        )

    if xgb_diff > 1e-5:
        raise RuntimeError(
            "Saved XGB predictions do not reproduce from the saved model."
        )

    # ------------------------------------------------------------
    # 5. Recalculate final actual metrics
    # ------------------------------------------------------------
    banner("5. RECALCULATE FINAL TEST METRICS")

    y_test = test_df[TARGET].to_numpy(dtype=float)

    rf_metrics = metrics(y_test, rf_now)
    rf_heavy = heavy_metrics(y_test, rf_now)

    xgb_metrics = metrics(y_test, xgb_now)
    xgb_heavy = heavy_metrics(y_test, xgb_now)

    print("ERA5 + IMERG Random Forest:")
    print(rf_metrics)
    print("Heavy:", rf_heavy)

    print()
    print("ERA5 + IMERG XGBoost:")
    print(xgb_metrics)
    print("Heavy:", xgb_heavy)

    # ------------------------------------------------------------
    # 6. Choose deployment model
    # ------------------------------------------------------------
    # RF has the lower RMSE, higher R2, and lower heavy-rain MAE/RMSE
    # in the actual saved 2025 holdout. XGB is retained as backup.
    deployment_model = "random_forest"

    deployment_reason = {
        "criterion": [
            "lower RMSE on the 2025 chronological holdout",
            "higher R2 on the 2025 chronological holdout",
            "lower heavy-rain MAE on the 2025 chronological holdout",
            "lower heavy-rain RMSE on the 2025 chronological holdout",
        ],
        "note": (
            "XGBoost has slightly lower overall MAE, so both models are "
            "retained; Random Forest is used as the primary deployment model "
            "because it is stronger on the other reported error measures."
        ),
    }

    # ------------------------------------------------------------
    # 7. Save exact feature list
    # ------------------------------------------------------------
    banner("6. FREEZE FEATURE ORDER")

    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(
            {
                "target": TARGET,
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Feature list saved: {FEATURE_COLUMNS_PATH}")
    print(f"Feature count: {len(feature_columns)}")

    # ------------------------------------------------------------
    # 8. Save deployment bundle
    # ------------------------------------------------------------
    banner("7. SAVE DEPLOYMENT BUNDLE")

    bundle = {
        "model_type": "RandomForestRegressor",
        "model": rf_model,
        "feature_columns": feature_columns,
        "target": TARGET,
        "heavy_threshold_mm": 5.0,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "dataset_path": str(DATA_PATH),
        "satellite_product": "NASA GPM IMERG Final Daily V07",
        "satellite_temporal_usage": (
            "Previous completed daily IMERG features only; "
            "no same-day/future leakage."
        ),
        "xgb_backup_path": str(XGB_PATH),
    }

    joblib.dump(
        bundle,
        BUNDLE_PATH,
    )

    print(f"Bundle saved: {BUNDLE_PATH}")

    # ------------------------------------------------------------
    # 9. Write final metadata
    # ------------------------------------------------------------
    banner("8. WRITE FINAL METADATA")

    metadata = {
        "project": "Mumbai rainfall/flood early warning prototype",
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max()),
        },
        "chronological_evaluation": {
            "train_rows": int(len(train_df)),
            "train_start": str(train_df["datetime"].min()),
            "train_end": str(train_end),
            "test_rows": int(len(test_df)),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "gap_hours_between_train_and_test": gap_hours,
        },
        "target": TARGET,
        "satellite_source": {
            "name": "NASA GPM IMERG Final Daily V07",
            "feature_count_added": int(
                sum(1 for c in feature_columns if c.startswith("imerg_"))
            ),
            "leakage_policy": (
                "Only previous completed IMERG days are used."
            ),
        },
        "deployment": {
            "primary_model": deployment_model,
            "reason": deployment_reason,
            "bundle_path": str(BUNDLE_PATH),
            "backup_xgb_path": str(XGB_PATH),
        },
        "metrics_2025_holdout": {
            "random_forest": {
                "overall": rf_metrics,
                "heavy": rf_heavy,
            },
            "xgboost": {
                "overall": xgb_metrics,
                "heavy": xgb_heavy,
            },
        },
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 10. Update the previous metrics JSON with honest date metadata
    # ------------------------------------------------------------
    if METRICS_PATH.exists():
        try:
            previous = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            previous = {}

        previous["final_verified_evaluation"] = {
            "train_start": str(train_df["datetime"].min()),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "gap_hours_between_train_and_test": gap_hours,
            "note": (
                "The available ERA5 record has a temporal gap; "
                "the test is the available 2025 period, not a continuous "
                "2025-01-01 to 2025-09-30 block."
            ),
        }

        METRICS_PATH.write_text(
            json.dumps(
                previous,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    banner("FINAL MODEL FROZEN")

    print(f"Primary deployment model : {deployment_model}")
    print(f"Deployment bundle        : {BUNDLE_PATH}")
    print(f"Feature list             : {FEATURE_COLUMNS_PATH}")
    print(f"Metadata                 : {METADATA_PATH}")
    print()
    print(
        f"2025 test period        : {test_start} -> {test_end}"
    )
    print(
        f"2025 test observations  : {len(test_df):,}"
    )
    print()
    print("Random Forest:")
    print(f"  MAE  : {rf_metrics['MAE_mm']:.6f} mm")
    print(f"  RMSE : {rf_metrics['RMSE_mm']:.6f} mm")
    print(f"  R2   : {rf_metrics['R2']:.6f}")
    print(f"  Heavy MAE : {rf_heavy['MAE_mm']:.6f} mm")
    print(f"  Heavy RMSE: {rf_heavy['RMSE_mm']:.6f} mm")
    print()
    print("XGBoost backup:")
    print(f"  MAE  : {xgb_metrics['MAE_mm']:.6f} mm")
    print(f"  RMSE : {xgb_metrics['RMSE_mm']:.6f} mm")
    print(f"  R2   : {xgb_metrics['R2']:.6f}")
    print()
    print("=" * 80)
    print("FINALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
