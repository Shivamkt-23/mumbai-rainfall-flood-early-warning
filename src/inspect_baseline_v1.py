from pathlib import Path
import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rainfall_model_v1.pkl"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "rainfall_features_v1.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_v1_feature_importance.csv"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading V1 model...")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("V1 FEATURE IMPORTANCE")
    print("=" * 55)

    for i, row in importance.iterrows():

        print(
            f"{i + 1:2d}. "
            f"{row['feature']:<25} "
            f"{row['importance']:.6f}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    importance.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()