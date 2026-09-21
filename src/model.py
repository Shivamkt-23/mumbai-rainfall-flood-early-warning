import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(df, feature_cols):
    split = max(1, int(len(df) * 0.8))
    train = df.iloc[:split]
    test = df.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[feature_cols], train["next_hour_rain"])
    pred = np.maximum(model.predict(test[feature_cols]), 0)

    test = test.copy()
    test["predicted_rain"] = pred

    mae = mean_absolute_error(test["next_hour_rain"], pred)
    rmse = np.sqrt(mean_squared_error(test["next_hour_rain"], pred))
    return model, train, test, {"MAE": mae, "RMSE": rmse}
