from pathlib import Path
import json
import joblib
import numpy as np


class ShardedRandomForestRegressor:
    """Tree-preserving RandomForest wrapper that loads shards once per process.

    The shard contents are unchanged. The only optimization is keeping the
    already-loaded DecisionTree estimators in memory so repeated predictions
    do not re-read ~900 MB of shard files from disk for every location.
    """

    def __init__(self, shard_dir):
        self.shard_dir = Path(shard_dir)
        metadata_path = self.shard_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing shard metadata: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.n_estimators = int(metadata["n_estimators"])
        self.n_features_in_ = int(metadata["n_features"])
        self.feature_names_in_ = np.array(metadata["feature_names"], dtype=object)
        self.shard_files = sorted(self.shard_dir.glob("shard_*.joblib"))
        if not self.shard_files:
            raise FileNotFoundError(f"No RF shards found in {self.shard_dir}")

        self._estimators = None

    def _ensure_loaded(self):
        if self._estimators is not None:
            return

        estimators = []
        for shard_file in self.shard_files:
            shard = joblib.load(shard_file)
            estimators.extend(shard["estimators"])
            del shard

        if len(estimators) != self.n_estimators:
            raise ValueError(
                f"Expected {self.n_estimators} trees, found {len(estimators)}"
            )

        self._estimators = estimators

    def predict(self, X):
        if hasattr(X, "loc"):
            missing = [
                c for c in self.feature_names_in_
                if c not in X.columns
            ]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            X = X.loc[:, list(self.feature_names_in_)].to_numpy()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        self._ensure_loaded()

        total = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self._estimators:
            total += tree.predict(X)

        return total / len(self._estimators)
