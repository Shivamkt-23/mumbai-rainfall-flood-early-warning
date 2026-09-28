from pathlib import Path
import gc
import json
import joblib
import numpy as np


class ShardedRandomForestRegressor:
    """Exact tree-preserving RF wrapper with bounded peak memory.

    Each prediction streams one 50-tree shard at a time, accumulates the
    predictions from those trees, and releases the shard before loading the
    next one. The model trees and averaging rule are unchanged; only the
    storage/loading strategy changes so Community Cloud does not need to keep
    the full ~938 MB serialized forest set resident at once.
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
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got "
                f"{X.shape[1] if X.ndim == 2 else 'invalid input'}"
            )

        total = np.zeros(X.shape[0], dtype=np.float64)
        tree_count = 0

        # Stream one shard at a time so peak RAM is bounded by one shard.
        for shard_file in self.shard_files:
            shard = joblib.load(shard_file)
            estimators = shard["estimators"]
            for tree in estimators:
                total += tree.predict(X)
                tree_count += 1

            del estimators
            del shard
            gc.collect()

        if tree_count != self.n_estimators:
            raise ValueError(
                f"Expected {self.n_estimators} trees, found {tree_count}"
            )

        return total / tree_count
