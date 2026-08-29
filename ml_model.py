"""
ML MODEL: SIC Ensemble — Random Forest ensemble for Sea Ice Concentration prediction.

Adapted from the ML_workflow.ipynb notebook. Uses an ensemble of 5
RandomForestRegressor models trained on bootstrap samples, providing
both point predictions and uncertainty estimates.

Features: latitude, longitude, SST, wind_speed, current_u, current_v
Target:   sea ice concentration (SIC, 0-100%)
"""
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ml_models")
MODEL_PATH = os.path.join(MODEL_DIR, "sic_ensemble.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")


class SICEnsemble:
    """
    Ensemble of Random Forest models for SIC prediction with uncertainty.

    Each model is trained on a bootstrap sample of the training data.
    Prediction uncertainty is estimated from the spread across models.
    """

    def __init__(self, n_models=5):
        self.models = []
        for i in range(n_models):
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=i,
                n_jobs=-1,
            )
            self.models.append(model)

    def train(self, X, y):
        """Train the ensemble on a DataFrame X and target Series y."""
        for i, model in enumerate(self.models):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_bootstrap = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            model.fit(X_bootstrap, y_bootstrap)
            print(f"  Model {i + 1}/{len(self.models)} trained")

    def predict(self, X):
        """
        Predict SIC with uncertainty.

        Returns dict with:
          - prediction: mean across models (clipped 0-100)
          - uncertainty: std across models
          - lower: prediction - 1.96 * uncertainty (clipped 0-100)
          - upper: prediction + 1.96 * uncertainty (clipped 0-100)
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)

        lower = mean_prediction - 1.96 * uncertainty
        upper = mean_prediction + 1.96 * uncertainty

        mean_prediction = np.clip(mean_prediction, 0, 100)
        lower = np.clip(lower, 0, 100)
        upper = np.clip(upper, 0, 100)

        return {
            "prediction": mean_prediction,
            "uncertainty": uncertainty,
            "lower": lower,
            "upper": upper,
        }

    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble on test data. Returns (mae, rmse, r2)."""
        result = self.predict(X_test)
        prediction = result["prediction"]

        mae = mean_absolute_error(y_test, prediction)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        r2 = r2_score(y_test, prediction)

        return mae, rmse, r2

    def save(self, path=None):
        """Save the trained ensemble to disk."""
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path=None):
        """Load a trained ensemble from disk."""
        path = path or MODEL_PATH
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def get_feature_importance(self, feature_names):
        """Get average feature importance across all models."""
        importances = np.array([m.feature_importances_ for m in self.models])
        mean_imp = importances.mean(axis=0)
        return dict(zip(feature_names, mean_imp.tolist()))


FEATURES = ["latitude", "longitude", "sst", "wind_speed", "current_u", "current_v"]
TARGET = "sic"
