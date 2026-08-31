"""Time-to-burn spatial inference engine.

Wraps trained survival models to produce per-H3-cell burn probability
estimates across multiple forecast horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from aerospatium.ml.survival_model import SurvivalModelTrainer


class SpatialInferencesEngine:
    """Produces spatial burn-probability rasters from survival models.

    Parameters
    ----------
    model : xgboost.Booster or SurvivalModelTrainer
        Trained survival model.
    feature_cols : list[str]
        Feature column names matching training.
    """

    def __init__(self, model, feature_cols: list[str]):
        self.feature_cols = feature_cols

        if isinstance(model, SurvivalModelTrainer):
            self.trainer = model
            self._booster = model._best_model
        elif HAS_XGBOOST and isinstance(model, xgb.Booster):
            self.trainer = None
            self._booster = model
        else:
            self.trainer = model
            self._booster = getattr(model, "_best_model", None) or getattr(model, "cox_model", None)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw risk scores (higher = shorter survival).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with columns matching feature_cols.

        Returns
        -------
        np.ndarray, shape (N,)
            Risk scores.
        """
        if self._booster is None:
            raise RuntimeError("No model loaded")

        dmatrix = xgb.DMatrix(X[self.feature_cols])
        return self._booster.predict(dmatrix)

    def predict_burn_probability(
        self,
        X: pd.DataFrame,
        arrival_times: np.ndarray | None = None,
        horizons_hours: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute burn probability at each forecast horizon.

        Uses a sigmoid transform of arrival times, or if arrival
        times are not provided, derives them from risk scores via
        an inverse-Cox baseline hazard approximation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        arrival_times : np.ndarray, optional
            Pre-computed arrival times in minutes. If None, estimated
            from risk scores.
        horizons_hours : list[float]
            Forecast horizons. Default [0, 3, 6, 12].

        Returns
        -------
        dict mapping horizon key (e.g. "burn_prob_3h") to np.ndarray.
        """
        if horizons_hours is None:
            horizons_hours = [0.0, 3.0, 6.0, 12.0]

        if arrival_times is None:
            risk_scores = self.predict_risk(X)
            # Inverse relationship: higher risk → shorter survival
            # Calibrate a rough mapping from risk to arrival time
            median_risk = np.median(risk_scores)
            mad = np.mean(np.abs(risk_scores - median_risk))
            # Map risk to estimated arrival time (minutes)
            # Higher risk → lower estimated arrival time
            normalized_risk = (risk_scores - median_risk) / (mad + 1e-8)
            # Survival time ~ baseline * exp(-risk)
            base_hours = 8.0
            arrival_times = base_hours * 60.0 * np.exp(-normalized_risk * 0.3)
            arrival_times = np.clip(arrival_times, 30.0, 720.0)

        arrival_times = np.asarray(arrival_times, dtype=np.float64)

        probabilities = {}
        for h in horizons_hours:
            horizon_min = h * 60.0
            # P(Burn <= t) = 1 - S(t) where S(t) = exp(-H0(t) * exp(risk))
            # Simplified: use sigmoid around arrival time
            diff = (horizon_min - arrival_times) / 30.0  # 30-min transition width
            diff = np.clip(diff, -10.0, 10.0)
            probs = 1.0 / (1.0 + np.exp(-diff))
            key = f"burn_prob_{int(h)}h"
            probabilities[key] = probs

        return probabilities

    def export_probability_grid(
        self,
        X: pd.DataFrame,
        h3_ids: np.ndarray | pd.Series,
        arrival_times: np.ndarray | None = None,
        horizons_hours: list[float] | None = None,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Export a full hazard probability DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with H3 static features.
        h3_ids : array-like
            H3 cell identifiers aligned with X.
        arrival_times : np.ndarray, optional
        horizons_hours : list[float], optional
        output_path : str, optional
            If provided, saves to Parquet.

        Returns
        -------
        pd.DataFrame
            With h3_id, centroid coordinates, and burn_prob columns.
        """
        probs = self.predict_burn_probability(X, arrival_times, horizons_hours)

        df = pd.DataFrame({"h3_id": h3_ids})
        for key, vals in probs.items():
            df[key] = vals.astype(np.float32)

        if output_path:
            df.to_parquet(output_path, compression="snappy")

        return df
