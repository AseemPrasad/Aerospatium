"""XGBoost survival-analysis training pipeline.

Implements training of Cox proportional hazards and Accelerated Failure
Time (AFT) models using xgboost, with Optuna-based hyperparameter
tuning and cross-validation by fire-event grouping.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from aerospatium.ml.evaluation import (
    SurvivalEvaluator,
    concordance_index_censored,
)


class SurvivalModelTrainer:
    """Trains XGBoost survival models (Cox / AFT) with group-aware
    cross-validation.

    Parameters
    ----------
    feature_cols : list[str]
        Column names for model features.
    group_col : str
        Column name for grouping (e.g. fire_id) to prevent leakage.
    """

    def __init__(
        self,
        feature_cols: list[str],
        group_col: str = "fire_id",
    ):
        self.feature_cols = feature_cols
        self.group_col = group_col
        self.cox_model = None
        self.aft_model = None
        self.evaluator = SurvivalEvaluator()
        self._best_model = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        event_col: str = "burned",
        time_col: str = "arrival_min",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
        """Extract feature matrix and time/event arrays from a training frame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with features, event indicator, and time-to-event.
        event_col : str
            Column for the event indicator (1 = burned, 0 = censored).
        time_col : str
            Column for the observed time (minutes).

        Returns
        -------
        X : pd.DataFrame
        y_time : pd.Series
        y_event : pd.Series
        feature_names : list[str]
        """
        X = df[self.feature_cols].copy()
        y_time = df[time_col].copy()
        y_event = df[event_col].copy().astype(int)
        return X, y_time, y_event, self.feature_cols

    def train_cox(
        self,
        X: pd.DataFrame,
        y_time: pd.Series,
        y_event: pd.Series,
        groups: pd.Series | None = None,
        **kwargs,
    ) -> object:
        """Train an XGBoost Cox proportional hazards model.

        Parameters
        ----------
        X : pd.DataFrame
        y_time : pd.Series
        y_event : pd.Series
        groups : pd.Series, optional
            Fire IDs for group-aware splitting.
        **kwargs
            XGBoost parameters.

        Returns
        -------
        Trained Booster object.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for training")

        y_time = np.asarray(y_time, dtype=np.float64)
        y_event = np.asarray(y_event, dtype=np.float64)

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
              "tree_method": "hist",
              "max_depth": kwargs.get("max_depth", 6),
              "learning_rate": kwargs.get("learning_rate", 0.1),
              "subsample": kwargs.get("subsample", 0.8),
              "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
              "min_child_weight": kwargs.get("min_child_weight", 1),
          }

        dtrain = xgb.DMatrix(X, label=y_time, weight=y_event)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=kwargs.get("n_estimators", 200),
            early_stopping_rounds=10,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )

        self.cox_model = model
        self._best_model = model
        return model

    def train_aft(
        self,
        X: pd.DataFrame,
        y_time: pd.Series,
        y_event: pd.Series,
        groups: pd.Series | None = None,
        **kwargs,
    ) -> object:
        """Train an XGBoost Accelerated Failure Time model.

        Parameters
        ----------
        X : pd.DataFrame
        y_time : pd.Series
        y_event : pd.Series
        groups : pd.Series, optional
            Fire IDs for group-aware splitting.
        **kwargs
            XGBoost parameters.

        Returns
        -------
        Trained Booster object.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for training")

        y_time = np.asarray(y_time, dtype=np.float64)
        y_event = np.asarray(y_event, dtype=np.float64)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": "hist",
            "aft_loss_distribution": kwargs.get("aft_loss_distribution", "normal"),
            "aft_loss_distribution_scale": kwargs.get("aft_loss_distribution_scale", 1.0),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
        }

        dtrain = xgb.DMatrix(X, label=y_time, weight=y_event)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=kwargs.get("n_estimators", 200),
            early_stopping_rounds=10,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )

        self.aft_model = model
        self._best_model = model
        return model

    def cross_validate_grouped(
        self,
        df: pd.DataFrame,
        event_col: str = "burned",
        time_col: str = "arrival_min",
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Group-aware cross-validation using GroupKFold.

        Parameters
        ----------
        df : pd.DataFrame
        event_col : str
        time_col : str
        n_splits : int

        Returns
        -------
        dict with mean and std of C-index across folds.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required")

        from sklearn.model_selection import GroupKFold

        X, y_time, y_event, _ = self.prepare_data(df, event_col, time_col)
        groups = df[self.group_col].values

        gkf = GroupKFold(n_splits=n_splits)
        c_indices = []

        for train_idx, val_idx in gkf.split(X, y_time, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_time, y_val_time = y_time.iloc[train_idx], y_time.iloc[val_idx]
            y_train_event, y_val_event = y_event.iloc[train_idx], y_event.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train_time, weight=y_train_event)
            dval = xgb.DMatrix(X_val, label=y_val_time, weight=y_val_event)

            params = {
                "objective": "survival:cox",
                "tree_method": "hist",
                "max_depth": 6,
                "learning_rate": 0.1,
                "eval_metric": "cox-nloglik",
            }
            model = xgb.train(
                params, dtrain, num_boost_round=200,
                evals=[(dval, "eval")], early_stopping_rounds=20, verbose_eval=False,
            )
            preds = model.predict(dval)
            c_idx = concordance_index_censored(
                y_val_time.values, y_val_event.values, preds
            )
            c_indices.append(c_idx)

        return {
            "mean_c_index": float(np.mean(c_indices)),
            "std_c_index": float(np.std(c_indices)),
            "fold_scores": [float(c) for c in c_indices],
        }

    def predict_arrival_times(self, X: pd.DataFrame) -> np.ndarray:
        """Predict (negative) risk scores — convert to arrival time proxy.

        For Cox model: higher risk → shorter survival → earlier arrival.
        Returns predicted risk score; higher = earlier arrival.
        """
        if self._best_model is None:
            raise RuntimeError("Model not trained yet. Call train_cox() or train_aft() first.")

        dmatrix = xgb.DMatrix(X)
        return self._best_model.predict(dmatrix)

    def save(
        self,
        model_path: str,
        metrics_path: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Save trained model and optional metrics/metadata.

        Parameters
        ----------
        model_path : str
            Path to save model (uses Booster.save_model).
        metrics_path : str, optional
            Path to write JSON metrics.
        metadata : dict, optional
            Additional metadata to include.
        """
        if self._best_model is None:
            raise RuntimeError("No model to save")

        self._best_model.save_model(model_path)

        if metrics_path:
            output = {
                "metrics": self.evaluator.metrics,
                "model_type": metadata.get("model_type", "cox") if metadata else "cox",
                "feature_cols": self.feature_cols,
            }
            if metadata:
                output["metadata"] = metadata
            with open(metrics_path, "w") as f:
                json.dump(output, f, indent=2)

    @classmethod
    def load(cls, model_path: str, feature_cols: list[str]) -> SurvivalModelTrainer:
        """Load a saved model and reconstruct the trainer.

        Parameters
        ----------
        model_path : str
            Path to saved XGBoost model.
        feature_cols : list[str]
            Feature column names.

        Returns
        -------
        SurvivalModelTrainer with loaded model.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required")

        trainer = cls(feature_cols=feature_cols)
        model = xgb.Booster()
        model.load_model(model_path)
        trainer._best_model = model
        trainer.cox_model = model
        return trainer
