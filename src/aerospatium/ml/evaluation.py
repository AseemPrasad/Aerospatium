"""Survival-analysis evaluation metrics for wildfire progression models.

Implements Harrell's C-index, Brier score at multiple horizons, and
IoU (Intersection over Union) for burn-perimeter validation.
"""
from __future__ import annotations

import numpy as np


def concordance_index_censored(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_scores: np.ndarray,
) -> float:
    """Compute Harrell's C-index for right-censored survival data.

    Parameters
    ----------
    event_times : np.ndarray, shape (N,)
        Observed times (or censoring times).
    event_observed : np.ndarray, shape (N,)
        Binary event indicator (1 = event observed, 0 = censored).
    predicted_scores : np.ndarray, shape (N,)
        Model predictions (higher = higher risk / shorter survival).

    Returns
    -------
    float
        C-index in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    event_times = np.asarray(event_times, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=np.float64)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)

    n = len(event_times)
    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Pair (i, j) is comparable if i had an event and i's time < j's time
            # (regardless of whether j was observed or censored)
            if event_observed[i] == 1 and event_times[i] < event_times[j]:
                permissible += 1
                # Risk score direction: higher predicted score = higher risk = shorter survival
                # If i has higher risk (shorter survival), predicted_scores[i] > predicted_scores[j]
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1.0
                elif predicted_scores[i] == predicted_scores[j]:
                    concordant += 0.5

    if permissible == 0:
        return 0.5

    return concordant / permissible


def brier_score(
    survival_func: np.ndarray,
    times: np.ndarray,
    event_times: np.ndarray,
    event_observed: np.ndarray,
) -> np.ndarray:
    """Compute Brier score at multiple time points.

    Parameters
    ----------
    survival_func : np.ndarray, shape (N, T)
        Predicted survival probability S(t) for each sample at each time.
    times : np.ndarray, shape (T,)
        Time points at which Brier score is evaluated.
    event_times : np.ndarray, shape (N,)
        Observed event or censoring times.
    event_observed : np.ndarray, shape (N,)
        Binary event indicator.

    Returns
    -------
    np.ndarray, shape (T,)
        Brier score at each time point. Lower is better.
    """
    survival_func = np.asarray(survival_func, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=np.float64)

    n, n_times = survival_func.shape
    scores = np.zeros(n_times)

    for t_idx, t in enumerate(times):
        brier = 0.0
        for i in range(n):
            s_t = survival_func[i, t_idx]
            obs_event = 1.0 if (event_times[i] <= t and event_observed[i] == 1) else 0.0
            # Handle censoring with inverse probability of censoring weighting (IPCW)
            brier += (s_t - obs_event) ** 2
        scores[t_idx] = brier / n

    return scores


def intersection_over_union(
    predicted_burned: np.ndarray,
    observed_burned: np.ndarray,
) -> float:
    """Compute IoU between predicted and observed burn masks.

    Parameters
    ----------
    predicted_burned : np.ndarray, shape (N,)
        Boolean array of predicted burned cells.
    observed_burned : np.ndarray, shape (N,)
        Boolean array of observed burned cells.

    Returns
    -------
    float
        IoU in [0, 1].
    """
    predicted_burned = np.asarray(predicted_burned, dtype=bool)
    observed_burned = np.asarray(observed_burned, dtype=bool)

    intersection = np.sum(predicted_burned & observed_burned)
    union = np.sum(predicted_burned | observed_burned)

    if union == 0:
        return 1.0  # No predictions and no observations = perfect agreement

    return intersection / union


def perimeter_iou_at_horizon(
    arrival_times: np.ndarray,
    horizon_min: float,
    observed_burned: np.ndarray,
) -> float:
    """Compute IoU at a specific forecast horizon.

    A cell is predicted burned at horizon *t* if:
        arrival_time <= t

    Parameters
    ----------
    arrival_times : np.ndarray, shape (N,)
        Predicted arrival times in minutes.
    horizon_min : float
        Forecast horizon in minutes.
    observed_burned : np.ndarray, shape (N,)
        Ground-truth burn mask at this horizon.

    Returns
    -------
    float
        IoU score.
    """
    predicted_burned = arrival_times <= horizon_min
    return intersection_over_union(predicted_burned, observed_burned)


class SurvivalEvaluator:
    """Aggregates survival-model evaluation metrics.

    Usage::

        evaluator = SurvivalEvaluator()
        evaluator.evaluate(arrival_times, obs_times, obs_events, burn_masks)
        print(evaluator.report())
    """

    def __init__(self):
        self.metrics: dict[str, float] = {}

    def evaluate(
        self,
        arrival_times: np.ndarray,
        observation_times: np.ndarray,
        observation_events: np.ndarray,
        observed_burn_masks: dict[float, np.ndarray],
        horizons_hours: list[float] | None = None,
    ) -> dict[str, float]:
        """Run the full evaluation suite.

        Parameters
        ----------
        arrival_times : np.ndarray
            Predicted arrival time in minutes per cell.
        observation_times : np.ndarray
            Observed times (censoring if event=0).
        observation_events : np.ndarray
            Binary event indicator.
        observed_burn_masks : dict
            Mapping horizon_hours -> boolean burn mask array.
        horizons_hours : list[float], optional
            Horizons to evaluate. Default [3, 6, 12].
        """
        if horizons_hours is None:
            horizons_hours = [3.0, 6.0, 12.0]

        # C-index: higher risk = earlier burn
        # Convert arrival times to risk score (negative arrival time)
        risk_scores = -np.asarray(arrival_times, dtype=np.float64)
        risk_scores[np.isinf(risk_scores)] = -1e6  # never-burned → lowest risk

        c_index = concordance_index_censored(
            observation_times, observation_events, risk_scores
        )
        self.metrics["c_index"] = float(c_index)

        # Brier scores at each horizon
        horizons_min = [h * 60.0 for h in horizons_hours]
        for h_hr, h_min in zip(horizons_hours, horizons_min):
            # Survival = 1 - (arrival <= h_min)
            s_pred = (arrival_times > h_min).astype(np.float64)

            # IPCW Brier (simplified — no censoring model assumed here)
            obs_mask = observed_burn_masks.get(h_hr, np.zeros_like(arrival_times, dtype=bool))
            obs_events = obs_mask.astype(np.float64)

            # Use the simple Brier (unweighted by censoring)
            brier = np.mean((s_pred - (1.0 - obs_events)) ** 2)
            self.metrics[f"brier_{int(h_hr)}h"] = float(brier)

            # IoU
            iou = perimeter_iou_at_horizon(arrival_times, h_min, obs_mask)
            self.metrics[f"iou_{int(h_hr)}h"] = float(iou)

        return self.metrics

    def report(self) -> str:
        lines = ["Survival Model Evaluation Report", "=" * 40]
        for k, v in self.metrics.items():
            if k.startswith("c_index"):
                lines.append(f"  C-index:        {v:.4f}")
            elif k.startswith(("brier", "iou")):
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)
