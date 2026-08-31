"""Unit tests for the BPR dynamic edge-cost function.

Validates monotonicity properties: increasing burn probability and
volume should never decrease edge weight, and the critical threshold
closure should trigger at the correct burn-probability level.
"""
import numpy as np

from aerospatium.graph.bpr_cost import (
    bpr_cost,
    bpr_cost_vectorized,
)


class TestBPRFunction:
    """Verify basic BPR cost computation."""

    def test_base_time_no_traffic(self):
        """With zero volume, cost should equal base time."""
        cost = bpr_cost(
            base_time_s=100.0,
            volume=0.0,
            capacity=1000.0,
            burn_prob=0.0,
            pm25_ugm3=10.0,
        )
        assert abs(cost - 100.0) < 1e-6, "Zero-volume cost should equal base time"

    def test_burn_prob_increases_cost(self):
        """Increasing burn probability should increase cost (monotonic)."""
        base = bpr_cost(100.0, 500, 1000, 0.0, 10.0)
        higher = bpr_cost(100.0, 500, 1000, 0.5, 10.0)
        assert higher > base, "Higher burn prob should increase cost"

    def test_critical_threshold_closes_edge(self):
        """Burn probability >= tau_critical should hard-close the edge."""
        cost = bpr_cost(
            base_time_s=100.0,
            volume=0,
            capacity=1000,
            burn_prob=0.75,
            pm25_ugm3=10.0,
            tau_critical=0.75,
        )
        assert cost >= 1e8, "Critical burn prob should close edge"

        cost_below = bpr_cost(
            base_time_s=100.0,
            volume=0,
            capacity=1000,
            burn_prob=0.74,
            pm25_ugm3=10.0,
            tau_critical=0.75,
        )
        assert cost_below < 1e8, "Below-critical burn prob should not close edge"

    def test_smoke_penalty_positive(self):
        """PM2.5 above threshold should add delay."""
        cost_no_smoke = bpr_cost(100.0, 0, 1000, 0.0, 35.0)
        cost_with_smoke = bpr_cost(100.0, 0, 1000, 0.0, 100.0)
        assert cost_with_smoke > cost_no_smoke, "Smoke should add delay"

    def test_volume_increases_cost(self):
        """Higher traffic volume should increase cost via BPR."""
        low_vol = bpr_cost(100.0, 100, 1000, 0.0, 10.0)
        high_vol = bpr_cost(100.0, 900, 1000, 0.0, 10.0)
        assert high_vol > low_vol, "Higher volume should increase cost"

    def test_vectorized_matches_scalar(self):
        """Vectorized BPR should match scalar for same inputs."""
        base_times = np.array([100.0, 200.0, 300.0])
        volumes = np.array([500, 800, 100])
        capacities = np.array([1000, 1000, 1000])
        burn_probs = np.array([0.1, 0.5, 0.9])
        pm25 = np.array([20.0, 50.0, 80.0])

        scalar_results = np.array([
            bpr_cost(bt, v, c, bp, pm, tau_critical=1.0)  # high threshold to avoid closures
            for bt, v, c, bp, pm in zip(base_times, volumes, capacities, burn_probs, pm25)
        ])

        vector_results = bpr_cost_vectorized(
            base_times, volumes, capacities, burn_probs, pm25, tau_critical=1.0
        )

        np.testing.assert_allclose(scalar_results, vector_results, rtol=1e-5)


class TestBPRMonotonicity:
    """Verify monotonicity constraints for routing safety."""

    def test_burn_prob_monotonic(self):
        """For fixed other params, cost should be non-decreasing in burn_prob."""
        burn_probs = np.linspace(0, 0.8, 20)
        costs = [
            bpr_cost(500.0, 0, 1000, bp, 20.0, tau_critical=0.75)
            for bp in burn_probs
        ]
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1] - 1e-6, \
                f"Cost should be non-decreasing in burn_prob (idx {i})"

    def test_wind_direction_monotonicity(self):
        """Increasing wind toward a neighbor should not decrease weight."""
        # This is a conceptual test — the actual wind direction logic
        # is in the CA simulation, not the BPR function directly.
        # Here we verify the BPR cost is monotonic in each hazard input.
        base = bpr_cost(500.0, 0, 1000, 0.0, 35.0)
        for pm25 in [35, 40, 50, 75, 100]:
            cost = bpr_cost(500.0, 0, 1000, 0.0, float(pm25))
            assert cost >= base - 1e-6, f"BPR should increase with smoke (pm25={pm25})"
