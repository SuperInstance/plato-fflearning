"""Tests for PLATO Forward-Forward Learning."""

import pytest
from plato_fflearning import ForwardForwardLearner


class TestGoodnessTracking:
    """Test goodness score tracking."""

    def test_initial_goodness_is_0_5(self):
        """New agents start with default goodness of 0.5."""
        ff = ForwardForwardLearner()
        state = ff.get_learning_state("new_agent")
        assert state["goodness"] == 0.5

    def test_positive_pass_increases_goodness(self):
        """Positive pass should increase goodness."""
        ff = ForwardForwardLearner()
        ff.state["test_agent"] = 0.5  # Reset to known state
        result = ff.positive_pass("test_agent", "good result")
        assert result["goodness"] > 0.5
        assert result["pass"] == "positive"

    def test_negative_pass_decreases_goodness(self):
        """Negative pass should decrease goodness."""
        ff = ForwardForwardLearner()
        ff.state["test_agent"] = 0.5  # Reset to known state
        result = ff.negative_pass("test_agent", "bad result")
        assert result["goodness"] < 0.5
        assert result["pass"] == "negative"

    def test_goodness_clamped_to_0_1(self):
        """Goodness should never go below 0 or above 1."""
        ff = ForwardForwardLearner()
        # Push to minimum
        ff.state["agent_min"] = 0.01
        ff.negative_pass("agent_min", "very bad")
        assert ff.state["agent_min"] >= 0.0

        # Push to maximum
        ff.state["agent_max"] = 0.99
        ff.positive_pass("agent_max", "very good")
        assert ff.state["agent_max"] <= 1.0

    def test_goodness_decay_applies(self):
        """Goodness decays by GOODNESS_DECAY factor."""
        ff = ForwardForwardLearner()
        ff.state["decay_test"] = 0.5
        ff.positive_pass("decay_test", "test")  # delta = 0.15
        expected = (0.5 * 0.95) + 0.15  # decay + boost
        assert abs(ff.state["decay_test"] - expected) < 0.001


class TestThresholdBehavior:
    """Test threshold and reinforcement logic."""

    def test_threshold_exceeded_on_high_goodness(self):
        """When goodness > THRESHOLD, tiles should be reinforced."""
        ff = ForwardForwardLearner()
        ff.state["thresh_agent"] = 0.65  # Near threshold
        result = ff.positive_pass(
            "thresh_agent",
            "good outcome",
            associated_tiles=["tile_1", "tile_2"]
        )
        assert result["threshold_exceeded"] is True
        assert len(result["tiles_reinforced"]) == 2

    def test_threshold_not_exceeded_below(self):
        """When goodness <= THRESHOLD, no reinforcement."""
        ff = ForwardForwardLearner()
        ff.state["low_agent"] = 0.3
        result = ff.positive_pass("low_agent", "test", associated_tiles=["tile_1"])
        assert result["threshold_exceeded"] is False
        assert result["tiles_reinforced"] == []


class TestLearningCycle:
    """Test complete learning cycles."""

    def test_run_learning_cycle_positive(self):
        """run_learning_cycle with real_outcome=True calls positive_pass."""
        ff = ForwardForwardLearner()
        ff.state["cycle_agent"] = 0.5
        result = ff.run_learning_cycle("cycle_agent", True, "positive result")
        assert result["pass"] == "positive"

    def test_run_learning_cycle_negative(self):
        """run_learning_cycle with real_outcome=False calls negative_pass."""
        ff = ForwardForwardLearner()
        ff.state["cycle_agent"] = 0.5
        result = ff.run_learning_cycle("cycle_agent", False, "negative result")
        assert result["pass"] == "negative"


class TestFleetState:
    """Test fleet-wide learning state."""

    def test_fleet_state_empty(self):
        """Empty fleet returns default values."""
        ff = ForwardForwardLearner()
        fleet = ff.get_fleet_learning_state()
        assert fleet["total_agents"] == 0
        assert fleet["fleet_goodness_avg"] == 0.5  # default

    def test_fleet_state_aggregates(self):
        """Fleet state correctly aggregates agent goodness."""
        ff = ForwardForwardLearner()
        ff.state["agent_1"] = 0.8
        ff.state["agent_2"] = 0.4
        fleet = ff.get_fleet_learning_state()
        assert fleet["total_agents"] == 2
        assert fleet["fleet_goodness_avg"] == 0.6
        assert fleet["high_reliability_agents"] == 1
        assert fleet["low_reliability_agents"] == 1


class TestGoodnessLevels:
    """Test goodness-to-level mapping."""

    def test_critical_level(self):
        ff = ForwardForwardLearner()
        ff.state["crit"] = 0.15
        assert ff.get_learning_state("crit")["level"] == "critical"

    def test_low_level(self):
        ff = ForwardForwardLearner()
        ff.state["low"] = 0.3
        assert ff.get_learning_state("low")["level"] == "low"

    def test_moderate_level(self):
        ff = ForwardForwardLearner()
        ff.state["mod"] = 0.5
        assert ff.get_learning_state("mod")["level"] == "moderate"

    def test_high_level(self):
        ff = ForwardForwardLearner()
        ff.state["high"] = 0.7
        assert ff.get_learning_state("high")["level"] == "high"

    def test_exceptional_level(self):
        ff = ForwardForwardLearner()
        ff.state["exc"] = 0.85
        assert ff.get_learning_state("exc")["level"] == "exceptional"


class TestTileCreation:
    """Test that tiles are created with correct structure."""

    def test_positive_tile_structure(self, monkeypatch):
        """Positive pass creates tile with correct fields."""
        captured = {}

        def mock_post(url, json, **kwargs):
            captured["json"] = json
            class Resp:
                status_code = 200
            return Resp()

        import plato_fflearning
        original = plato_fflearning.requests.post
        plato_fflearning.requests.post = mock_post

        try:
            ff = ForwardForwardLearner(plato_url="http://localhost:8847")
            ff.positive_pass("test_agent", "test experience", domain="test_domain")
            tile = captured["json"]
            assert tile["pass_type"] == "positive"
            assert tile["confidence"] == 0.9
            assert tile["agent"] == "test_agent"
            assert tile["domain"] == "test_domain"
        finally:
            plato_fflearning.requests.post = original

    def test_negative_tile_structure(self, monkeypatch):
        """Negative pass creates tile with correct fields."""
        captured = {}

        def mock_post(url, json, **kwargs):
            captured["json"] = json
            class Resp:
                status_code = 200
            return Resp()

        import plato_fflearning
        original = plato_fflearning.requests.post
        plato_fflearning.requests.post = mock_post

        try:
            ff = ForwardForwardLearner(plato_url="http://localhost:8847")
            ff.negative_pass("test_agent", "negative experience")
            tile = captured["json"]
            assert tile["pass_type"] == "negative"
            assert tile["confidence"] == 0.3
        finally:
            plato_fflearning.requests.post = original