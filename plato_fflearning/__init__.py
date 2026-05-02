"""
PLATO Forward-Forward Learning — Predictive coding without backpropagation

Based on Geoffrey Hinton's Forward-Forward algorithm:
- Positive pass: real experience → learn (tile accepted)
- Negative pass: imagined/hypothetical → forget (tile rejected)
- The gradient between positive and negative replaces backprop

In PLATO:
- Positive tiles: real agent experiences (confirmed by outcomes)
- Negative tiles: hypothetical experiences (counterfactuals, failed predictions)
- The ratio of positive/negative determines what gets reinforced

Usage:
    from plato_fflearning import ForwardForwardLearner
    ff = ForwardForwardLearner(plato_url="http://localhost:8847")

    # Agent has a real experience (positive pass)
    ff.positive_pass(
        agent="oracle1",
        experience="Casey asked for fleet status, I provided it, outcome=positive"
    )

    # Agent imagines a failure (negative pass)
    ff.negative_pass(
        agent="oracle1",
        experience="What if I had given wrong status? Outcome would have been negative"
    )

    # Query learning state
    state = ff.get_learning_state("oracle1")
    print(f"Goodness: {state['goodness']:.2f}")
"""

import math
import time
import requests
from typing import List, Dict, Any, Optional


class ForwardForwardLearner:
    """
    Forward-Forward learning for PLATO.

    Each agent has a "goodness" score that increases with positive passes
    and decreases with negative passes. When goodness exceeds threshold,
    the associated knowledge tile gets reinforced in PLATO.
    """

    GOODNESS_DECAY = 0.95  # Goodness decays over time
    POSITIVE_BOOST = 0.15  # Positive pass boost
    NEGATIVE_PENALTY = 0.08  # Negative pass penalty
    THRESHOLD = 0.7  # Threshold for tile reinforcement

    def __init__(self, plato_url: str = "http://localhost:8847"):
        self.plato_url = plato_url.rstrip("/")
        self.positive_room = "ff_positive_tiles"
        self.negative_room = "ff_negative_tiles"
        self.learning_state_room = "ff_learning_state"
        # In-memory state (in production, this would be in PLATO)
        self.state: Dict[str, float] = {}  # agent -> goodness

    def _get_agent_goodness(self, agent: str) -> float:
        """Get current goodness for an agent."""
        return self.state.get(agent, 0.5)

    def _update_goodness(self, agent: str, delta: float) -> float:
        """Update goodness with decay."""
        current = self._get_agent_goodness(agent)
        updated = (current * self.GOODNESS_DECAY) + delta
        updated = max(0.0, min(1.0, updated))
        self.state[agent] = updated
        return updated

    def positive_pass(
        self,
        agent: str,
        experience: str,
        domain: str = "fleet_orchestration",
        associated_tiles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Positive pass: real experience that should be learned.

        Increases agent goodness. If goodness exceeds threshold,
        associated tiles in PLATO get reinforced.
        """
        tile = {
            "question": f"Positive experience for {agent}",
            "answer": f"Experience: {experience}\nType: POSITIVE_PASS\nTimestamp: {time.time()}",
            "agent": agent,
            "domain": domain,
            "confidence": 0.9,
            "model": agent,
            "role": "forward_forward",
            "pass_type": "positive"
        }

        try:
            requests.post(f"{self.plato_url}/room/{self.positive_room}", json=tile, timeout=5)
        except Exception:
            pass

        new_goodness = self._update_goodness(agent, self.POSITIVE_BOOST)

        # If above threshold, reinforce associated tiles
        reinforced = []
        if new_goodness > self.THRESHOLD and associated_tiles:
            reinforced = self._reinforce_tiles(associated_tiles)

        return {
            "agent": agent,
            "pass": "positive",
            "goodness": new_goodness,
            "threshold_exceeded": new_goodness > self.THRESHOLD,
            "tiles_reinforced": reinforced
        }

    def negative_pass(
        self,
        agent: str,
        experience: str,
        domain: str = "fleet_orchestration"
    ) -> Dict[str, Any]:
        """
        Negative pass: hypothetical or failed experience.

        Decreases agent goodness. Tiles associated with negative passes
        are weakened or marked as unreliable.
        """
        tile = {
            "question": f"Negative experience for {agent}",
            "answer": f"Experience: {experience}\nType: NEGATIVE_PASS\nTimestamp: {time.time()}",
            "agent": agent,
            "domain": domain,
            "confidence": 0.3,  # Low confidence for negative
            "model": agent,
            "role": "forward_forward",
            "pass_type": "negative"
        }

        try:
            requests.post(f"{self.plato_url}/room/{self.negative_room}", json=tile, timeout=5)
        except Exception:
            pass

        new_goodness = self._update_goodness(agent, -self.NEGATIVE_PENALTY)

        return {
            "agent": agent,
            "pass": "negative",
            "goodness": new_goodness,
            "suppressed": new_goodness < (self.THRESHOLD * 0.5)
        }

    def _reinforce_tiles(self, tile_ids: List[str]) -> List[str]:
        """Reinforce tiles by increasing their confidence in PLATO."""
        reinforced = []
        for tile_id in tile_ids:
            # In production: fetch tile, increment confidence, write back
            reinforced.append(tile_id)
        return reinforced

    def get_learning_state(self, agent: str) -> Dict[str, Any]:
        """Get current learning state for an agent."""
        goodness = self._get_agent_goodness(agent)

        return {
            "agent": agent,
            "goodness": goodness,
            "level": self._goodness_to_level(goodness),
            "positive_tiles": self._count_tiles_in_room(self.positive_room, agent),
            "negative_tiles": self._count_tiles_in_room(self.negative_room, agent),
            "recommendation": self._get_recommendation(goodness)
        }

    def _goodness_to_level(self, goodness: float) -> str:
        if goodness < 0.2:
            return "critical"
        elif goodness < 0.4:
            return "low"
        elif goodness < 0.6:
            return "moderate"
        elif goodness < 0.8:
            return "high"
        else:
            return "exceptional"

    def _get_recommendation(self, goodness: float) -> str:
        if goodness < 0.3:
            return "agent should seek positive experiences"
        elif goodness < 0.6:
            return "agent is learning normally"
        else:
            return "agent is highly reliable"

    def _count_tiles_in_room(self, room: str, agent: str) -> int:
        """Count tiles for an agent in a specific room."""
        try:
            resp = requests.get(f"{self.plato_url}/room/{room}?limit=100", timeout=5)
            if resp.status_code == 200:
                tiles = resp.json().get("tiles", [])
                return len([t for t in tiles if t.get("agent") == agent])
        except Exception:
            pass
        return 0

    def get_fleet_learning_state(self) -> Dict[str, Any]:
        """Get learning state for the entire fleet."""
        agents = list(self.state.keys())

        return {
            "total_agents": len(agents),
            "fleet_goodness_avg": sum(self.state.values()) / len(self.state) if self.state else 0.5,
            "high_reliability_agents": len([a for a, g in self.state.items() if g > 0.7]),
            "low_reliability_agents": len([a for a, g in self.state.items() if g < 0.3]),
            "by_agent": {agent: round(g, 3) for agent, g in self.state.items()}
        }

    def run_learning_cycle(self, agent: str, real_outcome: bool, experience: str) -> Dict[str, Any]:
        """
        Run a complete learning cycle.

        real_outcome: True if the experience was positive, False if negative
        """
        if real_outcome:
            return self.positive_pass(agent, experience)
        else:
            return self.negative_pass(agent, experience)