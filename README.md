# PLATO Forward-Forward Learning

Predictive coding without backpropagation — based on Geoffrey Hinton's Forward-Forward algorithm.

## The Core Idea

Traditional neural networks learn via **backpropagation**: compute gradients, update weights backward. This requires:
- A differentiable loss function
- Synchronized forward-backward passes
- Access to the full network topology

**Forward-Forward (FF)** replaces this with two competing passes:

| Pass | Signal | Effect |
|------|--------|--------|
| **Positive** | Real experience | Increases "goodness" — the tile is accepted |
| **Negative** | Imagined/hypothetical | Decreases goodness — the tile is rejected |

The **gradient between positive and negative** replaces the gradient from backprop. No more backward pass needed.

## How It Works in PLATO

PLATO stores knowledge as **tiles** — discrete units with a question/answer structure. FF learning hooks into this:

1. **Positive tiles**: When an agent has a confirmed good outcome, the experience becomes a positive tile
2. **Negative tiles**: When an agent imagines or predicts a failure, that becomes a negative tile
3. **Goodness score**: Each agent accumulates a goodness score (0.0–1.0)
   - Positive passes boost goodness
   - Negative passes penalize goodness
   - Goodness decays over time (0.95x per event)
4. **Tile reinforcement**: When goodness exceeds threshold (0.7), associated tiles get reinforced in PLATO

## Quick Start

```python
from plato_fflearning import ForwardForwardLearner

ff = ForwardForwardLearner(plato_url="http://localhost:8847")

# Agent has a real positive experience
result = ff.positive_pass(
    agent="oracle1",
    experience="Casey asked for fleet status, I provided it accurately",
    associated_tiles=["tile_123", "tile_456"]
)

# Agent imagines a failure scenario
ff.negative_pass(
    agent="oracle1",
    experience="What if I had given wrong status? That would be bad."
)

# Query state
state = ff.get_learning_state("oracle1")
print(state["goodness"], state["level"])

# Fleet-wide view
fleet = ff.get_fleet_learning_state()
print(f"Fleet avg goodness: {fleet['fleet_goodness_avg']}")
```

## API Reference

### `ForwardForwardLearner(plato_url="http://localhost:8847")`

Creates a new FF learner instance.

### `positive_pass(agent, experience, domain, associated_tiles)`

Records a positive experience.

- `agent` (str): Agent identifier
- `experience` (str): Description of what happened
- `domain` (str, optional): Knowledge domain (default: "fleet_orchestration")
- `associated_tiles` (list, optional): Tile IDs to reinforce when threshold is exceeded

Returns: `{"agent", "pass", "goodness", "threshold_exceeded", "tiles_reinforced"}`

### `negative_pass(agent, experience, domain)`

Records a negative/hypothetical experience.

- `agent` (str): Agent identifier
- `experience` (str): Description of what was imagined or failed
- `domain` (str, optional): Knowledge domain

Returns: `{"agent", "pass", "goodness", "suppressed"}`

### `get_learning_state(agent)`

Returns current learning state for an agent.

Returns: `{"agent", "goodness", "level", "positive_tiles", "negative_tiles", "recommendation"}`

### `get_fleet_learning_state()`

Returns fleet-wide learning state.

Returns: `{"total_agents", "fleet_goodness_avg", "high_reliability_agents", "low_reliability_agents", "by_agent"}`

### `run_learning_cycle(agent, real_outcome, experience)`

Convenience method for a complete cycle.

- `real_outcome` (bool): True = positive pass, False = negative pass

## Goodness Levels

| Range | Level | Meaning |
|-------|-------|---------|
| 0.0–0.2 | critical | Agent needs positive experiences urgently |
| 0.2–0.4 | low | Agent is struggling, needs reinforcement |
| 0.4–0.6 | moderate | Normal learning range |
| 0.6–0.8 | high | Agent is reliable |
| 0.8–1.0 | exceptional | Top-tier performance |

## Comparison: Backprop vs Forward-Forward

| Aspect | Backprop | Forward-Forward |
|--------|----------|-----------------|
| Direction | Forward then backward | Two forward passes |
| Synchronization | All layers must wait | Layer-local |
| Supervision | Global loss signal | Local goodness signal |
| Requires gradients | Yes | No |
| PLATO integration | N/A | Native — tiles are the data |
| Hypothetical learning | No | Yes (negative pass) |

## Key Advantages

1. **No gradient communication**: Each layer computes independently
2. **Online learning**: No need to store activations for backprop
3. **Hypothetical reasoning**: Negative pass enables counterfactual thinking
4. **Biological plausibility**: More closely matches cortical circuitry
5. **PLATO-native**: Tiles are the natural representation for FF learning

## Architecture

```
Positive Experience → positive_pass() → ff_positive_tiles room (PLATO)
                                        ↓
                              Goodness increases
                                        ↓
                        Threshold exceeded → tiles reinforced

Negative Experience → negative_pass() → ff_negative_tiles room (PLATO)
                                       ↓
                             Goodness decreases
```

## Installation

```bash
pip install plato-fflearning
```

## License

MIT