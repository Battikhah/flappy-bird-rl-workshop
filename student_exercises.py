"""The only four functions students need to complete during live coding.

Use NumPy for the table and the supplied random.Random instance for exploration.
Only the seven public observations may influence the reward or encoded state.
"""
import numpy as np


def initialize_q_table(state_count):
    """Exercise 1: return a fresh float32 table with one row per state, two actions.

    Start with no knowledge. Never load, copy, or manually populate learned values.
    """
    return np.zeros((state_count, 2), dtype=np.float32)


class QLearner:
    """Connect the student exercises to the supplied episode loop."""

    def __init__(self, discretizer, config, q=None):
        """Every workshop run must start from an empty, independently owned table."""
        if q is not None:
            raise ValueError('Workshop runs cannot resume or warm-start')
        self.discretizer = discretizer
        self.config = config
        self.q = initialize_q_table(discretizer.state_count)
        if (not isinstance(self.q, np.ndarray) or self.q.dtype != np.float32
                or self.q.shape != (discretizer.state_count, 2) or np.any(self.q != 0)):
            raise ValueError('Exercise 1: expected a fresh zero float32 table shaped (states, 2)')

    def action(self, state, epsilon, rng):
        """Exercise 2: explore with probability epsilon; otherwise use this Q row.

        rng is a seeded random.Random instance. Explore over both legal actions.
        Return a built-in int. For reproducibility, ties should choose action 0.
        """
        if rng.random() < epsilon:
            return rng.randrange(2)
        values = self.q[state]
        return int(values[1] > values[0])

    def update(self, state, action, reward, next_state, terminated):
        """Exercise 3: update only the visited state/action using one-step Q-learning.

        self.config supplies alpha and gamma. A collision ends future return;
        artificial time/pipe truncations still bootstrap. The loop handles flags.
        """
        future = 0.0 if terminated else float(max(self.q[next_state]))
        current = float(self.q[state, action])
        target = reward + self.config.gamma * future
        self.q[state, action] = current + self.config.alpha * (target - current)


def training_reward(observation, official_reward, terminated, config):
    """Exercise 4: return one finite scalar, keeping official score untouched.

    config has survival, pipe, collision, and alignment weights. observation has
    exactly the seven approved values; y grows downward. official_reward is the
    public transition reward (positive when a pipe is passed). Give collisions
    their own penalty, and optionally shape using observed bird/gap height error.
    """
    if terminated:
        return config.collision
    error = abs(observation[0] - observation[3]) / 100
    return config.survival + config.pipe * max(0, official_reward) - config.alignment * error
