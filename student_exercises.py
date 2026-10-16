"""The only four functions students need to complete during live coding.

Use NumPy for the table and the supplied random.Random instance for exploration.
Only the seven public observations may influence the reward or encoded state.
"""
import numpy as np


def initialize_q_table(state_count):
    """Exercise 1: return a fresh float32 table with one row per state, two actions.

    Start with no knowledge. Never load, copy, or manually populate learned values.
    """
    # TODO 1: choose the table shape, initial values, and NumPy dtype.
    raise NotImplementedError('Exercise 1: implement initialize_q_table in student_exercises.py')


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
        # TODO 2: use the supplied RNG, not a new generator or a game heuristic.
        raise NotImplementedError('Exercise 2: implement QLearner.action in student_exercises.py')

    def update(self, state, action, reward, next_state, terminated):
        """Exercise 3: update only the visited state/action using one-step Q-learning.

        self.config supplies alpha and gamma. A collision ends future return;
        artificial time/pipe truncations still bootstrap. The loop handles flags.
        """
        # TODO 3: form the target, handle terminal transitions, then update in place.
        raise NotImplementedError('Exercise 3: implement QLearner.update in student_exercises.py')


def training_reward(observation, official_reward, terminated, config):
    """Exercise 4: return one finite scalar, keeping official score untouched.

    config has survival, pipe, collision, and alignment weights. observation has
    exactly the seven approved values; y grows downward. official_reward is the
    public transition reward (positive when a pipe is passed). Give collisions
    their own penalty, and optionally shape using observed bird/gap height error.
    """
    # TODO 4: combine the configured terms using only these public inputs.
    raise NotImplementedError('Exercise 4: implement training_reward in student_exercises.py')
