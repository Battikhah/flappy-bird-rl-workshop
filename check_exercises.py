"""Run one small behavioral check per exercise without giving implementation code."""
import argparse
import math
import random
from types import SimpleNamespace

import numpy as np
from student_exercises import initialize_q_table, QLearner, training_reward


def check(number):
    """Use small numerical examples so failures can be worked through on paper."""
    if number == 1:
        q = initialize_q_table(3)
        assert q.shape == (3, 2) and q.dtype == np.float32 and np.all(q == 0), 'Exercise 1: shape, dtype, or initialization'
        other = initialize_q_table(3)
        q[0, 0] = 9
        assert other[0, 0] == 0, 'Exercise 1: tables must be independent'
        return
    # Isolate each exercise so students can check it before completing the others.
    learner = object.__new__(QLearner)
    learner.config = SimpleNamespace(alpha=.5, gamma=.5)
    learner.q = np.array([[2, 0], [4, 0]], dtype=np.float32)
    if number == 2:
        assert learner.action(0, 0, random.Random(1)) == 0, 'Exercise 2: greedy choice'
        rng = random.Random(42)
        actions = [learner.action(0, 1, rng) for _ in range(100)]
        assert set(actions) == {0, 1} and all(type(a) is int for a in actions), 'Exercise 2: uniform legal exploration'
    elif number == 3:
        learner.update(0, 0, 3, 1, False)
        assert learner.q[0, 0] == 3.5, 'Exercise 3: nonterminal example should give 3.5'
        learner.q[0, 0] = 2
        learner.update(0, 0, 3, 1, True)
        assert learner.q[0, 0] == 2.5, 'Exercise 3: terminal example should give 2.5'
        assert learner.q[0, 1] == 0 and learner.q[1, 0] == 4, 'Exercise 3: update only the visited entry'
    else:
        config = SimpleNamespace(survival=.2, pipe=10, collision=-20, alignment=.4)
        observation = (300, 0, 100, 300, 288, 288, 3)
        death = training_reward(observation, -1, True, config)
        alive = training_reward(observation, 0, False, config)
        passed = training_reward(observation, 1, False, config)
        assert all(math.isfinite(v) for v in (death, alive, passed)), 'Exercise 4: finite scalar rewards'
        assert death == config.collision and passed > alive, 'Exercise 4: collision penalty and pipe bonus'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exercise', type=int, choices=(1, 2, 3, 4), nargs='?')
    args = parser.parse_args()
    try:
        for number in (range(1, 5) if args.exercise is None else (args.exercise,)):
            check(number)
            print(f'Exercise {number}: passed')
    except (NotImplementedError, AssertionError, TypeError, ValueError) as error:
        parser.exit(2, f'{error}\n')
