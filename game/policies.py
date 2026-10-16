"""Small non-learning baselines; policies never inspect pipes or environment state."""


def heuristic(observation):
    """Brake a fall early and flap toward the currently observable gap center."""
    y, velocity, _, target, _, _, _ = observation
    return int(velocity >= -1.0 and y + 6 * velocity > target)


def periodic(step, period=19):
    """Use an unchanged open-loop flap cycle, without observing the map."""
    return int(step % period == 0)


def never_flap(observation):
    """Provide the zero-input baseline without any environment knowledge."""
    return 0


def random_action(observation, rng, flap_probability=0.5):
    """Sample independent actions using an RNG separate from map generation."""
    return int(rng.random() < flap_probability)
