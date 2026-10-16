"""The ordered seven-value interface; no renderer or agent dependencies."""

OBSERVATION_FIELDS = (
    'bird_y', 'bird_velocity', 'pipe_distance', 'gap_center_y',
    'distance_to_ground', 'distance_to_ceiling', 'pipe_velocity',
)


def observe(env):
    """Target the first pipe whose trailing edge has not cleared the bird."""
    c = env.config
    pipe = next(p for p in env.pipes if not p.passed)
    return (
        env.bird.y,
        env.bird.velocity,
        pipe.x - (c.bird_x + c.bird_radius),
        pipe.gap_center,
        c.height - env.bird.y - c.bird_radius,
        env.bird.y - c.bird_radius,
        env.pipe_speed,
    )
