"""Fixed-tick physics; all distances are pixels and velocities are pixels/tick."""

from dataclasses import dataclass

GRAVITY = 0.45
FLAP_VELOCITY = -4.5


@dataclass
class Bird:
    """Store only vertical state because the world scrolls past a fixed bird x."""

    y: float
    velocity: float = 0.0

    def advance(self, action):
        """Apply impulse before gravity, using the same update in every mode."""
        if action == 1:
            self.velocity = FLAP_VELOCITY
        self.velocity += GRAVITY
        self.y += self.velocity


def collides(bird, pipes, config, old_y, speed):
    """Conservatively sweep rectangular hitboxes so fast motion cannot tunnel."""
    radius = config.bird_radius
    top = min(old_y, bird.y) - radius
    bottom = max(old_y, bird.y) + radius
    if top <= 0 or bottom >= config.height:
        return True
    for pipe in pipes:
        # Pipes already moved; add speed to recover the previous right edge.
        if (pipe.x <= config.bird_x + radius
                and pipe.x + config.pipe_width + speed >= config.bird_x - radius):
            if top <= pipe.gap_center - config.gap_height / 2 or bottom >= pipe.gap_center + config.gap_height / 2:
                return True
    return False
