"""Seeded varied gaps, validated by a causal flight through the actual physics."""

from dataclasses import dataclass

from .physics import Bird, collides
from .policies import heuristic


@dataclass
class Pipe:
    """Track a pipe pair and whether its trailing edge has earned a point."""

    x: float
    gap_center: float
    passed: bool = False
    world_x: float | None = None
    gap_height: float | None = None


class PipeGenerator:
    """Carry a private reference flight, independent of the player's actions.

    Each accepted transition starts exactly when the previous pipe is passed.
    Its controller sees only the current gap, so no future-map lookahead is needed.
    The frontier stores velocity as well as height; endpoint distance alone cannot
    establish reachability under gravity and repeated flap impulses.
    """

    def __init__(self, rng, config):
        """Start the reference flight at the same state as a reset environment."""
        self.rng = rng
        self.config = config
        self.bird = Bird(config.height / 2)
        self.distance = 0.0
        self.last_world_x = None
        self.center = config.height / 2
        self.count = 0

    def flight(self, world_x, center):
        """Return the exit state if a causal, collision-free trajectory exists."""
        c = self.config
        bird = Bird(self.bird.y, self.bird.velocity)
        distance = self.distance
        # Match the emitted pipe so reachability uses its actual opening.
        gap_height = max(c.min_gap_height, c.gap_height - self.count // c.gap_every)
        pipe = Pipe(world_x - distance, center, gap_height=gap_height)
        speed = min(c.max_speed, c.initial_speed + c.speed_increment * (self.count // c.speed_every))
        while pipe.x + c.pipe_width >= c.bird_x - c.bird_radius:
            obs = (bird.y, bird.velocity, pipe.x - c.bird_x - c.bird_radius,
                   center, c.height - bird.y - c.bird_radius,
                   bird.y - c.bird_radius, speed)
            old_y = bird.y
            bird.advance(heuristic(obs))
            distance += speed
            pipe.x = world_x - distance
            if collides(bird, (pipe,), c, old_y, speed):
                return None
        return bird, distance

    def next_pipe(self, previous_x=None):
        """Rejection-sample varied centers, retaining a verified route per pipe."""
        c = self.config
        spacing = self.rng.randint(c.spacing_min, c.spacing_max)
        x = c.width if previous_x is None else previous_x + spacing
        world_x = c.width if self.last_world_x is None else self.last_world_x + spacing
        # Global sampling permits ascents, descents, reversals, and disjoint gaps.
        for _ in range(32):
            center = self.rng.uniform(120, c.height - 120)
            flight = self.flight(world_x, center)
            if flight is not None:
                break
        else:
            # A level continuation preserves the reference flight if random
            # candidates all fail; never emit an unvalidated transition.
            center = self.center
            flight = self.flight(world_x, center)
            if flight is None:
                # Extreme custom speeds may need extra settling time after exit.
                extra = c.max_speed * 80
                x += extra
                world_x += extra
                flight = self.flight(world_x, center)
                if flight is None:
                    raise RuntimeError('No feasible continuation for the reference flight')
        self.bird, self.distance = flight
        self.last_world_x = world_x
        self.center = center
        gap_height = max(c.min_gap_height, c.gap_height - self.count // c.gap_every)
        self.count += 1
        return Pipe(float(x), center, world_x=world_x, gap_height=gap_height)


def fill_pipes(pipes, generator, config):
    """Maintain a short lookahead without consulting the player's bird state."""
    if not pipes:
        pipes.append(generator.next_pipe())
    while pipes[-1].x < config.width + config.spacing_max:
        pipes.append(generator.next_pipe(pipes[-1].x))
