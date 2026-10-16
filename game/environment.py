"""Deterministic simulation API for human play and future tabular agents."""

import random

from .config import Config
from .observations import observe
from .physics import Bird, collides
from .pipes import PipeGenerator, fill_pipes


class FlappyBirdEnv:
    """Own episode state; wall-clock time and Pygame never enter physics."""

    def __init__(self, config=None):
        """Create independent RNG and rendering resources for each environment."""
        self.config = config if config is not None else Config()
        self._renderer = None
        self._rng = random.Random()
        self.reset()

    @property
    def pipe_speed(self):
        """Expose positive leftward speed for the next physics tick."""
        c = self.config
        return min(c.max_speed, c.initial_speed + c.speed_increment * (self.score // c.speed_every))

    def reset(self, seed=None):
        """Start a fresh seeded stream; None requests a new nondeterministic seed."""
        self._rng.seed(seed)
        self.bird = Bird(self.config.height / 2)
        self.pipes = []
        self._generator = PipeGenerator(self._rng, self.config)
        self.score = self.steps = 0
        self._scroll_distance = 0.0
        self.done = False
        fill_pipes(self.pipes, self._generator, self.config)
        return self.observe()

    def observe(self):
        """Return the seven-value numeric tuple without consuming randomness."""
        return observe(self)

    def step(self, action):
        """Advance one tick; reward is +1/pass, -1/collision, otherwise zero."""
        if type(action) is not int or action not in (0, 1):
            raise ValueError('action must be integer 0 or 1 (booleans are invalid)')
        if self.done:
            raise RuntimeError('Episode finished; call reset() before step()')
        speed = self.pipe_speed
        old_y = self.bird.y
        self.bird.advance(action)
        previous_distance = self._scroll_distance
        self._scroll_distance += speed
        for pipe in self.pipes:
            if pipe.world_x is None:
                pipe.world_x = pipe.x + previous_distance
            # Shared world coordinates keep generator replay exact at fractional speeds.
            pipe.x = pipe.world_x - self._scroll_distance
        self.steps += 1
        terminated = collides(self.bird, self.pipes, self.config, old_y, speed)
        passed = 0
        if not terminated:
            for pipe in self.pipes:
                if not pipe.passed and pipe.x + self.config.pipe_width < self.config.bird_x - self.config.bird_radius:
                    pipe.passed = True
                    passed += 1
            self.score += passed
        self.pipes = [p for p in self.pipes if p.x + self.config.pipe_width >= 0]
        fill_pipes(self.pipes, self._generator, self.config)
        # Collision takes precedence if it coincides with a time limit.
        truncated = not terminated and (self.steps >= self.config.max_steps or self.score >= self.config.max_pipes)
        self.done = terminated or truncated
        reason = None
        if terminated:
            reason = 'collision'
        elif truncated:
            reason = 'max_pipes' if self.score >= self.config.max_pipes else 'max_steps'
        reward = -1.0 if terminated else float(passed)
        info = {'score': self.score, 'steps': self.steps, 'reason': reason}
        return self.observe(), reward, terminated, truncated, info

    def render(self):
        """Lazily open a window and return its Pygame Surface without stepping."""
        if self._renderer is None:
            from .renderer import Renderer
            self._renderer = Renderer(self.config)
        return self._renderer.draw(self)

    def close(self):
        """Release the optional window; headless environments need no cleanup."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
