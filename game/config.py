"""One workshop geometry with configurable episode budgets and scroll progression."""

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar


@dataclass(frozen=True)
class Config:
    """Share difficulty rules so generation, collisions, and drawing agree."""

    max_steps: int = 10000
    max_pipes: int = 100
    initial_speed: float = 3.0
    speed_increment: float = 0.5
    speed_every: int = 10
    max_speed: float = 6.0

    width: ClassVar[int] = 480
    height: ClassVar[int] = 600
    bird_x: ClassVar[int] = 120
    bird_radius: ClassVar[int] = 12
    pipe_width: ClassVar[int] = 64
    gap_height: ClassVar[int] = 160
    min_gap_height: ClassVar[int] = 130
    gap_every: ClassVar[int] = 3
    spacing_min: ClassVar[int] = 230
    spacing_max: ClassVar[int] = 330
    fps: ClassVar[int] = 60

    def __post_init__(self):
        """Reject configurations that invalidate episode or collision guarantees."""
        for name in ('max_steps', 'max_pipes', 'speed_every'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('initial_speed', 'speed_increment', 'max_speed'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if not 0 < self.initial_speed <= self.max_speed <= self.pipe_width:
            raise ValueError('Require 0 < initial_speed <= max_speed <= pipe_width')
        if self.speed_increment < 0:
            raise ValueError('speed_increment must be nonnegative')
