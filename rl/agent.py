"""Portable inference: NumPy, seven numeric observations, and a frozen Q-table.

This module has no dependency on the environment, trainer, or baseline policies.
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path

import numpy as np

OBSERVATION_FIELDS = (
    'bird_y', 'bird_velocity', 'pipe_distance', 'gap_center_y',
    'distance_to_ground', 'distance_to_ceiling', 'pipe_velocity',
)


@dataclass(frozen=True)
class Discretizer:
    """Bin relative gap height, velocity, and time to the current pipe.

    Clipped outer bins cover all finite values. Empty edge lists remove a feature
    without changing the public seven-observation contract. Exact edges use the
    upper bin. Flattened indices make the Bellman update a single table lookup.
    """

    height_edges: tuple = tuple(range(-240, 241, 12))
    velocity_edges: tuple = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10)
    time_edges: tuple = (0, 12, 30, 60)

    def __post_init__(self):
        """Normalize JSON lists and reject ambiguous or nonfinite bin boundaries."""
        for name in ('height_edges', 'velocity_edges', 'time_edges'):
            edges = tuple(getattr(self, name))
            if not all(isfinite(v) for v in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
                raise ValueError(f'{name} must be finite and strictly increasing')
            object.__setattr__(self, name, edges)

    @property
    def state_count(self):
        """Expose the dense table budget before starting a laptop experiment."""
        return prod(len(edges) + 1 for edges in (self.height_edges, self.velocity_edges, self.time_edges))

    def encode(self, observation):
        """Use only approved observations; reject malformed agent inputs early."""
        try:
            if len(observation) != 7 or not all(isfinite(x) for x in observation):
                raise ValueError('Expected exactly seven finite numerical observations')
        except TypeError as error:
            raise ValueError('Expected exactly seven finite numerical observations') from error
        y, velocity, distance, gap, ground, ceiling, speed = observation
        if speed <= 0:
            raise ValueError('pipe_velocity must be positive')
        height_bin = bisect_right(self.height_edges, y - gap)
        velocity_bin = bisect_right(self.velocity_edges, velocity)
        time_bin = bisect_right(self.time_edges, distance / speed)
        return ((height_bin * (len(self.velocity_edges) + 1) + velocity_bin)
                * (len(self.time_edges) + 1) + time_bin)


class QAgent:
    """Choose greedily from a read-only table; there is no learning path here."""

    def __init__(self, discretizer, q, metadata=None):
        """Validate exported dimensions and freeze an independent table copy."""
        table = np.array(q, dtype=np.float32, copy=True)
        if table.shape != (discretizer.state_count, 2) or not np.isfinite(table).all():
            raise ValueError('Q-table must be finite with shape (state_count, 2)')
        table.flags.writeable = False
        self.discretizer = discretizer
        self.q = table
        self.metadata = dict(metadata or {})

    def act(self, observation):
        """Return action 0 or 1 solely by Q value; deterministic ties choose zero."""
        values = self.q[self.discretizer.encode(observation)]
        return int(values[1] > values[0])

    def save(self, path, metadata=None):
        """Write NPZ + JSON; checksums detect interrupted or mismatched file pairs."""
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Model path must end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, q=self.q)
        document = dict(self.metadata)
        document.update(metadata or {})
        document.update({'format_version': 1, 'observation_fields': list(OBSERVATION_FIELDS),
                         'actions': [0, 1], 'discretization': asdict(self.discretizer),
                         'state_count': self.discretizer.state_count,
                         'table_bytes': self.q.nbytes, 'dtype': 'float32',
                         'sha256': sha256(temporary.read_bytes()).hexdigest()})
        json_path = path.with_suffix('.json')
        json_temp = json_path.with_suffix('.json.tmp')
        json_temp.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
        json_temp.replace(json_path)

    @classmethod
    def load(cls, path):
        """Load inference artifacts without pickle, environment, or training code."""
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        if (metadata.get('format_version') != 1
                or metadata.get('observation_fields') != list(OBSERVATION_FIELDS)
                or metadata.get('actions') != [0, 1]
                or metadata.get('sha256') != sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Model format, input/action contract, or checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            q = data['q']
        return cls(Discretizer(**metadata['discretization']), q, metadata)
