"""Readable one-step tabular Q-learning; no baseline controller imports."""

import csv
from hashlib import sha256
from uuid import uuid4
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from game.config import Config
from game.environment import FlappyBirdEnv
from .agent import Discretizer, QAgent
from student_exercises import QLearner, training_reward


OFFICIAL_EPISODE_LIMIT = 3000
PUBLIC_VALIDATION_SEEDS = range(3000000, 3000100)
HISTORY_FIELDS = ('episode', 'seed', 'training_mode', 'epsilon', 'score', 'steps', 'return')


@dataclass(frozen=True)
class LearningConfig:
    """Keep epsilon decay episode-based so checkpoint resumes reproduce exactly."""

    alpha: float = .1
    gamma: float = .99
    epsilon_start: float = 1.0
    epsilon_min: float = .01
    epsilon_decay: float = .998

    def __post_init__(self):
        """Reject parameters that violate the standard update/exploration bounds."""
        values = asdict(self).values()
        if not all(isfinite(v) for v in values):
            raise ValueError('Learning parameters must be finite')
        if not (0 < self.alpha <= 1 and 0 <= self.gamma < 1
                and 0 <= self.epsilon_min <= self.epsilon_start <= 1
                and 0 < self.epsilon_decay <= 1):
            raise ValueError('Invalid alpha, gamma, or epsilon schedule')

    def epsilon(self, episode):
        """Decay from the original schedule, never restart exploration on resume."""
        return max(self.epsilon_min, self.epsilon_start * self.epsilon_decay ** episode)


@dataclass(frozen=True)
class RewardConfig:
    """Training-only reward weights; official pipe scores remain untouched."""

    survival: float = .1
    pipe: float = 5.0
    collision: float = -10.0
    alignment: float = .1

    def __post_init__(self):
        """Keep rewards finite and alignment a nonnegative proximity penalty."""
        if not all(isfinite(v) for v in asdict(self).values()) or self.alignment < 0:
            raise ValueError('Reward weights must be finite; alignment must be nonnegative')


@dataclass(frozen=True)
class EpisodePlan:
    """Keep map sampling separate from exploration and optionally add slow practice.

    Defaults preserve Phase 2 exactly. A bounded seed pool deliberately repeats
    maps for the overfitting experiment; it is not the recommended default.
    """

    map_seed: int | None = None
    seed_pool: int | None = None
    slow_every: int = 0

    def __post_init__(self):
        """Validate integer schedules so seed assignment is unambiguous."""
        if self.map_seed is not None and type(self.map_seed) is not int:
            raise ValueError('map_seed must be an integer or None')
        if self.seed_pool is not None and (type(self.seed_pool) is not int or self.seed_pool <= 0):
            raise ValueError('seed_pool must be a positive integer or None')
        if type(self.slow_every) is not int or self.slow_every < 0:
            raise ValueError('slow_every must be a nonnegative integer')

    def seed_for(self, rng_seed, episode):
        """Choose a map without reading or consuming the action RNG."""
        base = rng_seed if self.map_seed is None else self.map_seed
        return base + (episode if self.seed_pool is None else episode % self.seed_pool)

    def is_slow(self, episode):
        """Select fixed-speed-3 practice episodes, starting at episode zero."""
        return self.slow_every > 0 and episode % self.slow_every == 0






def train(output, episodes, seed=0, discretizer=None, config=None, rewards=None,
          env_config=None, resume=None, checkpoint_every=500, plan=None, official=False):
    """Train additional episodes and export self-contained resumable checkpoints.

    By default map seeds are seed + episode_index; EpisodePlan can separate them.
    Statistics are diagnostics only, never agent inputs or exploration overrides.
    Official runs always start at zero and cannot resume; the library defaults
    to research mode for compatibility. The student CLI defaults to official.
    """
    if type(episodes) is not int or episodes <= 0 or type(checkpoint_every) is not int or checkpoint_every <= 0:
        raise ValueError('episodes and checkpoint_every must be positive integers')
    if official and episodes > OFFICIAL_EPISODE_LIMIT:
        raise ValueError('Official runs allow at most 3,000 training episodes')
    if official and resume is not None:
        raise ValueError('Official runs require a fresh zero Q-table; resume is research only')
    run_id = uuid4().hex
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start_episode = 0
    elapsed_before = 0.0
    history = []
    rng = random.Random(seed)
    q = None
    if resume is not None:
        previous = QAgent.load(resume)
        meta = previous.metadata
        discretizer = previous.discretizer
        config = LearningConfig(**meta['learning'])
        rewards = RewardConfig(**meta['rewards'])
        plan = EpisodePlan(**meta.get('episode_plan', {}))
        env_config = Config(**meta['environment'])
        seed = meta['training_seed']
        start_episode = meta['episodes']
        existing = output / 'model.npz'
        if existing.exists() and QAgent.load(existing).metadata.get('episodes') != start_episode:
            raise ValueError('Resume into a fresh output directory to preserve a different checkpoint')
        elapsed_before = meta['training_seconds']
        state = meta['rng_state']
        rng.setstate((state[0], tuple(state[1]), state[2]))
        q = previous.q
        history_path = Path(resume).parent / 'training.csv'
        if history_path.exists():
            with history_path.open() as stream:
                history = [row for row in csv.DictReader(stream) if int(row['episode']) <= start_episode]
    elif (output / 'model.npz').exists() or (output / 'training.csv').exists():
        raise ValueError('Output already contains a run; use --resume or another output directory')
    discretizer = discretizer or Discretizer()
    config = config or LearningConfig()
    rewards = rewards or RewardConfig()
    env_config = env_config or Config()
    plan = plan or EpisodePlan()
    if official and any(plan.seed_for(seed, episode) in PUBLIC_VALIDATION_SEEDS
                        for episode in range(episodes)):
        raise ValueError('Official training seeds must exclude public validation seeds')
    learner = QLearner(discretizer, config, q)
    env = FlappyBirdEnv(env_config)
    slow_env = FlappyBirdEnv(replace(env_config, initial_speed=3, max_speed=3)) if plan.slow_every else None
    # Persist every completed episode, even if a run stops between checkpoints.
    history_path = output / 'training.csv'
    with history_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    started = perf_counter()
    for episode in range(start_episode, start_episode + episodes):
        active_env = slow_env if plan.is_slow(episode) else env
        map_seed = plan.seed_for(seed, episode)
        observation = active_env.reset(map_seed)
        state = discretizer.encode(observation)
        epsilon = config.epsilon(episode)
        total = score = steps = 0
        while True:
            action = learner.action(state, epsilon, rng)
            observation, official_reward, terminated, truncated, _ = active_env.step(action)
            next_state = discretizer.encode(observation)
            reward = training_reward(observation, official_reward, terminated, rewards)
            learner.update(state, action, reward, next_state, terminated)
            total += reward
            score += max(0, official_reward)
            steps += 1
            state = next_state
            if terminated or truncated:
                break
        history.append({'episode': episode + 1, 'seed': map_seed,
                        'training_mode': 'slow' if plan.is_slow(episode) else 'progression',
                        'epsilon': epsilon, 'score': int(score), 'steps': steps,
                        'return': total})
        with history_path.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(history[-1])
        completed = episode + 1
        if completed % checkpoint_every == 0 or completed == start_episode + episodes:
            # Immutable history snapshots make early checkpoints independently auditable.
            snapshot = output / f'history_{completed:06d}.csv'
            snapshot.write_bytes(history_path.read_bytes())
            metadata = {'episodes': completed, 'checkpoint_episode': completed,
                        'run_kind': 'official' if official else 'research',
                        'run_id': run_id, 'run_episode_budget': start_episode + episodes,
                        'initialization': 'zeros' if resume is None else 'resumed',
                        'official_episode_limit': OFFICIAL_EPISODE_LIMIT if official else None,
                        'history_file': snapshot.name,
                        'history_sha256': sha256(snapshot.read_bytes()).hexdigest(),
                        'training_seed': seed,
                        'training_seed_range': [plan.seed_for(seed, 0),
                                                plan.seed_for(seed, 0) + (completed if plan.seed_pool is None else min(completed, plan.seed_pool)) - 1],
                        'episode_plan': asdict(plan),
                        'rng_state': rng.getstate(), 'learning': asdict(config),
                        'rewards': asdict(rewards), 'environment': asdict(env_config),
                        'training_seconds': elapsed_before + perf_counter() - started,
                        'algorithm': 'one-step tabular Q-learning',
                        'truncation_bootstrap': True}
            model = QAgent(discretizer, learner.q, metadata)
            model.save(output / f'model_{completed:06d}.npz')
            model.save(output / 'model.npz')
            recent = history[-100:]
            average = sum(float(row['score']) for row in recent) / len(recent)
            print(f'episode={completed} epsilon={epsilon:.4f} mean_last100={average:.2f}', flush=True)
    return model


def review_history(model_path):
    """Check a model's recorded history and return a compact review summary.

    This detects accidental artifact mismatches, not dishonest edits to Python
    or metadata. Competition enforcement beyond this trainer is out of scope.
    """
    model_path = Path(model_path)
    metadata = QAgent.load(model_path).metadata
    path = model_path.parent / metadata.get('history_file', 'training.csv')
    if metadata.get('history_sha256') and sha256(path.read_bytes()).hexdigest() != metadata['history_sha256']:
        raise ValueError('Training history checksum mismatch')
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    count = metadata.get('training_budget_episodes', metadata['episodes'])
    if [int(row['episode']) for row in rows] != list(range(1, count + 1)):
        raise ValueError('Training history must log every counted episode exactly once')
    return {'run_kind': metadata.get('run_kind', 'legacy research'),
            'training_seed': metadata['training_seed'], 'logged_episodes': len(rows),
            'run_episode_budget': metadata.get('run_episode_budget'),
            'checkpoint_episode': metadata.get('checkpoint_episode', metadata['episodes']),
            'total_steps': sum(int(row['steps']) for row in rows),
            'history_file': str(path), 'checksum_recorded': 'history_sha256' in metadata}
