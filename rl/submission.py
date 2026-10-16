"""Validate and package model data without importing submitted student code.

The supported encoder is the trusted Discretizer with declarative bin edges.
Checks establish artifact consistency, not the honesty of a training algorithm.
"""
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from game.config import Config
from .agent import Discretizer, QAgent

MAX_FILE_BYTES = 16 * 1024 * 1024
MEMBERS = {'model.npz', 'model.json', 'training.csv'}


def integer(value, name, low, high):
    """Reject bools and fractional counters before comparing episode budgets."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return value


def finite(value, name):
    """Keep nonfinite values out of metadata as well as the Q-table."""
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f'{name} must be finite and numeric')
    return value


def validate_model(path):
    """Accept a local NPZ/JSON/history trio or a bounded, data-only submission ZIP."""
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temporary:
            entries = archive.infolist()
            if len(entries) != 3 or {entry.filename for entry in entries} != MEMBERS:
                raise ValueError('Submission ZIP must contain only model.npz, model.json, training.csv')
            if any(entry.file_size > MAX_FILE_BYTES for entry in entries):
                raise ValueError('Submission file exceeds the 16 MiB limit')
            for entry in entries:
                (Path(temporary) / entry.filename).write_bytes(archive.read(entry))
            return validate_model(Path(temporary) / 'model.npz')
    if path.suffix != '.npz':
        raise ValueError('Choose a model .npz or submission .zip')
    for file in (path, path.with_suffix('.json')):
        if file.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Model file exceeds the 16 MiB limit')
    metadata = json.loads(path.with_suffix('.json').read_text())
    required = {'format_version', 'observation_fields', 'actions', 'discretization',
                'state_count', 'table_bytes', 'dtype', 'sha256', 'episodes',
                'checkpoint_episode', 'run_episode_budget', 'run_kind', 'run_id',
                'initialization', 'official_episode_limit', 'training_seed',
                'training_seed_range', 'learning', 'rewards', 'environment',
                'episode_plan', 'training_seconds', 'history_file', 'history_sha256'}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        missing = sorted(required - set(metadata)) if isinstance(metadata, dict) else ['metadata object']
        raise ValueError(f'Missing required model metadata: {missing}')
    if (metadata['actions'] != [0, 1]
            or any(type(action) is not int for action in metadata['actions'])):
        raise ValueError('Model actions must be exactly the integers [0, 1]')
    integer(metadata['format_version'], 'format_version', 1, 1)
    integer(metadata['state_count'], 'state_count', 1, 1000000)
    integer(metadata['table_bytes'], 'table_bytes', 8, MAX_FILE_BYTES)
    encoder = metadata['discretization']
    if not isinstance(encoder, dict) or set(encoder) != {'height_edges', 'velocity_edges', 'time_edges'}:
        raise ValueError('Unsupported state encoder; use declarative height/velocity/time bin edges')
    discretizer = Discretizer(**encoder)
    integer(discretizer.state_count, 'state_count', 1, 1000000)
    if (metadata.get('state_count') != discretizer.state_count
            or metadata.get('dtype') != 'float32'
            or metadata.get('table_bytes') != discretizer.state_count * 2 * 4):
        raise ValueError('Table size/dtype metadata does not match the supported encoder')
    # Bound the nested NPZ before NumPy allocates memory from an untrusted header.
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['q.npy'] or archive.getinfo('q.npy').file_size > MAX_FILE_BYTES:
            raise ValueError('NPZ must contain only a bounded q array')
        with archive.open('q.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise ValueError('Unsupported NumPy array format; re-export the model')
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            if shape != (discretizer.state_count, 2) or dtype != np.dtype('float32'):
                raise ValueError('Q array must have two actions and float32 values matching the encoder')
    agent = QAgent.load(path)  # Validates seven observations, actions, checksum, and finite values.
    if metadata.get('run_kind') != 'official' or metadata.get('initialization') != 'zeros':
        raise ValueError('Official submissions require a fresh zero-table run; research/legacy models are ineligible')
    episodes = integer(metadata['episodes'], 'episodes', 1, 3000)
    checkpoint = integer(metadata['checkpoint_episode'], 'checkpoint_episode', 1, 3000)
    budget = integer(metadata.get('training_budget_episodes', episodes), 'training_budget_episodes', 1, 3000)
    requested = integer(metadata['run_episode_budget'], 'run_episode_budget', 1, 3000)
    if checkpoint != episodes or not checkpoint <= budget == requested:
        raise ValueError('Incomplete run accounting: run evaluate to select a checkpoint with the full history')
    if metadata.get('official_episode_limit') != 3000 or not isinstance(metadata.get('run_id'), str) or not metadata['run_id']:
        raise ValueError('Official limit or run identity is missing')
    seed = metadata['training_seed']
    if type(seed) is not int:
        raise ValueError('training_seed must be an integer')
    learning = metadata['learning']
    if set(learning) != {'alpha', 'gamma', 'epsilon_start', 'epsilon_min', 'epsilon_decay'}:
        raise ValueError('Missing or unsupported learning configuration')
    for name, value in learning.items():
        finite(value, name)
    if not (0 < learning['alpha'] <= 1 and 0 <= learning['gamma'] < 1
            and 0 <= learning['epsilon_min'] <= learning['epsilon_start'] <= 1
            and 0 < learning['epsilon_decay'] <= 1):
        raise ValueError('Invalid learning configuration bounds')
    rewards = metadata['rewards']
    if set(rewards) != {'survival', 'pipe', 'collision', 'alignment'}:
        raise ValueError('Missing reward configuration')
    for name, value in rewards.items():
        finite(value, name)
    if rewards['alignment'] < 0 or finite(metadata['training_seconds'], 'training_seconds') < 0:
        raise ValueError('Alignment and training time must be nonnegative')
    environment = Config(**metadata['environment'])
    plan = metadata['episode_plan']
    if set(plan) != {'map_seed', 'seed_pool', 'slow_every'}:
        raise ValueError('Unsupported training seed plan')
    base = seed if plan['map_seed'] is None else plan['map_seed']
    if type(base) is not int:
        raise ValueError('map_seed must be an integer')
    pool = plan['seed_pool']
    if pool is not None:
        integer(pool, 'seed_pool', 1, 2**63 - 1)
    slow = integer(plan['slow_every'], 'slow_every', 0, 2**63 - 1)
    if metadata['training_seed_range'] != [base, base + (min(budget, pool) if pool else budget) - 1]:
        raise ValueError('Training history seed range is inconsistent')
    validation = metadata.get('selection_validation_seeds')
    if validation is not None and (not validation or any(type(s) is not int or s not in range(3000000, 3000100) for s in validation)
                                   or len(set(validation)) != len(validation)):
        raise ValueError('Checkpoint selection must use public validation seeds only')
    history_name = metadata['history_file']
    if not isinstance(history_name, str) or Path(history_name).name != history_name or '\\' in history_name:
        raise ValueError('History must be a file beside the model')
    history = path.parent / history_name
    if history.stat().st_size > MAX_FILE_BYTES or sha256(history.read_bytes()).hexdigest() != metadata['history_sha256']:
        raise ValueError('Training history size or checksum mismatch')
    with history.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['episode', 'seed', 'training_mode', 'epsilon', 'score', 'steps', 'return']:
            raise ValueError('Unsupported training history columns')
        rows = list(reader)
    if len(rows) != budget:
        raise ValueError('Training history row count does not match the full episode budget')
    for index, row in enumerate(rows):
        expected_seed = base + (index % pool if pool else index)
        if (int(row['episode']) != index + 1 or int(row['seed']) != expected_seed
                or expected_seed in range(3000000, 3000100)
                or row['training_mode'] != ('slow' if slow and index % slow == 0 else 'progression')):
            raise ValueError('Training history episode/seed/mode is inconsistent or overlaps validation')
        integer(int(row['steps']), 'history steps', 1, environment.max_steps)
        integer(int(row['score']), 'history score', 0, environment.max_pipes)
        if not 0 <= finite(float(row['epsilon']), 'history epsilon') <= 1:
            raise ValueError('Training history epsilon is outside [0, 1]')
        finite(float(row['return']), 'history return')
    probes = [(y, v, distance, 300, 588-y, y-12, speed)
              for y in (12, 300, 588) for v in (-4, 0, 10)
              for distance in (-50, 100, 400) for speed in (3, 6)]
    # Roundtrip uses only the trusted inference module, never submitted Python.
    with tempfile.TemporaryDirectory() as temporary:
        roundtrip = Path(temporary) / 'model.npz'
        agent.save(roundtrip)
        restored = QAgent.load(roundtrip)
        before = agent.q.copy()
        for observation in probes:
            action = agent.act(observation)
            if type(action) is not int or action not in (0, 1) or restored.act(observation) != action:
                raise ValueError('Invalid or irreproducible inference output')
        if not np.array_equal(agent.q, before) or not np.array_equal(agent.q, restored.q):
            raise ValueError('Serialization or inference changed the Q-table')
    return {'valid': True, 'actions': [0, 1], 'observations': 7,
            'states': discretizer.state_count, 'training_episodes': budget,
            'checkpoint_episode': checkpoint, 'training_seed': seed,
            'source_review_required': ['Q-learning only', 'no warm-start or manual Q values',
                                       'no hidden observations or action overrides']}


def submit_model(model_path, output):
    """Export only three data files, preserving the full selected run's history."""
    model_path, output = Path(model_path), Path(output)
    if model_path.suffix != '.npz' or output.suffix != '.zip':
        raise ValueError('submit needs a model .npz and an output .zip')
    summary = validate_model(model_path)
    metadata = json.loads(model_path.with_suffix('.json').read_text())
    history = model_path.parent / metadata['history_file']
    metadata['history_file'] = 'training.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, 'model.npz')
        archive.writestr('model.json', json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        archive.write(history, 'training.csv')
    return {**summary, 'submission': str(output)}


def load_submission(path):
    """Return a frozen agent from validated data, without importing training code."""
    path = Path(path)
    validate_model(path)
    if path.suffix == '.npz':
        return QAgent.load(path)
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temporary:
        for name in ('model.npz', 'model.json'):
            (Path(temporary) / name).write_bytes(archive.read(name))
        return QAgent.load(Path(temporary) / 'model.npz')
