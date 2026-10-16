"""Select a frozen checkpoint using public validation, retaining full accounting."""

from pathlib import Path
from hashlib import sha256
from .agent import QAgent
from .training import OFFICIAL_EPISODE_LIMIT, PUBLIC_VALIDATION_SEEDS, review_history


def export_best(directory, validation, budget_model):
    """Keep the best validation checkpoint, counting the full training budget."""
    directory = Path(directory)
    name = max(sorted(validation), key=lambda key: validation[key]['mean'])
    winner = QAgent.load(directory / name)
    extra = {}
    if (winner.metadata.get('run_kind') == 'official'
            or budget_model.metadata.get('run_kind') == 'official'):
        meta = budget_model.metadata
        if (meta.get('run_kind') != 'official'
                or meta['episodes'] > OFFICIAL_EPISODE_LIMIT
                or meta['run_episode_budget'] > OFFICIAL_EPISODE_LIMIT
                or meta['initialization'] != 'zeros'
                or winner.metadata.get('run_kind') != 'official'
                or winner.metadata.get('run_id') != meta['run_id']
                or winner.metadata['episodes'] > meta['episodes']):
            raise ValueError('Checkpoint and budget must belong to the same official run')
        seed_lists = [[row['seed'] for row in metrics.get('records', [])]
                      for metrics in validation.values()]
        if (not seed_lists[0] or any(seeds != seed_lists[0] for seeds in seed_lists)
                or len(set(seed_lists[0])) != len(seed_lists[0])
                or not set(seed_lists[0]).issubset(PUBLIC_VALIDATION_SEEDS)):
            raise ValueError('Official checkpoint selection requires the same public validation seeds')
        # The selected table retains the full run history, including later experience.
        history = directory / meta['history_file']
        if sha256(history.read_bytes()).hexdigest() != meta['history_sha256']:
            raise ValueError('Training history checksum mismatch')
        destination = directory / 'best.history.csv'
        destination.write_bytes(history.read_bytes())
        extra = {'history_file': destination.name, 'history_sha256': meta['history_sha256'],
                 'selection_validation_seeds': seed_lists[0]}
    winner.save(directory / 'best.npz', {
        **extra,
        'selected_checkpoint': name,
        'selection': 'maximum greedy validation mean; earlier checkpoint wins ties',
        'training_budget_episodes': budget_model.metadata['episodes'],
        'training_budget_seconds': budget_model.metadata['training_seconds'],
        'training_seed_range': budget_model.metadata['training_seed_range'],
    })
    if budget_model.metadata.get('run_kind') == 'official':
        review_history(directory / 'best.npz')
    return name
