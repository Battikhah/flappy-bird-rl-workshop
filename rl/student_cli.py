"""Student defaults around the shared CLI; no learning or inference policy here."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from game.config import Config
from .agent import QAgent
from .evaluation import evaluate


def remember_run(directory):
    """Track a successful local run without embedding machine paths in submissions."""
    latest = Path('runs/latest.json')
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps({'directory': str(directory)}, indent=2) + '\n')


def latest_directory():
    """Give beginners a useful next step when no training has finished yet."""
    latest = Path('runs/latest.json')
    if not latest.exists():
        raise ValueError('No trained run. Complete the exercises, then run python main.py train')
    return Path(json.loads(latest.read_text())['directory'])


def prepare(args):
    """Resolve no-argument workshop commands while preserving explicit model paths."""
    if args.mode == 'train':
        if args.research or args.resume:
            raise ValueError('Student runs must start fresh, cannot resume, and have a 2,400-episode cap')
        if args.output is None:
            args.output = Path('runs') / f'run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
        if args.seed is None:
            args.seed = 11
        return False
    if args.mode not in ('evaluate', 'watch', 'validate', 'submit', 'history'):
        return False
    if args.mode == 'evaluate' and args.model is None:
        directory = latest_directory()
        start = 3000000 if args.seed is None else args.seed
        seeds = range(start, start + args.seeds)
        if not set(seeds).issubset(range(3000000, 3000100)):
            raise ValueError('Checkpoint selection uses public validation seeds 3000000–3000099 only')
        from .checkpoints import export_best
        from .plotting import plot_run
        budget = QAgent.load(directory / 'model.npz')
        results = {}
        for path in sorted(directory.glob('model_*.npz')):
            metrics = evaluate(QAgent.load(path), seeds, Config(**budget.metadata['environment']))
            results[path.name] = metrics
            print(f'{path.stem}: validation mean={metrics["mean"]:.2f}')
        destination = args.output or directory / 'validation.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2) + '\n')
        if destination != directory / 'validation.json':
            (directory / 'validation.json').write_text(destination.read_text())
        selected = export_best(directory, results, budget)
        plot_run(directory)
        print(f'Selected {selected}; saved {directory / "best.npz"}')
        return True
    if args.model is None:
        directory = latest_directory()
        args.model = directory / ('best.npz' if (directory / 'best.npz').exists() else 'model.npz')
    if args.mode == 'evaluate' and args.seed is None:
        args.seed = 3000000
    return False
