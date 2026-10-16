"""Offline learning curves from inspectable CSV and validation JSON artifacts."""

import csv
import json
from pathlib import Path

import numpy as np


def plot_run(directory):
    """Plot exploratory training scores separately from frozen greedy validation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with (directory / 'training.csv').open() as stream:
        records = list(csv.DictReader(stream))
    episodes = np.array([int(row['episode']) for row in records])
    scores = np.array([float(row['score']) for row in records])
    epsilon = np.array([float(row['epsilon']) for row in records])
    window = min(100, len(scores))
    smoothed = np.convolve(scores, np.ones(window) / window, mode='valid')
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout='constrained')
    axes[0].plot(episodes[window - 1:], smoothed, label=f'Training mean ({window} episodes)')
    validation = directory / 'validation.json'
    if validation.exists():
        data = json.loads(validation.read_text())
        points = sorted((int(name.split('_')[1].split('.')[0]), row['mean'])
                        for name, row in data.items() if name.startswith('model_'))
        if points:
            axes[0].plot(*zip(*points), marker='o', label='Greedy validation mean')
    axes[0].set(ylabel='Official pipes passed', title=f'Tabular Q-learning: {directory.name}')
    axes[0].legend()
    axes[0].grid(alpha=.2)
    axes[1].plot(episodes, epsilon, color='tab:orange')
    axes[1].set(xlabel='Completed training episodes', ylabel='Exploration epsilon')
    axes[1].grid(alpha=.2)
    fig.savefig(directory / 'learning_curve.png', dpi=160)
    plt.close(fig)
