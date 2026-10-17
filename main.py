"""Workshop CLI: original simulation modes plus tabular training and deployment."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

from game.config import Config
from game.environment import FlappyBirdEnv
from game.policies import heuristic, periodic


def example_policy(step):
    """Retain the original periodic test helper for existing workshop callers."""
    return periodic(step)


def simulate(args):
    """Benchmark complete headless episodes, including resets and observations."""
    env = FlappyBirdEnv(Config(max_steps=args.max_steps, max_pipes=args.max_pipes))
    ticks = score = 0
    start = perf_counter()
    for episode in range(args.episodes):
        observation = env.reset(None if args.seed is None else args.seed + episode)
        while not env.done:
            observation = env.step(heuristic(observation))[0]
        ticks += env.steps
        score += env.score
    seconds = perf_counter() - start
    print(json.dumps({'episodes': args.episodes, 'steps': ticks, 'pipes': score, 'seconds': round(seconds, 4), 'steps_per_second': round(ticks / seconds), 'episodes_per_second': round(args.episodes / seconds, 2)}))


def playback(args):
    """Map SPACE presses to impulses; playback rate never changes physics dt."""
    import pygame

    env = FlappyBirdEnv(Config(max_steps=args.max_steps, max_pipes=args.max_pipes))
    env.reset(args.seed)
    clock = pygame.time.Clock()
    try:
        env.render()
        env._renderer.mode = 'AI' if args.mode == 'demo' else 'HUMAN'
        env._renderer.detail = 'OBSERVATION DEMO' if args.mode == 'demo' else ''
        ready = args.mode == 'play'
        while not env.done:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN:
                    if ready and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        ready = False
                        action = 1
                    elif not ready and event.key == pygame.K_SPACE:
                        action = 1
            if args.mode == 'demo':
                action = heuristic(env.observe())
            # Waiting on START must not consume physics ticks or episode budget.
            if not ready:
                env.step(action)
            env.render()
            clock.tick(env.config.fps * args.playback_speed)
        print(f'Score: {env.score}; steps: {env.steps}')
    finally:
        env.close()


def positive_float(value):
    """Reject invalid playback rates at the CLI boundary."""
    from math import isfinite
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return number


def positive_int(value):
    """Validate episode counts and limits before entering simulation loops."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main():
    """Expose simulation and RL modes; headless paths never initialize graphics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('play', 'simulate', 'demo', 'train', 'evaluate', 'watch', 'history', 'validate', 'submit'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--episodes', type=positive_int, default=None)
    parser.add_argument('--max-steps', type=positive_int, default=10000)
    parser.add_argument('--max-pipes', type=positive_int, default=100)
    parser.add_argument('--playback-speed', type=positive_float, default=1.0)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--research', action='store_true', help='Allow larger/resumed organizer runs; not official')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint-every', type=positive_int, default=500)
    parser.add_argument('--seeds', type=positive_int, default=100)
    parser.add_argument('--speed-levels', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = (5000 if args.research else 3000) if args.mode == 'train' else 1
    if args.mode in ('train', 'evaluate', 'watch', 'history', 'validate', 'submit'):
        from rl.cli import run
        run(args)
        return
    if args.mode == 'simulate':
        simulate(args)
    else:
        playback(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError, NotImplementedError, BadZipFile, EOFError, OverflowError) as error:
        raise SystemExit(f'Error: {error}. Check your inputs or see docs/TROUBLESHOOTING.md') from None
