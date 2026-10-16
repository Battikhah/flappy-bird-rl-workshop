"""Frozen-model evaluation on seed cohorts that never supply learning updates."""

from dataclasses import replace
from statistics import mean, median, pstdev

from game.config import Config
from game.environment import FlappyBirdEnv


def summarize(records):
    """Keep official pipe counts separate from training rewards and timeouts."""
    scores = [record['score'] for record in records]
    return {'episodes': len(scores), 'mean': mean(scores), 'median': median(scores),
            'max': max(scores), 'std': pstdev(scores),
            'reach_percent': {str(n): 100 * sum(s >= n for s in scores) / len(scores)
                              for n in (10, 25, 50, 100)},
            'timeouts': sum(row['reason'] == 'max_steps' for row in records),
            'records': records}


def evaluate(agent, seeds, config=None):
    """Expose only the observation to inference; collect scores from public rewards."""
    env = FlappyBirdEnv(config or Config())
    records = []
    for seed in seeds:
        observation = env.reset(seed)
        score = steps = 0
        while True:
            observation, reward, terminated, truncated, info = env.step(agent.act(observation))
            score += max(0, reward)
            steps += 1
            if terminated or truncated:
                break
        records.append({'seed': seed, 'score': int(score), 'steps': steps, 'reason': info['reason']})
    return summarize(records)


def evaluate_cohorts(agent, seeds, config, speed_levels=False):
    """Fixed-speed cohorts isolate exposure rather than hiding early-stage failures."""
    cohorts = {'progression': evaluate(agent, seeds, config)}
    if speed_levels:
        for speed in (3, 3.5, 4, 4.5, 5, 5.5, 6):
            cohorts[str(speed)] = evaluate(agent, seeds, replace(config, initial_speed=speed, max_speed=speed))
    return cohorts
