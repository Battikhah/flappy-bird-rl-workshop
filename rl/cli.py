"""Thin CLI adapters; inference and learning implementations stay separate."""

from dataclasses import asdict
import json
from pathlib import Path

from game.config import Config
from game.environment import FlappyBirdEnv
from .agent import Discretizer, QAgent
from .evaluation import evaluate_cohorts, summarize


def run(args):
    """Dispatch the Phase 2 commands without changing legacy play/demo/simulate."""
    student = (Path(__file__).resolve().parents[1] / 'student_exercises.py').is_file()
    if student:
        from .student_cli import prepare
        if prepare(args):
            return
    if args.mode in ('validate', 'submit'):
        from .submission import validate_model, submit_model
        if args.model is None:
            raise ValueError('--model is required outside the student project')
        result = (validate_model(args.model) if args.mode == 'validate' else
                  submit_model(args.model, args.output or Path('submissions') / f'{args.model.parent.name}.zip'))
        print(json.dumps(result, indent=2))
        return
    if args.mode == 'train':
        from .training import EpisodePlan, LearningConfig, RewardConfig, train
        if args.model:
            raise ValueError('Training starts from zero; --model is for inference/history only')
        default_config = Path(__file__).resolve().parents[1] / 'configs/workshop_student.json'
        config_path = args.config or (None if args.research else default_config)
        settings = json.loads(config_path.read_text()) if config_path else {}
        if args.resume and args.config:
            raise ValueError('Resume restores its saved settings; omit --config')
        train(args.output or Path('runs/qlearning'), args.episodes, seed=args.seed or 0,
              discretizer=Discretizer(**settings.get('discretization', {})),
              config=LearningConfig(**settings.get('learning', {})),
              rewards=RewardConfig(**settings.get('rewards', {})),
              plan=EpisodePlan(**settings.get('training', {})),
              env_config=Config(max_steps=args.max_steps, max_pipes=args.max_pipes),
              resume=args.resume, checkpoint_every=args.checkpoint_every, official=not args.research)
        from .plotting import plot_run
        plot_run(args.output or Path('runs/qlearning'))
        if student:
            from .student_cli import remember_run
            remember_run(args.output)
        return
    if args.model is None:
        raise ValueError('--model is required for evaluate/watch/history')
    if args.mode == 'history':
        from .training import review_history
        print(json.dumps(review_history(args.model), indent=2))
        return
    if args.model.suffix == '.zip':
        from .submission import load_submission
        agent = load_submission(args.model)
    else:
        agent = QAgent.load(args.model)
    config = Config(max_steps=args.max_steps, max_pipes=args.max_pipes)
    if args.mode == 'watch':
        watch(agent, args, config)
        return
    start = 1000000 if args.seed is None else args.seed
    seeds = range(start, start + args.seeds)
    training_range = agent.metadata.get('training_seed_range')
    if training_range and not (seeds[-1] < training_range[0] or seeds[0] > training_range[1]):
        raise ValueError('Evaluation seeds overlap the saved training seed range')
    results = {'model': str(args.model), 'episodes_trained': agent.metadata.get('episodes', 0),
               'seed_start': start, 'seed_count': args.seeds, 'environment': asdict(config),
               'cohorts': evaluate_cohorts(agent, seeds, config, args.speed_levels)}
    if args.baselines:
        # These are comparison runs only; neither trainer nor agent imports policies.
        from benchmark import evaluate as baseline_evaluate, POLICIES
        results['baselines'] = {name: summarize(baseline_evaluate(name, seeds, config)['records'])
                                for name in POLICIES}
    destination = args.output or Path('evaluation.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + '\n')
    for name, metrics in results['cohorts'].items():
        print(f'{name}: mean={metrics["mean"]:.2f} median={metrics["median"]:g} '
              f'max={metrics["max"]} std={metrics["std"]:.2f} reach%={metrics["reach_percent"]}')
    print(f'Full evidence: {destination}')


def watch(agent, args, config):
    """Render frozen Q-table actions with episode/seed information in the caption."""
    import pygame

    env = FlappyBirdEnv(config)
    clock = pygame.time.Clock()
    playback_speed = args.playback_speed
    try:
        for episode in range(args.episodes):
            seed = None if args.seed is None else args.seed + episode
            observation = env.reset(seed)
            env.render()
            env._renderer.mode = 'AI'
            env._renderer.playback_speed = playback_speed
            env.render()
            env._renderer.detail = f'EPISODE {episode + 1}/{args.episodes} / Q-TABLE'
            pygame.display.set_caption(f'Q-table agent | Episode {episode + 1}/{args.episodes} | Seed {seed}')
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        return
                    if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                            and env._renderer.speed_button_clicked(event.pos)):
                        playback_speed = next((rate for rate in (1, 2, 4, 8)
                                               if rate > playback_speed), 1)
                        env._renderer.playback_speed = playback_speed
                observation, _, terminated, truncated, _ = env.step(agent.act(observation))
                env.render()
                clock.tick(config.fps * playback_speed)
                if terminated or truncated:
                    break
    finally:
        env.close()
