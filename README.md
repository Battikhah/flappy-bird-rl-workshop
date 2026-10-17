# Flappy Bird RL Workshop

Build a tabular Q-learning agent by completing four short Python exercises. The game, observation encoder, training loop, evaluation, model export, and submission checks are supplied. Your learning algorithm is completed during the workshop.

## Before class

Install **Python 3.11**, **Git**, and **Visual Studio Code**. Clone the public workshop repository:

```bash
git clone https://github.com/Battikhah/flappy-bird-rl-workshop.git
cd flappy-bird-rl-workshop
code .
```

If `code` is not recognized, open VS Code and choose **File → Open Folder**, then select `flappy-bird-rl-workshop`. In VS Code, select the `.venv` Python interpreter after setup.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python --version
```

If PowerShell blocks activation, do not change system policy. Install and run through the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py play
```

### macOS and Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python --version
```

Check that Python reports `3.11.x`. Then verify the installation:

```bash
python main.py play
```

Press **SPACE** to flap and **ESC** to exit. The game runs locally in a Pygame window.

## Four live-coding exercises

Edit only `student_exercises.py` during the guided exercises. The four TODOs are:

1. Initialize a fresh, all-zero Q-table with the correct shape and type.
2. Select an action using epsilon-greedy Q-table lookup.
3. Apply the tabular Q-learning update, handling terminal transitions.
4. Calculate the configurable training reward from approved observations.

Check an exercise with `python check_exercises.py 1` (use 2, 3, or 4 for the others). Run `python check_exercises.py` to check all four. The checks give small examples; they do not contain the exercise answers. See [the live-coding guide](docs/LIVE_CODING.md) and [setup checklist](docs/SETUP_CHECKLIST.md).

The game engine under `game/` is protected. Do not change its physics, pipes, scoring, observations, or action interface.

## Train, watch, and evaluate

After completing all four TODOs:

```bash
python main.py train
python main.py history
python main.py evaluate
python main.py watch
```

Each `train` starts with a new empty Q-table. The official limit is **3,000 training episodes per run**; checkpoints can be selected using the public validation maps. Independent experiments are unlimited, but do not resume, combine, or warm-start tables. `evaluate` uses the public validation seeds and does not learn. `watch` plays the selected model with learning and exploration disabled.

To set a reproducible training seed or checkpoint interval:

```bash
python main.py train --seed 29 --episodes 3000 --checkpoint-every 1000
```

More commands and experiment ideas are in [Experiments](docs/EXPERIMENTS.md).

## Validate and submit

```bash
python main.py validate
python main.py submit
```

`submit` creates a data-only ZIP under `submissions/`. Send that ZIP only. Do not submit your source folder or virtual environment. To save to a different path, use `python main.py submit --output submissions/my_run.zip`. Read the [submission instructions](docs/SUBMISSIONS.md) before submitting.

## Competition rules

- The policy receives **exactly seven approved observations**: `bird_y`, `bird_velocity`, `pipe_distance`, `gap_center_y`, `distance_to_ground`, `distance_to_ceiling`, and `pipe_velocity`.
- The policy returns exactly two possible actions: **0 = WAIT**, **1 = FLAP**.
- Use tabular Q-learning and a discrete state derived only from those observations.
- Inference must use the trained Q-table only. No heuristic action overrides or other decision systems.
- Every run starts from an empty Q-table and counts at most **3,000 training episodes**. Checkpoint selection is allowed within that budget.
- ChatGPT assistance is allowed; you must understand and explain the code you submit.

## Common problems

- **`python` is not Python 3.11:** activate `.venv`, select its interpreter in VS Code, then check `python --version`.
- **`No module named pygame` (or another package):** activate `.venv` and run `python -m pip install -r requirements.txt` again.
- **Window does not open:** run `python main.py play` from the repository folder on a desktop; hosted Colab sessions are for headless training and cannot display the local Pygame window.
- **An exercise check fails:** run that numbered check, confirm the required function in `student_exercises.py` is implemented, then compare its input/output with the example in [Troubleshooting](docs/TROUBLESHOOTING.md).
- **Training still reports an exercise TODO:** finish all four exercises and rerun the numbered checks before training.

For help, see [Troubleshooting](docs/TROUBLESHOOTING.md), [Setup](docs/SETUP.md), and [Submissions](docs/SUBMISSIONS.md).

## Arcade graphics

See [display options and previews](docs/arcade/README.md). Original offline
pixel art, parallax scenery, animated bird and replay HUD are included.

## Difficulty

Pipe gaps are 144 pixels high (previously 160), requiring more precise
steering. Each generated transition still has a verified collision-free route.
Retrain agents for this geometry; earlier scores are not directly comparable.
