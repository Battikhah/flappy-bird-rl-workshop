# Portable submissions and competition contract

Run `python main.py evaluate`, `python main.py validate`, then `python main.py submit`.
The submission ZIP contains exactly:

- `model.npz`: one finite float32 Q array, `(state_count, 2)`, loaded without pickle.
- `model.json`: declarative height/velocity/time bins, seven input names, two actions,
  table dimensions/checksum, run identity, full episode budget, checkpoint episode,
  RNG seed, map plan, learning/reward/environment configuration, and history checksum.
- `training.csv`: full run history, including episodes after the selected checkpoint.

JSON/NPZ are sufficient for inference. History is required for submission review.
No student Python, saved policy object, neural network, or external model is included.
The trusted evaluator reconstructs only the supported encoder and Q lookup.

Enforceable artifact checks: exact input/action contract, supported bin schema,
finite values and dimensions, bounded files (16 MiB each, at most one million
states), zero-initialization declaration, official run kind and 3,000 cap, contiguous
logged episodes, seed/configuration consistency, public-only checkpoint selection,
serialization roundtrip, and reproducible legal actions on numerical probes.

Rules requiring source-code inspection or trusted execution: whether training
actually used only Q-learning, whether the initial values really were empty,
whether another table/expert/model influenced learning, whether logged episodes
are truthful, and whether reward/action code used unauthorized information.
Checksums detect mismatches, not deliberate cheating. Passing validation is not
proof of source compliance. ChatGPT-assisted implementation is allowed.

Organizers must evaluate with their own trusted project, not run submitted Python.
The trusted `main.py validate --model submission.zip` and
`main.py evaluate --model submission.zip --seed PUBLIC_SEED --seeds COUNT` commands
load data only. Hidden final seeds are supplied separately by the organizer; they
are never shipped in this project. No tournament orchestration is included.
