# Improve your agent with controlled experiments

First complete all exercises and save a baseline run. Predict one change's effect,
change only that factor, train a **fresh** table, and compare greedy validation at
the same episode budget. Repeat promising comparisons with action RNG seeds 11,
29, and 47 if time allows. Keep map sampling and public validation seeds fixed.
Training scores include exploration; validation scores do not.

Beginner challenges:

1. **Reward shaping:** compare alignment weight 0 versus 0.4. Does the denser
   observed alignment signal help at 1,000, 2,000, and 3,000 episodes?
2. **Learning rate:** compare alpha 0.04 versus 0.08, keeping gamma 0.98. Is early
   learning faster? Does the last checkpoint beat the best earlier checkpoint?
3. **Exploration:** compare epsilon floor 0.01 versus 0.002 or change decay. How
   do the exploratory training curve and frozen validation curve differ?

Advanced challenges:

1. **State discretization:** compare 8-pixel versus 16-pixel `height_edges`, or
   change velocity/time edges. Predict the table size first; more bins mean fewer
   visits per Q entry within 3,000 episodes. The portable encoder supports only
   these declarative edges, derived from approved inputs. A custom Python encoder
   cannot be submitted; new encoder types need a trusted organizer format extension.
2. **Training seeds:** change `training.map_seed`, or add `seed_pool: 100` to
   repeat a bounded pool. Remove it or use `null` for fresh maps. Does validation
   reveal overfitting? Public validation seeds cannot appear in training.

For each independent run, record:

| Run directory | Hypothesis / one change | RNG seed | Episodes | Selected checkpoint | Validation mean / median / cap% | Observation / next step |
|---|---|---|---|---|---|---|
| | | | | | | |

Copy `configs/workshop_student.json` to a named JSON file before editing it;
`--config` selects it. Each model JSON records the actual parameter and encoder
configuration. Keep notes about code changes to your reward/Q update too: metadata
weights alone cannot describe arbitrary edits. The run CSV records every episode;
`learning_curve.png` gains frozen validation points after `evaluate`.

Up to 3,000 episodes per official run; unlimited independent runs. Do not merge
Q-tables or initialize from a previous run. Use public validation for development;
hidden seeds belong only to final organizer evaluation. Submit an honest unsuccessful
comparison as well as a successful one. ChatGPT help is allowed; document what you changed.
