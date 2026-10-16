# Four exercises, about 25 minutes

Only edit `student_exercises.py` during the guided part. Read each docstring and
complete one TODO at a time. Raising a clear “Exercise N” error is intentional:
the starting project cannot train until your functions exist. Do not replace the
TODO with a call into the engine's reference flight controller.

| Time | Exercise | Check |
|---|---|---|
| 0–3 min | Initialize a fresh table: one row per discrete state, one column per action | `python check_exercises.py 1` |
| 3–9 min | Epsilon-greedy: random exploration or the largest value in the current row | `python check_exercises.py 2` |
| 9–18 min | One-step Q update; collisions have no future return | `python check_exercises.py 3` |
| 18–23 min | Scalar reward: survival, pipe success, collision; optional observed alignment | `python check_exercises.py 4` |
| 23–25 min | Check all four; launch your first run | `python check_exercises.py` then `python main.py train` |

Check each exercise separately even if earlier ones are unfinished: the checks
isolate the function being tested. Checks give behavioral examples, not function
implementations. Passing these small checks is a starting point; inspect learning
curves and validation outcomes too.

For Exercise 2, use the provided `rng` so repeated seeds reproduce exploration.
Return Python integers, not NumPy scalar actions or booleans. Ties choose WAIT.
After completing action selection, a full `train` may report missing Exercise 4
before Exercise 3: the loop calculates reward before updating Q. Use the isolated
Exercise 3 check next, then finish reward as taught.
For Exercise 3, `terminated` means a collision. The loop ends on either environment
flag, but time/pipe truncations still bootstrap. Only one Q entry should change.
For Exercise 4, do not change the official score or infer hidden future gaps.

After training, run `evaluate`, then `watch`, `validate`, and `submit`. The first
greedy table before learning naturally waits and dies. Rewarding survival alone
may not teach good pipe alignment. A trained table can also get worse later;
checkpoint selection exists for that reason.
