# Troubleshooting

| Symptom | Check / correction |
|---|---|
| `Exercise N` error | That TODO is still incomplete. Run `python check_exercises.py N`; ask the instructor for one hint at a time. |
| All actions rejected | Return built-in integer 0/1, not a bool or NumPy scalar. |
| No learning / dies immediately | Run exercise checks; confirm epsilon explores, Q entries change, and you encode the next observation after stepping. |
| Q values explode | Check alpha/gamma bounds, current-value subtraction, and the terminal future-value mask. |
| Reward exceeds pipe score | Expected: shaped reward and official pipes are different. Compare official validation scores. |
| Training looks worse than watch | Training explores; watch is greedy with a frozen table. Compare validation checkpoints. |
| No trained run | Finish all exercises and run `python main.py train` in this folder. |
| Existing output directory | Use a new `--output` or omit it for an automatic fresh directory. Never resume an official run. |
| Existing submission ZIP | Use a new `--output submissions/name.zip`; files are not overwritten. |
| Invalid model/history | Re-run `evaluate`, then `validate`. Keep JSON, NPZ, and the referenced CSV together. Do not hand-edit counts, table values, or checksums. |
| Unsupported encoder | Use the supplied declarative bin-edge format. Custom inference Python is not accepted. |
| Roundtrip or checksum failure | Model files may come from different checkpoints. Restore an intact pair from your original run, then re-export. |
| Public validation overlap | Choose a training map range outside 3000000–3000099. |
| No Pygame window | Check the VS Code interpreter and desktop session. Train/evaluate headlessly; Colab cannot display interactive Pygame windows. |
| PowerShell activation blocked | Use `.venv\Scripts\python.exe` directly; see setup. |
| Import/module missing | Install requirements using the same Python interpreter that runs `main.py`; avoid executing inside the ZIP. |

Physics is fixed: y increases downwards; negative velocity moves upward. Flap is
an impulse, not a permanent upward mode. Edge clearances are measured from the
bird's edges. Collision termination and artificial truncation are different.
If a later checkpoint is worse, select an earlier one using public validation.

If validation fails, the command reports an error and exits without publishing
anything. Correct your code/configuration, train a new independent run when needed,
and retry. Keep old run histories for your experiment notes.
