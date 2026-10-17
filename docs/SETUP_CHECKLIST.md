# Student setup checklist

Complete before the session:

- [ ] Extract `student.zip`; open the folder containing `main.py` in VS Code.
- [ ] Install Python **3.11**, VS Code, and its Python extension.
- [ ] Follow `SETUP.md` for your OS; create and activate `.venv`.
- [ ] Select this `.venv` in **Python: Select Interpreter**.
- [ ] `python --version` reports 3.11.x; `python -m pip install -r requirements.txt` succeeds.
- [ ] `python main.py play` opens a window; SPACE flaps, ESC exits.
- [ ] `python check_exercises.py 1` reports Exercise 1 (expected before live coding).
- [ ] Keep the extracted project on a writable local folder, outside the ZIP.

During class:

- [ ] Complete the four exercises; run each numbered check, then all checks.
- [ ] Run `train`, `evaluate`, `watch`, `history`, `validate`, then `submit`.
- [ ] Use a new zero-table run for every experiment; maximum 2,400 episodes.
- [ ] Record settings and public validation results; submit only the generated ZIP.

If a command cannot find `main.py` or `requirements.txt`, your terminal is in the
wrong directory. Reopen the project folder; do not edit paths in the game code.
If rendering fails, continue with headless training and ask the instructor to
show a replay. Colab is a headless fallback, not a desktop Pygame environment.
