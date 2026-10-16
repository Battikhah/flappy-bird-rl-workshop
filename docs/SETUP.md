# Setup in VS Code

Install **Python 3.11** for this workshop, VS Code, and its Python extension. Extract the
student ZIP first; do not run Python from inside the archive. Use **File → Open
Folder** to open the extracted `student` directory (the one containing `main.py`).
Open **Terminal → New Terminal**.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, open the command palette → **Python: Select Interpreter** → select
this folder's `.venv`. Run `python --version` and `python main.py play`.
Check that the activated interpreter reports **3.11.x** before installing.
Do not use Python 3.14 with these pinned dependencies: the rehearsal exposed
a wheel/interpreter mismatch. Other Python versions have not been release-tested.
If `python3.11` is missing, install/select Python 3.11 first; do not silently
substitute an unversioned `python3`. With pyenv, select your installed 3.11
version, or use its full executable path when creating the environment.

If PowerShell blocks activation, do not change system security policy: use
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and that executable
instead of `python` in commands. On macOS/Linux the equivalent is `.venv/bin/python`.
On Debian/Ubuntu a missing `venv` module may require your administrator to install
the matching `python3.11-venv` package where available. Use a Python version with available NumPy/Pygame wheels.

Pygame is only initialized by visual modes. Training, evaluation, validation,
submission, and the exercise checks work without a window or display. Matplotlib
writes learning curves to PNG; it does not open a window. Runtime needs NumPy,
Pygame (only for windows), and Matplotlib; pytest is supplied for optional testing.
No GPU, asset downloads, accounts, or external models are needed.

Optional Colab fallback: upload/extract **student.zip**, change into its `student/`
directory, install `requirements.txt`, and run the same headless commands with
`!python`. Download the run directory/ZIP before the runtime disconnects. Interactive
Pygame windows are not supported in Colab; use VS Code locally to play/watch.
