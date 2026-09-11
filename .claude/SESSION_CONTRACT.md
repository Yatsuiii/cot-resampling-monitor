# SESSION_CONTRACT

Objective: A Kaggle runner for the Phase 1 sweep, so the grid can execute on a
free T4. Follows the lifecycle already proven in scripts/repair_clean_kaggle.py:
start vLLM, wait for health, render through the model's chat template, and write
partial output after every cell so a killed session still leaves evidence.

Branch: resampling-monitor

Parent: e6fd72a

Allowed files:
- /home/Yatsuiii/MATS/** only
- specifically: scripts/sweep_phase1_kaggle.py (new), tests/**,
  .claude/SESSION_CONTRACT.md
- NOT: src/mats/** (the library is done for Phase 1), resample.py, metrics.py,
  experiment.py, repair.py, scripts/repair_clean_kaggle.py

Non-goals:
- Running it. No GPU here; the local check is import and argument handling only.
- Phase 2 or Phase 3.
- Changing the sweep library, which is committed and tested at e6fd72a.

Carried over from repair_clean_kaggle.py because it is already proven on this
hardware: _wait_ready polling on /health, atexit server termination,
float16 with max-model-len 6144 and gpu-memory-utilization 0.92 on a single T4,
chat-template rendering via AutoTokenizer, and partial output after each unit of
work. That last one matters most - a Kaggle session that dies at hour four must
not lose the first three.

PRECOMMITTED, fixed before implementation:
H29 Partial output is written after every cell, not at the end. A session killed
    mid-grid leaves a summary containing the cells that finished and their trace
    files, and the summary is recomputable from those traces.
H30 A cell that raises is recorded in failed_cells and the run continues. One
    unbuildable family must not abort a grid that costs GPU-hours.

Acceptance gates:
1. `python -m pytest -q` passes; all 100 existing tests unchanged.
2. The script imports without vLLM, transformers or a GPU present, so a syntax
   or import error surfaces here rather than after a Kaggle queue wait.
3. A test asserts the cell-loop writes partial state after each cell, driven on
   DummyBackend with no server.
4. Model, token caps, workers and grid are environment-overridable, matching how
   repair_clean_kaggle.py is parameterised.

Verification:
- `python -m pytest -q`
- `python -c "import ast; ast.parse(open('scripts/sweep_phase1_kaggle.py').read())"`
- `git diff --stat src/mats` is empty

Status: active
