# SESSION_CONTRACT

Objective: Add the Phase 1 CLI so the elicitation sweep can actually run. It
iterates the cue-family x corpus grid, filters questions to those the subject
answers correctly, writes traces and a manifest, and reports gate G-A per cell.
Last piece before Phase 1 executes.

Branch: resampling-monitor

Parent: 40b4535

Allowed files:
- /home/Yatsuiii/MATS/** only
- specifically: scripts/sweep_phase1.py (new), src/mats/sweep.py (cell-runner
  glue only), tests/**, .claude/SESSION_CONTRACT.md
- NOT: resample.py, signals scoring, metrics.py, experiment.py, repair.py

Non-goals:
- Running against a real model here. No GPU on this machine and no vLLM; the
  local path is DummyBackend only. The real run goes to Kaggle or a rented GPU.
- Phase 2 detection or Phase 3 statistical repairs.
- Making any repository public.

Baseline at 40b4535: 97 tests. cues.py, datasets.py and sweep.py exist;
run_cell and summarise work; nothing iterates the grid or writes to disk.

PRECOMMITTED, fixed before implementation:
H27 A cell that cannot deliver its cue fails loudly rather than reporting a
    zero flip rate. Specifically, a few-shot family with too few bias-letter
    examples raises, and that raise is recorded in the results file as a failed
    cell rather than silently omitted.
H28 Every number in the summary file is recomputable from the traces written
    beside it. Asserted by a test that reloads traces and recomputes.

Acceptance gates:
1. `python -m pytest -q` passes; all 97 existing tests unchanged.
2. `python scripts/sweep_phase1.py --backend dummy --n 8` runs end to end with
   no network, writing results/phase1/<run_id>/ containing a manifest, one
   trace file per cell, and a summary.
3. The summary records, per cell: n_items, n_flipped, n_positive,
   verbalization rate given flip, and passes_G_A.
4. A test asserts H27 and a test asserts H28.
5. results/ is not gitignored; traces are written there and committed by the
   caller, not excluded.

Verification:
- `python -m pytest -q`
- `python scripts/sweep_phase1.py --backend dummy --n 8` twice; the second run
  writes a new run_id and both summaries are recomputable from their traces
- `git diff --stat` shows the Phase 2 scoring path untouched

Status: active
