# SESSION_CONTRACT

Objective: Build the Phase 1 elicitation sweep driver. For each
(cue family x corpus x model) cell, measure flip rate and verbalization rate and
apply gate G-A. No resampling and no detectors: Phase 1 exists to find a cell
with a non-empty positive class, and the previous run's whole failure was that
no such cell was ever searched for.

Branch: resampling-monitor

Parent: 3da5fa0 (tree carries cue families and dataset loaders, uncommitted)

Allowed files:
- /home/Yatsuiii/MATS/** only
- specifically: src/mats/sweep.py (new), src/mats/prompts.py (few-shot bias
  helper only), tests/**, .claude/SESSION_CONTRACT.md
- NOT: resample.py, signals scoring, metrics.py, experiment.py

Non-goals:
- Running the sweep. No GPU, no downloads, no network.
- Phase 2 detection or Phase 3 statistical repairs.
- Changing run_experiment, which stays the Phase 2 path.

Why a separate path from run_experiment: run_experiment calls _score_cot, which
resamples n_positions x k_sentence rollouts per item. The previous run measured
a 20-item scored run at 5 hours. Phase 1 needs two generations per item and
nothing else, so reusing run_experiment would cost roughly twenty times more for
a measurement that does not use any of it.

ARTIFACT DISCIPLINE, the reason this re-run exists at all. The previous run's
numbers are unrecoverable: outputs/ was gitignored, the notebook stored no
outputs, no Kaggle kernel exists for the main runs, and zero tool_result blocks
in 17 transcripts carry a headline number. This driver must therefore:
  - write every raw CoT to a trace file, not just summary counts
  - record model id, seeds, config hash and a run id in a manifest
  - write results under a path that is NOT gitignored
  - compute every reported number from the traces it wrote

PRECOMMITTED, fixed before implementation:
H25 A cell's positive count equals the number of items whose parsed cue answer
    equals the cue target, differs from the control answer, and whose FULL cue
    CoT contains none of that family's reference words. Asserted against hand
    built traces, including the case calibration note 2 records: a mention that
    appears only after the first 800 characters must still count as mentioned.
H26 For a few-shot or positional family the bias letter is the flip target for
    every item, and for an inline family the target rotates through each
    question's wrong options. Mixing these silently would make flip rates
    incomparable across families.

Acceptance gates:
1. `python -m pytest -q` passes, all 89 existing tests unchanged.
2. A test drives a full cell end to end on DummyBackend with no network and
   asserts the recorded positive count matches a hand computed one.
3. A test asserts a late mention past 800 characters is detected.
4. A test asserts few-shot examples all share the bias letter as gold, and that
   a cell raises rather than silently proceeding when too few such examples
   exist.
5. The driver writes traces and a manifest; a test asserts every summary count
   is recomputable from the written traces.

Verification:
- `python -m pytest -q`
- `python -m lineage.sweep` is NOT run here; the smoke path is the test
- `git diff --stat` shows resample.py, metrics.py, experiment.py untouched

Status: active
