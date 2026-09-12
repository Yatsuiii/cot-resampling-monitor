# SESSION_CONTRACT

Objective: Fix the two defects in keep_answerable that made the full grid run
6+ hours and biased its item selection. Parallelise both of its loops, and
thread max_tokens through so the filter generates under the same cap as the
cells it feeds.

Branch: resampling-monitor

Parent: 7f67201

Allowed files:
- /home/Yatsuiii/MATS/** only
- src/mats/data.py, scripts/sweep_phase1_kaggle.py, scripts/sweep_phase1.py,
  tests/**, .claude/SESSION_CONTRACT.md
- NOT: resample.py, signals scoring, metrics.py, experiment.py, repair.py

Non-goals:
- Changing the sweep, cue families, or detectors.
- Re-running anything here. No GPU on this machine.

The two defects, both in one function I never opened:

1. SPEED. keep_answerable loops over questions sequentially, and
   correctness_rate loops over its k rollouts sequentially inside that. It takes
   no max_workers. Finding 40 answerable items may scan 50-100 questions at 4
   generations each, per dataset, entirely serial. That runs BEFORE any cell, so
   the ThreadPoolExecutor added to run_cell never got reached.

2. CORRECTNESS, and this is the worse one. The filter calls
   backend.complete(prompt, seed=...) with no max_tokens, taking the 1024
   default, while the cells run at 8192. Measured chains are 5,589-9,719
   characters, well over 1024 tokens. So the filter truncates most of its own
   rollouts and marks items unanswerable because the chain was cut off rather
   than because the model was wrong. It selects for short-reasoning questions
   and biases the entire grid.

Defect 2 is the same truncation bug class already fixed at the answer boundary
and the mention boundary. I fixed those two where they reproduced and did not
read the rest of the file.

PRECOMMITTED:
H31 Parallelising must not change which questions are kept. With a fixed
    backend and seed, the kept list is identical for max_workers 1 and 16, and
    in the same order.
H32 The filter's max_tokens reaches backend.complete. Asserted by a recording
    backend that captures the kwargs it was called with.

Acceptance gates:
1. `python -m pytest -q` passes; all 108 existing tests unchanged.
2. A test asserts H31 across the early-stopping boundary, since `limit` makes
   order matter.
3. A test asserts H32.
4. The runner passes both max_tokens and max_workers into keep_answerable.

Verification:
- `python -m pytest -q`
- `git diff --stat` shows resample.py, metrics.py, experiment.py untouched

Status: active
