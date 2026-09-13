# SESSION_CONTRACT

Objective: Make the corpus sample a sample. `keep_answerable` scans in the
order it is handed and stops at its limit, so the 09-12 grid's entire 40-item
"MMLU" arm came from the first 45 questions of one subject. Then re-run the
five MMLU cells so that arm is both a fair sample and internally consistent.

Branch: resampling-monitor

Parent: be2a1ea (plus the uncommitted deadline and duplicate-family fixes)

Allowed files:
- /home/Yatsuiii/MATS/** only
- src/mats/datasets.py, scripts/sweep_phase1_kaggle.py, tests/**,
  .claude/SESSION_CONTRACT.md
- NOT: data.py, sweep.py scoring, cues.py, signals.py, resample.py, metrics.py,
  experiment.py, repair.py, prompts.py

Non-goals:
- Changing keep_answerable. Its prefix-and-stop behaviour is deliberate: order
  decides which questions win, so concurrency must not be able to change them.
  The defect is the order it is handed, not what it does with it.
- Re-running the ARC half. Its five cells are sha256-verified and unaffected by
  any fix in flight; ARC and MMLU are never compared item-wise, so one run_id
  per corpus is sound.
- Changing MAX_TOKENS, the gate, or what counts as a positive.

The defect, from the 09-12 traces:

  All 40 MMLU items are abstract_algebra, corpus indices 0-44 of 14,042. The
  filter scanned 45 questions, kept 40 at an 89 percent pass rate, and stopped.
  `cais/mmlu` `all` is grouped by subject, so a prefix of it is one subject.
  The results table calls that arm "mmlu", which overstates it.

  ARC is affected by the same mechanism and merely got away with it: its
  validation split is not semantically grouped, so a prefix is close to a
  random sample. The fix is applied to every corpus rather than special-casing
  MMLU.

Acceptance gates:
1. Taking a prefix of what the sweep hands the filter is a fair sample: a
   subject-grouped corpus must not yield a single-subject prefix.
2. The order is seeded, so the sample is reproducible from the manifest.
3. The runner passes every corpus through it, with no per-corpus special case.
4. Full suite passes, with a test that fails against the 09-12 behaviour.

Verification:
- `python -m pytest -q`
- `ruff check src/ scripts/ tests/` (7 pre-existing findings, no new ones)
- Re-run evidence: 5 MMLU cells, one item set, subjects spread across MMLU

Status: active
