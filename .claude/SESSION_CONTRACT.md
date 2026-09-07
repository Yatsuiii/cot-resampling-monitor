# SESSION_CONTRACT

Objective: Phase 1 of the repair experiment on 32 questions. Deliver two things
from one run: (a) the dose-response result - how strongly the cue bit predicts
how much contamination survives removing the cue from the prompt, which the
15-question pilot measured at r=0.670, bootstrap CI [0.345, 0.858]; and (b) the
32 donor prefixes and the explicit-correction baseline that phase 2's
annotation-dependent policies need. Phase 2 (targeted vs sham deletion) is a
separate checkpoint gated on Raghav's annotation.

Branch: resampling-monitor

Parent: HEAD

Allowed files:
- MATS/src/mats/prompts.py (the generic `note` parameter - done)
- MATS/src/mats/repair.py, MATS/tests/test_repair.py, MATS/tests/test_prompts.py
- MATS/.claude/SESSION_CONTRACT.md, MATS/README.md
- NOT: annotation tooling or the deletion policies - phase 2.
- NOT: any application prose, exec summary, or form answers.

Non-goals:
- No sentence-level importance sweep, no capitulation curves. Dead endpoints.
- No question selection based on any outcome. Questions are a fixed contiguous
  block in dataset order; the only filter is clean accuracy, a property of the
  control condition, applied post-hoc.
- No re-tuning of the cue after seeing results.
- No new datasets or models.

Baseline:
- `cd MATS && python -m pytest -q` -> 61 passed.
- `python scripts/smoke.py --backend dummy` -> exits 0.
- Pilot v2 (15 questions after filter): cue_effect 0.683, residual pull 0.208
  (median 0.000, 8/15 at exactly zero), r=0.670 for cue strength vs residual.

Acceptance gates: precommitted before the run, not adjustable after.
1. The run completes 32 questions x 4 conditions x 8 generations within 2.5
   allocated GPU-hours, or reports how far it got.
2. Unusable (unparseable) generations < 5% in every condition, with truncation
   reported separately per condition.
3. The dose-response correlation is reported with a question-clustered
   bootstrap CI. A CI excluding zero is the positive result; a CI spanning zero
   is reported as inconclusive, not spun.
4. The 8/15 "complete repair" subgroup from the pilot is checked for
   replication: report what fraction of the 32 have residual pull exactly 0.
5. `python -m pytest -q` passes and the kernel path runs GPU-free on
   DummyBackend before it touches a GPU.

Verification:
- `cd MATS && python -m pytest -q` -> all pass.
- Kernel logs actual call count, per-condition finish reasons, and elapsed time.
- Manual: Raghav reads a random sample of rendered prompts for all four
  conditions, confirming the source-removal prompt really is the clean question
  carrying a cued prefix, and the explicit-correction note reads sensibly.

Status: active
