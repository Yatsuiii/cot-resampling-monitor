# SESSION_CONTRACT

Objective: Run the 12-question screening pilot from the second-opinion memo
(`~/mats-second-opinion-2026-09-06.md`, gate 1) to decide whether the
repair-policy experiment is viable. The pilot answers three questions: does the
clean prompt give adequate accuracy, does the authority cue move the answer
enough to study, and does a cued prefix keep pulling toward the cue once the
cue itself is removed from the prompt. Hard stop at 1 allocated GPU-hour or 2
active human hours, whichever comes first.

Branch: resampling-monitor

Parent: HEAD

Allowed files:
- MATS/src/mats/repair.py (new), MATS/tests/test_repair.py (new)
- MATS/src/mats/backend.py + backend_vllm.py: add a detailed-completion path
  returning finish_reason and token counts. Gate 4 and the memo's artifact
  lineage both require per-request finish reasons; `complete()` returning bare
  text cannot distinguish truncation from refusal.
- MATS/.claude/SESSION_CONTRACT.md, MATS/README.md
- NOT: the main-run harness (5 policies, annotation tooling, figures) - that is
  gated behind the pilot passing.
- NOT: any application prose, exec summary, or form answers.

Non-goals:
- No sentence-level importance sweep. The memo explicitly rules it out for this
  design; the baselines come from full generations.
- No annotation tooling or the three annotation-dependent policies
  (source-linked deletion, sham deletion, explicit correction) until the pilot
  passes.
- No new cue types, no new datasets, no model change.
- No retrospective relabelling of the original no-mention endpoint.

Baseline:
- `cd MATS && python -m pytest -q` -> 51 passed.
- `python scripts/smoke.py --backend dummy` -> exits 0.

Acceptance gates: the memo's screening thresholds, precommitted.
1. Clean accuracy on the 12 pilot questions >= 75%.
2. The authority cue raises the mean cue-answer rate by >= 25 percentage points
   over the clean condition.
3. Continuing an intact cued prefix under a CLEAN prompt retains >= 15
   percentage points of excess cue-following over restarting from the clean
   prompt.
4. Measurement audit before any interpretation: < 5% unparsable or truncated
   generations, and no materially unequal truncation between conditions.
5. `python -m pytest -q` passes and the pilot script runs GPU-free against
   DummyBackend.

Failing 1-3 kills the repair experiment. Failing 4 means fix instrumentation
before reading any number.

Verification:
- `cd MATS && python -m pytest -q` -> all pass.
- Pilot kernel completes inside 1 allocated GPU-hour; log actual request count,
  finish reasons, and token counts, not just wall-clock.
- Manual: Raghav reads a random sample of raw rendered prompts and completions
  to confirm the clean/cued prompts and the transplanted prefixes are what we
  think they are.

Status: active
