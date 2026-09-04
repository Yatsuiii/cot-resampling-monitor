# SESSION_CONTRACT

Objective: Build the analysis harness for a resampling-only CoT-unfaithfulness
monitor (MATS 12.0 Neel stream, Option A). The harness must let Raghav run the
experiment on free Kaggle GPU and produce an ROC/PR comparison of three
detectors: (1) sentence-level resampling-importance profile, (2) an LLM
CoT-monitor baseline, (3) final-answer-entropy baseline. Raghav owns experiment
design, all research decisions, interpretation, sanity-checking, and every
written word of the application. Claude writes tooling only.

Branch:
resampling-monitor

Parent: 3412f0b

Allowed files:
- MATS/** (this repo only)
- specifically: src/**, scripts/**, notebooks/**, tests/**, configs/**,
  prompts/**, README.md, requirements.txt, pyproject.toml
- NOT: any application prose (exec summary, form answers, writeup) — those are
  Raghav's and live outside this repo or in a clearly-marked drafts area he
  writes himself.

Non-goals:
- No generating the executive summary, application-form answers, or the research
  writeup. Claude may build plotting/reporting code, not the narrative.
- No attention-based methods (attention aggregation / suppression) in this
  checkpoint — black-box resampling only. Add later only if core is done.
- No paid APIs, no paid GPU. Kaggle free tier (2xT4) + HuggingFace weights only.
- No model internals / probing. This is a pure black-box project.
- No scope expansion to user-models or other Neel problem areas.

Baseline:
- `cd MATS && python -m pytest -q` -> currently no tests, exits 5 (no tests
  collected). After first checkpoint: passes.
- `git log --oneline` -> shows `init: repo skeleton` at 3412f0b.

Acceptance gates:
1. `python -m pytest -q` passes with tests covering: CoT sentence splitting,
   the semantic-similarity dedup filter, resampling-importance computation on a
   synthetic fixture (no model calls), and ROC/PR metric computation.
2. `python scripts/smoke.py --backend dummy` runs the full pipeline end to end
   on a stubbed model backend and writes an ROC plot PNG + a metrics JSON to
   outputs/, with no network and no GPU.
3. Model-serving is behind a single `Backend` interface with a `VLLMBackend`
   (Kaggle) and a `DummyBackend` (tests); swapping backends touches no analysis
   code.
4. A Kaggle-runnable notebook (notebooks/run_kaggle.ipynb) exists that: starts
   vLLM on Qwen3-4B, runs the cue-flip vs control experiment at configurable
   n_prompts / n_rollouts, and saves rollouts to disk so re-analysis needs no
   regeneration.
5. README documents: how to run the smoke test, how to run on Kaggle, the exact
   knobs (n_prompts, n_rollouts, model), and the known limitations (rollout
   count below the 100 best-practice, single small model, small n).

Verification:
- `cd MATS && python -m pytest -q` -> all pass.
- `python scripts/smoke.py --backend dummy` -> exits 0, outputs/roc.png and
  outputs/metrics.json created.
- Manual: read scripts/smoke.py and confirm the resampling-importance and
  base-rate handling match what Raghav specified before he defends it on the
  form.
- Manual: Raghav runs notebooks/run_kaggle.ipynb on Kaggle with n_prompts=4,
  n_rollouts=8 and confirms it completes and the cue flips answers at a
  plausible rate.

Status: active
