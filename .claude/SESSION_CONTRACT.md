# SESSION_CONTRACT

Objective: Make the Phase 1 grid runnable on Kaggle by removing the
cap/window mismatch that made every vLLM completion return 400, and make that
class of mismatch impossible to repeat silently.

Branch: resampling-monitor

Parent: bd4b74c

Allowed files:
- /home/Yatsuiii/MATS/** only
- scripts/sweep_phase1_kaggle.py, tests/**, .claude/SESSION_CONTRACT.md
- NOT: src/mats/sweep.py scoring, cues.py, signals.py, resample.py, metrics.py,
  experiment.py, repair.py, data.py

Non-goals:
- Changing the cue families, the detectors, the gate, or what counts positive.
- Changing MAX_TOKENS itself. The 8192 cap is measured and stays.
- Re-running anything here. No GPU on this machine.

The defect, from the kernel log of yoursonly/mats-phase1-full-grid:

The runner starts vLLM with `--max-model-len 6144` and generates with
`max_tokens=8192`. vLLM rejects any request whose max_tokens exceeds the
context window, so all 48 completion requests returned 400 Bad Request and the
run died on the first call out of `keep_answerable`, after the model had
already loaded. Two numbers in the same file, 24 lines apart, that must agree
and had nothing enforcing it. The cap was raised to 8192 in 1980d27 to stop
truncating chains; the window was never raised with it.

Evidence:
- log line 135: `max_seq_len=6144`; scripts/sweep_phase1_kaggle.py:29 `8192`
- log lines 225-272: 48 consecutive `POST /v1/completions 400 Bad Request`,
  the first one immediately after the corpora finish downloading
- log line 168: `GPU KV cache size: 27,264 tokens` - the window can be raised
  without running out of KV cache
- sweep.py:37 records why 8192 and not less: at a 4,096 cap, 3 of 8 measured
  chains truncated

Prompt reserve is measured, not guessed
(scratchpad/measure_prompt.py over the real corpora, repo row adapters):

  worst prompt in the entire grid   5,843 chars  (mmlu / few_shot, 3-shot)
  p99                               3,153 chars
  median                            1,281 chars
  ~1,950 tokens at a conservative 3 chars/token

Reserve 3,072 gives ~57 percent headroom over that worst case, so window
= 8,192 + 3,072 = 11,264, which fits the 27,264-token KV cache.

Acceptance gates:
1. The context window is derived from the generation cap in code, not written
   as a second literal. Grepping the runner finds no independent window number.
2. MAX_TOKENS and MAX_WORKERS have exactly one definition in the repo
   (src/mats/sweep.py). The Kaggle runner and the kernel script read it.
3. A preflight completion at the configured cap runs before the corpus filter,
   so a mismatch fails in seconds rather than after the model loads.
4. Full test suite passes, with a new test covering the derived-window
   invariant.

Verification:
- `python -m pytest -q` from /home/Yatsuiii/MATS
- `python -c "import ast; ast.parse(open('scripts/sweep_phase1_kaggle.py').read())"`
- grep the runner for a literal max-model-len value: must find none

Status: active
