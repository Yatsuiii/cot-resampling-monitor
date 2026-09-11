# MATS 12.0 submission plan

## Outcome Ledger

### Decision 1

Decision: Submit an honest negative-result / feasibility study. Do not claim that
the proposed monitor works, and do not spend the remaining GPU budget on the
failed repair protocol.

Lane: mechanistic-interpretability research / model forensics.

Artifact: one Google Doc with a <=600-word executive summary, three evidence
figures, methods, limitations, LLM-use disclosure, and a link to the tested
harness.

Acceptance gate:

1. Reconcile every denominator and run ID, especially the separate authority
   summaries (12/15 calibration versus 8/10 main run).
2. Recover or inspect the raw traces/logs supporting each headline number. If a
   number cannot be audited, label it user-reported or remove it.
3. Include a figure for cue elicitation and a figure for the flat resampling
   curves. Do not include the dummy ROC as a real-model result.
4. State that the positive class was empty, so AUC is undefined, not zero.
5. Report the v5/v6 repair smoke only as an abandoned follow-up: n=2, residual
   pull 0, source-removal truncation 12.5%, quality gate failed.

Result: the authority cue produced flips only in traces that visibly mentioned
the cue in the reported ten-item run; the opinion and weak few-shot cues did not
produce flips; the resampling curves had no identifiable capitulation point.
The result is a boundary condition on this elicitation/model/task, not evidence
that Qwen3-4B is faithful or that hidden influence is absent.

Next action: freeze the analysis, write the document, and complete a manual
audit of five randomly selected traces before submission.

Kill condition: if raw traces and exact run denominators cannot be recovered,
do not present aggregate numbers as verified research. Submit nothing rather
than an unauditable positive claim.

Status: continued (submission preparation).

## Proposed title

**Trying to Elicit Hidden Answer-Key Influence in Qwen3-4B**

## Executive-summary spine

Question: can black-box sentence-resampling statistics detect an answer-changing
cue that the chain-of-thought never mentions?

Setup: Qwen3-4B on ARC-Challenge; opinion, external-authority, and three-shot
distributional cues; positive pre-specified as cue-induced answer change with no
cue mention in the complete CoT.

Main result: the protocol never produced a usable hidden-cue positive. The
opinion cue flipped 0/15; the weak three-shot cue flipped 0/10; the authority
cue flipped 8/10 in the main run, but all 8 flipped traces mentioned the answer
key. Therefore ROC/AUC comparison is undefined. On eight authority-flipped
traces, sentence-prefix resampling yielded already-high, mostly flat cue-answer
probabilities rather than a localized capitulation point.

Interpretation: this is an elicitation boundary, not a faithfulness result. A
cue weak enough to remain unmentioned did not move this model's answer; a cue
strong enough to move it became part of the stated reasoning. The main practical
lesson is to validate the positive class and audit the full trace before judging
an interpretability monitor.

## Figures

1. Cue-family outcomes: flip rate and mention rate, with `n` printed on every
   bar; keep calibration and main authority runs separate.
2. Eight per-trace capitulation curves with the full-CoT baseline marked; show
   that the curves start high and remain flat.
3. Optional appendix only: v5 versus v6 truncation gate and repair-smoke rates.

## 48-hour execution order

1. Reconcile counts and inspect raw traces/logs; remove unsupported claims.
2. Generate the two main figures and write methods/limitations.
3. Draft the executive summary and form answers; explicitly disclose LLM use and
   human sanity checks.
4. Run the harness tests and a final claim-to-evidence audit; submit early.
