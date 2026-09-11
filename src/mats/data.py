"""Keep only the questions the subject model answers correctly without a cue,
so a planted cue has room to flip them.

Corpus loading and row-schema mapping live in `datasets.py`; this module is the
subject-dependent filtering layer. The split matters because the filter is what
makes difficulty bite: on an easy corpus `keep_answerable` selects confident
items that only a blunt cue can move, and blunt cues get narrated.
"""

from __future__ import annotations

from mats.cot import parse_answer
from mats.datasets import LETTERS as _LETTERS  # noqa: F401  (kept for callers)
from mats.datasets import arc_row_to_question as _row_to_question  # noqa: F401
from mats.datasets import load as load_corpus
from mats.datasets import load_arc_challenge
from mats.prompts import Question, build_prompt


def correctness_rate(backend, question: Question, *, k: int, seed: int) -> float:
    """Fraction of `k` no-cue rollouts whose parsed answer is the gold letter."""
    prompt = build_prompt(question)
    correct = 0
    for j in range(k):
        answer = parse_answer(backend.complete(prompt, seed=seed + j))
        correct += int(answer == question.gold)
    return correct / k


def keep_answerable(
    backend, questions: list[Question], *, threshold: float, k: int, seed: int, limit: int
) -> list[Question]:
    """Return up to `limit` questions with correctness_rate >= threshold."""
    kept: list[Question] = []
    for offset, question in enumerate(questions):
        if correctness_rate(backend, question, k=k, seed=seed + offset * k) >= threshold:
            kept.append(question)
        if len(kept) >= limit:
            break
    return kept
