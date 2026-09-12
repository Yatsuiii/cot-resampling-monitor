"""Keep only the questions the subject model answers correctly without a cue,
so a planted cue has room to flip them.

Corpus loading and row-schema mapping live in `datasets.py`; this module is the
subject-dependent filtering layer. The split matters because the filter is what
makes difficulty bite: on an easy corpus `keep_answerable` selects confident
items that only a blunt cue can move, and blunt cues get narrated.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from mats.cot import parse_answer
from mats.datasets import LETTERS as _LETTERS  # noqa: F401  (kept for callers)
from mats.datasets import arc_row_to_question as _row_to_question  # noqa: F401
from mats.datasets import load as load_corpus
from mats.datasets import load_arc_challenge
from mats.prompts import Question, build_prompt


def correctness_rate(backend, question: Question, *, k: int, seed: int,
                     max_tokens: int = 8192, max_workers: int = 1) -> float:
    """Fraction of `k` no-cue rollouts whose parsed answer is the gold letter.

    `max_tokens` must match what the experiment itself generates at. The default
    on `complete()` is 1024, and measured chains run well past that, so a filter
    left on the default truncates its own rollouts and marks items unanswerable
    because the chain was cut off rather than because the model was wrong. That
    selects for short-reasoning questions and biases everything downstream.
    """
    prompt = build_prompt(question)
    seeds = [seed + j for j in range(k)]

    def one(s):
        return parse_answer(backend.complete(prompt, seed=s, max_tokens=max_tokens))

    if max_workers <= 1:
        answers = [one(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, k)) as pool:
            answers = list(pool.map(one, seeds))
    return sum(a == question.gold for a in answers) / k


def keep_answerable(
    backend, questions: list[Question], *, threshold: float, k: int, seed: int,
    limit: int, max_tokens: int = 8192, max_workers: int = 1
) -> list[Question]:
    """Return up to `limit` questions with correctness_rate >= threshold.

    Candidates are scored in batches so the scoring parallelises, but results are
    consumed strictly in the original order and the scan stops at `limit`. That
    keeps the kept list identical to the serial version: `limit` makes order
    matter, so concurrency must not be allowed to change which questions win.
    """
    if max_workers <= 1:
        kept = []
        for offset, question in enumerate(questions):
            if correctness_rate(backend, question, k=k, seed=seed + offset * k,
                                max_tokens=max_tokens) >= threshold:
                kept.append(question)
            if len(kept) >= limit:
                break
        return kept

    kept, batch = [], max(1, max_workers // k)
    for start in range(0, len(questions), batch):
        chunk = list(enumerate(questions[start:start + batch], start=start))
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            rates = list(pool.map(
                lambda oq: correctness_rate(backend, oq[1], k=k,
                                            seed=seed + oq[0] * k,
                                            max_tokens=max_tokens,
                                            max_workers=k),
                chunk))
        for (_, question), rate in zip(chunk, rates):
            if rate >= threshold:
                kept.append(question)
            if len(kept) >= limit:
                return kept
    return kept
