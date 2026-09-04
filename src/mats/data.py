"""Load the base multiple-choice questions and keep only the ones the subject
model answers correctly without a cue (so a planted cue has room to flip them).
"""

from __future__ import annotations

from mats.cot import parse_answer
from mats.prompts import Question, build_prompt

_LETTERS = "ABCDE"


def load_arc_challenge(split: str = "validation") -> list[Question]:
    """ARC-Challenge as `Question`s. `datasets` is imported lazily (Kaggle only)."""
    from datasets import load_dataset

    rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    questions: list[Question] = []
    for row in rows:
        question = _row_to_question(row)
        if question is not None:
            questions.append(question)
    return questions


def _row_to_question(row: dict) -> Question | None:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    if not 2 <= len(labels) <= len(_LETTERS):
        return None
    remap = {original: _LETTERS[i] for i, original in enumerate(labels)}
    gold = remap.get(row["answerKey"])
    if gold is None:
        return None
    options = {remap[label]: text for label, text in zip(labels, texts)}
    return Question(qid=str(row["id"]), stem=row["question"], options=options, gold=gold)


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
