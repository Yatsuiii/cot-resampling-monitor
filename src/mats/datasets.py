"""Row-schema adapters for the multiple-choice corpora the sweep varies over.

One function per corpus, each mapping that corpus's row dict into the shared
`Question`. They are pure and take plain dicts, so the whole schema layer is
testable without the network or the `datasets` package.

Difficulty is the reason more than one corpus exists here. `keep_answerable`
filters to items the subject answers correctly at `correct_threshold`, which on
an easy corpus selects confident items that only a blunt cue can move - and
blunt cues get narrated in the chain of thought. Harder corpora let a subtler
cue flip an answer, which is the regime where unverbalized influence is
plausible at all.
"""

from __future__ import annotations

import hashlib
import random

from .prompts import Question

LETTERS = "ABCDE"


def arc_row_to_question(row: dict) -> Question | None:
    """ARC-Challenge: per-choice labels that may not already be A, B, C, D."""
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if not 2 <= len(labels) <= len(LETTERS):
        return None
    remap = {original: LETTERS[i] for i, original in enumerate(labels)}
    gold = remap.get(row["answerKey"])
    if gold is None:
        return None
    return Question(qid=str(row["id"]), stem=row["question"],
                    options={remap[l]: t for l, t in zip(labels, texts)}, gold=gold)


def mmlu_row_to_question(row: dict, *, index: int) -> Question | None:
    """MMLU: a flat choices list with an integer answer index.

    MMLU rows carry no id, so the qid is synthesised from the subject and row
    index. It must be stable, because `_seed_for(qid)` derives the run seed.
    """
    choices = row["choices"]
    answer = row["answer"]
    if not 2 <= len(choices) <= len(LETTERS):
        return None
    if not isinstance(answer, int) or not 0 <= answer < len(choices):
        return None
    options = {LETTERS[i]: text for i, text in enumerate(choices)}
    subject = row.get("subject", "mmlu")
    return Question(qid=f"mmlu-{subject}-{index}", stem=row["question"],
                    options=options, gold=LETTERS[answer])


def gpqa_row_to_question(row: dict, *, index: int) -> Question | None:
    """GPQA: correct and incorrect answers in SEPARATE named fields.

    The correct answer always arrives in the same field, so presenting them in
    field order would put gold in the same position for every item and confound
    every cue result with position. The permutation is therefore derived from a
    hash of the question text: stable across reruns and across machines, and
    independent of iteration order.
    """
    stem = row.get("Question")
    correct = row.get("Correct Answer")
    wrong = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    if stem is None or correct is None or any(w is None for w in wrong):
        return None

    texts = [correct, *wrong]
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    order = _stable_permutation(len(texts), digest)
    options = {LETTERS[slot]: texts[src] for slot, src in enumerate(order)}
    gold = LETTERS[order.index(0)]          # index 0 is the correct answer
    return Question(qid=f"gpqa-{index}", stem=stem, options=options, gold=gold)


def _stable_permutation(n: int, digest: bytes) -> list[int]:
    """Fisher-Yates driven by digest bytes. Same digest, same permutation."""
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def load_arc_challenge(split: str = "validation") -> list[Question]:
    from datasets import load_dataset
    rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    return [q for q in (arc_row_to_question(r) for r in rows) if q is not None]


def load_mmlu(split: str = "test", subject: str = "all") -> list[Question]:
    from datasets import load_dataset
    rows = load_dataset("cais/mmlu", subject, split=split)
    out = [mmlu_row_to_question(r, index=i) for i, r in enumerate(rows)]
    return [q for q in out if q is not None]


def load_gpqa(split: str = "train", config: str = "gpqa_diamond") -> list[Question]:
    """GPQA is a gated dataset on the Hub; the caller needs an accepted licence
    and a token in the environment. The failure is a permissions error at
    download time, not something this module can work around."""
    from datasets import load_dataset
    rows = load_dataset("Idavidrein/gpqa", config, split=split)
    out = [gpqa_row_to_question(r, index=i) for i, r in enumerate(rows)]
    return [q for q in out if q is not None]


LOADERS = {
    "arc-challenge": load_arc_challenge,
    "mmlu": load_mmlu,
    "gpqa-diamond": load_gpqa,
}


def load(name: str, **kwargs) -> list[Question]:
    """Select a corpus by config string so the sweep needs no imports."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


def sampling_order(questions: list[Question], *, seed: int) -> list[Question]:
    """The corpus reordered so that a prefix of it is a fair sample.

    `keep_answerable` scans in the order it is handed and stops once it has
    enough, so whatever it returns is a prefix. A corpus grouped by topic then
    yields a sample from one topic: the 09-12 grid's whole 40-item MMLU arm
    came from the first 45 questions of abstract_algebra, because `cais/mmlu`
    `all` is ordered by subject.

    Seeded, so the sample is reproducible from the run manifest, and applied to
    every corpus rather than to the one that was caught.
    """
    shuffled = list(questions)
    random.Random(seed).shuffle(shuffled)
    return shuffled
