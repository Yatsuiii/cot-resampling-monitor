"""The model boundary: everything below this line is a text-completion engine.

`Backend` is intentionally tiny - one method, prompt string in, text out,
deterministic given `seed`. Chat templating, CoT parsing, cue injection, and
resampling all live above it, so swapping the Kaggle vLLM server for a stub in
tests touches no analysis code.

`DummyBackend` is that stub: a deterministic fake reasoning model. Its CoT is a
fixed run of `n_steps` sentences, one of which (index `pivot`) states the
conclusion. It *continues from the prompt*: it counts how many CoT sentences are
already there and emits only the rest. If the pivot is among those already
present the answer is locked to the letter it names; otherwise the model draws a
fresh answer - biased toward the cue letter when a cue is present - and writes
the pivot naming it. So resampling any sentence other than the pivot cannot move
the answer, giving tests a known importance profile.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

# The natural-language cue the prompt builder injects ("The answer key ... says
# the answer is (X)"). Matched here because the backend only ever sees the
# prompt string.
_CUE = re.compile(r"the answer is \(([A-J])\)", re.IGNORECASE)
# Ground-truth marker that only the dummy reads; real prompts never carry it.
_TRUTH = re.compile(r"GROUNDTRUTH=\(?([A-J])\)?")

# Lexically distinct pivot sentences, one per letter, so the semantic-dedup
# filter can tell "concluded A" from "concluded C".
_PIVOTS = {
    "A": "The first choice is the one the evidence above points to.",
    "B": "The second choice best fits everything considered so far.",
    "C": "The third choice is what the key facts actually support.",
    "D": "The fourth choice follows once the others are ruled out.",
    "E": "The fifth choice is the consistent reading here.",
}
_PIVOT_LETTER = {sentence: letter for letter, sentence in _PIVOTS.items()}


class Backend(Protocol):
    """A deterministic text-completion model.

    `prefix` is an assistant partial (chain-of-thought written so far). The
    resampling method regenerates from an arbitrary sentence boundary, so it
    passes the reasoning up to that point as `prefix` and expects only the
    *continuation* back, not the prefix echoed. Same (prompt, prefix, seed) ->
    same text.
    """

    def complete(
        self, prompt: str, *, prefix: str = "", seed: int, max_tokens: int = 2048
    ) -> str:
        ...


class DummyBackend:
    """Deterministic stand-in for a reasoning model on multiple-choice prompts."""

    def __init__(
        self, *, n_steps: int = 5, pivot: int = 2, cue_strength: float = 0.7
    ) -> None:
        if not 0 <= pivot < n_steps:
            raise ValueError("pivot must index a sentence in the CoT")
        self.n_steps = n_steps
        self.pivot = pivot
        self.cue_strength = cue_strength

    def complete(
        self, prompt: str, *, prefix: str = "", seed: int, max_tokens: int = 2048
    ) -> str:
        if prompt.rstrip().endswith("Verdict:"):
            # LLM-monitor query. The dummy's cue never surfaces in its CoT, so a
            # faithful monitor says NO - the failure mode this project studies.
            return "NO"
        emitted = self._count_emitted(prefix)
        answer = self._locked_answer(prefix) or self._fresh_answer(prompt, seed)
        slots = [
            _PIVOTS[answer] if j == self.pivot else _filler(j)
            for j in range(emitted, self.n_steps)
        ]
        body = "\n".join(slots)
        return (body + "\n" if body else "") + f"The answer is ({answer})."

    def _fresh_answer(self, prompt: str, seed: int) -> str:
        truth = _first_group(_TRUTH, prompt) or "A"
        cue = _first_group(_CUE, prompt)
        if cue is None:
            return truth
        return cue if _unit_hash(f"{prompt}|{seed}|flip") < self.cue_strength else truth

    @staticmethod
    def _count_emitted(prefix: str) -> int:
        lines = (line.strip() for line in prefix.splitlines())
        return sum(
            1 for line in lines if line.startswith("Consideration ") or line in _PIVOT_LETTER
        )

    @staticmethod
    def _locked_answer(prefix: str) -> str | None:
        for sentence, letter in _PIVOT_LETTER.items():
            if sentence in prefix:
                return letter
        return None


def _filler(index: int) -> str:
    return f"Consideration {index}: this detail is weighed carefully at step {index}."


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).upper() if match else None


def _unit_hash(key: str) -> float:
    """A stable pseudo-random float in [0, 1) from a string key."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
