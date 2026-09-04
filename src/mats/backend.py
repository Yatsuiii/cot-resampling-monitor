"""The model boundary: everything below this line is a text-completion engine.

`Backend` is intentionally tiny - one method, prompt string in, text out,
deterministic given `seed`. Chat templating, CoT parsing, cue injection, and
resampling all live above it, so swapping the Kaggle vLLM server for a stub in
tests touches no analysis code.

`DummyBackend` is that stub: a deterministic fake reasoning model whose final
answer depends on exactly one "pivot" sentence of its CoT. Tests can therefore
assert a known resampling-importance profile (all the weight on the pivot)
without a GPU or network.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

# Matches the cue line the prompt builder injects for the planted-cue condition.
_CUE = re.compile(r"HINT:\s*the answer is \(?([A-J])\)?", re.IGNORECASE)
# Matches the ground-truth marker the dummy prompt carries so the fake model
# "knows" the unbiased answer. Real prompts never contain this.
_TRUTH = re.compile(r"GROUNDTRUTH=\(?([A-J])\)?")


class Backend(Protocol):
    """A deterministic text-completion model."""

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 2048) -> str:
        """Return the model's continuation of `prompt`. Same (prompt, seed) ->
        same text."""
        ...


class DummyBackend:
    """Deterministic stand-in for a reasoning model on multiple-choice prompts.

    The emitted CoT has a fixed shape of `n_steps` sentences. One of them (index
    `pivot`) is the only sentence that determines the final answer:

      - with no cue in the prompt, the pivot resolves to GROUNDTRUTH;
      - with a "HINT: the answer is X" line, the pivot resolves to X on the
        fraction of seeds set by `cue_strength` (a controllable flip rate), and
        never mentions the hint (so a CoT-reading monitor cannot catch it).

    Every other sentence is filler whose resampled variants do not move the
    answer, giving tests a clean importance profile.
    """

    def __init__(
        self,
        *,
        n_steps: int = 5,
        pivot: int = 2,
        cue_strength: float = 0.7,
    ) -> None:
        if not 0 <= pivot < n_steps:
            raise ValueError("pivot must index a sentence in the CoT")
        self.n_steps = n_steps
        self.pivot = pivot
        self.cue_strength = cue_strength

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 2048) -> str:
        truth = _first_group(_TRUTH, prompt) or "A"
        cue = _first_group(_CUE, prompt)
        answer = self._resolve_answer(prompt, truth, cue, seed)
        steps = self._steps_before(prompt, seed)
        steps.insert(self.pivot, f"Working through the options, it comes out to {answer}.")
        return "\n".join(steps) + f"\nThe answer is ({answer})."

    def _resolve_answer(
        self, prompt: str, truth: str, cue: str | None, seed: int
    ) -> str:
        if cue is None:
            return truth
        # Deterministic per (prompt, seed): a stable fraction of seeds flip.
        draw = _unit_hash(f"{prompt}|{seed}|flip")
        return cue if draw < self.cue_strength else truth

    def _steps_before(self, prompt: str, seed: int) -> list[str]:
        """Filler sentences. Their content varies with seed so the semantic
        dedup filter has real variation to work on, but none of it is
        answer-bearing."""
        n_filler = self.n_steps - 1
        return [
            f"Consideration {i}: weighing detail {_unit_hash(f'{prompt}|{seed}|{i}'):.3f}."
            for i in range(n_filler)
        ]


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).upper() if match else None


def _unit_hash(key: str) -> float:
    """A stable pseudo-random float in [0, 1) from a string key."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
