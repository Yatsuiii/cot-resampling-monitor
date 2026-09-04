"""Turn a raw chain-of-thought string into the units the monitor works on.

Two jobs, both purely textual:
  - `split_sentences`: chunk the CoT into the sentences we resample from.
  - `parse_answer`: recover the model's final multiple-choice letter.

The resampling method is sentence-level, so the split defines the granularity of
every downstream importance score. We keep it deliberately simple and inspectable
rather than pulling in an NLP tokenizer: a reader can predict exactly where the
cuts land, which matters when a sentence's importance is a headline number.
"""

from __future__ import annotations

import re

# A boundary is . ? or ! followed by whitespace and then a capital/digit/quote.
# Requiring whitespace after the period already protects decimals ("9.11" has no
# space after the dot), so the only false boundary left is an abbreviation
# followed by a capitalised word ("e.g. This ...").
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Ms.", "Fig.")
_BOUNDARY = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9(\"'\u201c])")


def split_sentences(text: str) -> list[str]:
    """Split a CoT into trimmed, non-empty sentences.

    Newlines are treated as hard boundaries (reasoning models lean on them for
    structure); within a line we split on sentence punctuation.
    """
    sentences: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(_split_line(line))
    return [s for s in (s.strip() for s in sentences) if s]


def _split_line(line: str) -> list[str]:
    pieces = _BOUNDARY.split(line)
    merged: list[str] = []
    for piece in pieces:
        if merged and _should_merge(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _should_merge(prev: str) -> bool:
    """True when `prev` ended on a false boundary (a known abbreviation)."""
    tail = prev.rstrip()
    return any(tail.endswith(abbr) for abbr in _ABBREVIATIONS)


# Ordered most- to least-explicit. The first hit wins, so an explicit
# "answer is (B)" beats a stray capital letter earlier in the text.
_ANSWER_PATTERNS = (
    re.compile(r"\\boxed\{\s*([A-J])\s*\}"),
    re.compile(r"answer\s*(?:is|:)\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])\)?\s*$", re.MULTILINE),
)


def parse_answer(text: str) -> str | None:
    """Recover the final choice letter from a completion, or None if absent.

    When a pattern matches several times we take the last occurrence: the model's
    final statement is its answer, and earlier mentions are deliberation.
    """
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None
