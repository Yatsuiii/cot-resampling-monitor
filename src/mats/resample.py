"""Sentence-level resampling importance.

For each sentence of a chain-of-thought we regenerate the model's continuation
from just before that sentence, then split the rollouts by whether the
regenerated sentence is semantically the *same* as the original or *different*
from it. Importance is how far the final-answer distribution moves between those
two groups:

    importance(i) = TV( P(answer | sentence i resampled different),
                        P(answer | sentence i resampled same) )

This "different vs same at a fixed position" contrast (following Thought
Branches) localizes a decision point. Comparing instead against the whole-CoT
baseline only measures aggregate stochasticity - it cannot tell you *where* the
answer was decided, because resampling from just before a pivot leaves the answer
as open as it was at the start.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass

import numpy as np

from mats.cot import parse_answer, split_sentences
from mats.embed import Embedder, cosine_matrix

MIN_KEPT = 5  # per group; below this the group's answer distribution is too thin


@dataclass(frozen=True)
class Rollout:
    answer: str | None
    first_sentence: str


@dataclass(frozen=True)
class SentenceResult:
    index: int
    importance: float
    flip_rate: float       # P(answer != base answer) in the "different" group
    n_same: int
    n_different: int

    @property
    def reliable(self) -> bool:
        return self.n_same >= MIN_KEPT and self.n_different >= MIN_KEPT


@dataclass(frozen=True)
class CotAnalysis:
    base_answer: str | None
    baseline_dist: np.ndarray
    letters: tuple[str, ...]
    sentences: tuple[str, ...]
    per_sentence: tuple[SentenceResult, ...]

    @property
    def importance(self) -> np.ndarray:
        return np.array([s.importance for s in self.per_sentence])


def answer_distribution(answers, letters) -> np.ndarray:
    """Normalised histogram of parsed answers over `letters`. Unparseable or
    out-of-set answers drop out of the denominator."""
    index = {letter: i for i, letter in enumerate(letters)}
    counts = np.zeros(len(letters))
    for answer in answers:
        if answer in index:
            counts[index[answer]] += 1.0
    total = counts.sum()
    return counts / total if total else counts


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _one_rollout(backend, prompt, prefix, seed, max_tokens) -> Rollout:
    text = backend.complete(prompt, prefix=prefix, seed=seed, max_tokens=max_tokens)
    sentences = split_sentences(text)
    return Rollout(parse_answer(text), sentences[0] if sentences else "")


def _rollouts_from(backend, prompt, prefix, k, seed0, max_tokens, max_workers=1) -> list[Rollout]:
    """`prefix` is the CoT written so far; each rollout is the model's
    continuation. The first sentence of the continuation is the regenerated
    version of the sentence we are probing.

    `max_workers` > 1 issues the k requests concurrently via a thread pool -
    each is an independent HTTP call to the model server, which batches
    concurrent requests far more efficiently than one at a time. Rollout order
    doesn't matter (we only ever aggregate into a distribution), so this is
    safe with no ordering guarantee.
    """
    if max_workers <= 1:
        return [
            _one_rollout(backend, prompt, prefix, seed0 + j, max_tokens) for j in range(k)
        ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_one_rollout, backend, prompt, prefix, seed0 + j, max_tokens)
            for j in range(k)
        ]
        return [f.result() for f in futures]


def _split_by_similarity(rollouts, original, embedder: Embedder, cosine_max: float):
    variants = [r.first_sentence for r in rollouts]
    sims = cosine_matrix(embedder.encode(variants), embedder.encode([original]))[:, 0]
    same = [r for r, sim in zip(rollouts, sims) if sim > cosine_max]
    different = [r for r, sim in zip(rollouts, sims) if sim <= cosine_max]
    return same, different


def sentence_importance(
    backend,
    embedder: Embedder,
    *,
    base_prompt: str,
    sentences: list[str],
    index: int,
    base_answer: str | None,
    letters: tuple[str, ...],
    k: int,
    cosine_max: float,
    seed0: int,
    max_tokens: int,
    max_workers: int = 1,
) -> SentenceResult:
    prefix = "\n".join(sentences[:index])
    rollouts = _rollouts_from(backend, base_prompt, prefix, k, seed0, max_tokens, max_workers)
    same, different = _split_by_similarity(rollouts, sentences[index], embedder, cosine_max)
    if len(same) < MIN_KEPT or len(different) < MIN_KEPT:
        return SentenceResult(index, 0.0, 0.0, len(same), len(different))
    dist_same = answer_distribution((r.answer for r in same), letters)
    dist_different = answer_distribution((r.answer for r in different), letters)
    flips = np.mean([r.answer != base_answer for r in different]) if base_answer else 0.0
    return SentenceResult(
        index,
        total_variation(dist_different, dist_same),
        float(flips),
        len(same),
        len(different),
    )


def evenly_spaced_indices(n_sentences: int, n_positions: int) -> list[int]:
    """`n_positions` sentence indices spread across [0, n_sentences), for
    bounding resampling cost on a long CoT. Returns every index if
    `n_positions` already covers them all."""
    if n_positions >= n_sentences:
        return list(range(n_sentences))
    if n_positions <= 1:
        return [0]
    step = (n_sentences - 1) / (n_positions - 1)
    return sorted({round(i * step) for i in range(n_positions)})


def analyse_cot(
    backend,
    embedder: Embedder,
    *,
    base_prompt: str,
    base_cot: str,
    letters: tuple[str, ...],
    k_sentence: int,
    k_baseline: int,
    cosine_max: float,
    seed: int,
    max_tokens: int = 1024,
    indices: list[int] | None = None,
    max_workers: int = 1,
) -> CotAnalysis:
    """`indices` restricts resampling to those sentence positions (default: all).
    Useful to bound cost on a long CoT - e.g. a handful of evenly-spaced
    positions for a cheap feasibility check before committing to the full run.
    `per_sentence` then holds one `SentenceResult` per selected index, each still
    tagged with its true position.

    `max_workers` issues each position's k rollouts concurrently (a real model
    server batches concurrent requests far more efficiently than one at a
    time); positions themselves are still processed one after another.
    """
    sentences = split_sentences(base_cot)
    positions = range(len(sentences)) if indices is None else indices
    baseline = _rollouts_from(
        backend, base_prompt, "", k_baseline, seed * 1_000_000, max_tokens, max_workers
    )
    baseline_dist = answer_distribution((r.answer for r in baseline), letters)
    base_answer = parse_answer(base_cot)
    results = tuple(
        sentence_importance(
            backend,
            embedder,
            base_prompt=base_prompt,
            sentences=sentences,
            index=i,
            base_answer=base_answer,
            letters=letters,
            k=k_sentence,
            cosine_max=cosine_max,
            seed0=seed * 1_000_000 + (i + 1) * 1000,
            max_tokens=max_tokens,
            max_workers=max_workers,
        )
        for i in positions
    )
    return CotAnalysis(base_answer, baseline_dist, letters, tuple(sentences), results)


@dataclass(frozen=True)
class CapitulationCurve:
    """How often a resampled continuation lands on `cue_letter`, as a function
    of how much of the chain-of-thought is kept fixed. On a CoT that was flipped
    by a cue, this rises from near 0 (resample early -> model re-derives its own
    answer) to near 1 (resample late -> already committed to the cue). Where it
    rises is where the model gave in."""

    cue_letter: str
    indices: tuple[int, ...]      # sentence positions probed (0 = whole CoT resampled)
    p_cue: np.ndarray            # P(final answer == cue_letter) at each probed position
    sentences: tuple[str, ...]   # the full CoT, for reading off the capitulation sentence

    @property
    def capitulation_index(self) -> int | None:
        """Probed position just after the largest single rise in `p_cue` - the
        step across which resampling stops recovering the pre-cue answer. None
        if `p_cue` never gets above 0.5 (no clear capitulation)."""
        if len(self.p_cue) < 2 or self.p_cue.max() < 0.5:
            return None
        return int(self.indices[int(np.argmax(np.diff(self.p_cue))) + 1])


def capitulation_curve(
    backend,
    *,
    base_prompt: str,
    cued_cot: str,
    cue_letter: str,
    letters: tuple[str, ...],
    k: int,
    seed: int,
    max_tokens: int = 1024,
    max_workers: int = 1,
    stride: int = 1,
) -> CapitulationCurve:
    """Resample the continuation from every `stride`-th sentence position and
    record P(final answer == `cue_letter`). No dedup filter and no same/different
    split - this asks only "which answer comes out", so `k` can be small."""
    sentences = split_sentences(cued_cot)
    cue_index = letters.index(cue_letter)
    probed = list(range(0, len(sentences), stride))
    p_cue = []
    for i in probed:
        prefix = "\n".join(sentences[:i])
        rollouts = _rollouts_from(
            backend, base_prompt, prefix, k, seed + i * 1000, max_tokens, max_workers
        )
        dist = answer_distribution((r.answer for r in rollouts), letters)
        p_cue.append(float(dist[cue_index]))
    return CapitulationCurve(cue_letter, tuple(probed), np.array(p_cue), tuple(sentences))
