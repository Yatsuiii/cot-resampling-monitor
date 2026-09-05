"""Resampling importance on a known fixture: the DummyBackend answer is decided
by exactly one pivot sentence, so importance must concentrate there."""

import numpy as np

from mats.backend import DummyBackend
from mats.cot import split_sentences
from mats.embed import HashEmbedder
from mats.prompts import Question, build_prompt
from mats.resample import (
    analyse_cot,
    answer_distribution,
    capitulation_curve,
    evenly_spaced_indices,
    total_variation,
)

QUESTION = Question(
    qid="fix-1",
    stem="Which option?",
    options={"A": "a", "B": "b", "C": "c", "D": "d"},
    gold="C",
)
LETTERS = ("A", "B", "C", "D")


def _analysis(cue_letter):
    backend = DummyBackend(n_steps=5, pivot=2, cue_strength=0.7)
    prompt = build_prompt(QUESTION, cue_letter=cue_letter, groundtruth_marker=True)
    cot = backend.complete(prompt, seed=7)
    return backend, analyse_cot(
        backend, HashEmbedder(), base_prompt=prompt, base_cot=cot, letters=LETTERS,
        k_sentence=60, k_baseline=40, cosine_max=0.85, seed=7,
    )


def test_total_variation_bounds():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert total_variation(p, p) == 0.0
    assert total_variation(p, q) == 1.0


def test_answer_distribution_ignores_unparseable():
    # None drops out of the denominator: 2 A + 1 B over 3 parsed answers.
    dist = answer_distribution(["A", "A", None, "B"], LETTERS)
    assert dist[0] == 2 / 3 and dist[1] == 1 / 3 and dist[2] == 0.0


def test_importance_concentrates_on_pivot_for_cued_item():
    _, analysis = _analysis(cue_letter="A")
    importance = analysis.importance
    pivot = int(np.argmax(importance))
    # pivot sentence index is 2 (0-indexed) in a 5-step CoT
    assert pivot == 2
    assert importance[2] > 0.5
    assert importance.sum() - importance[2] < 0.2  # everything else near zero


def test_control_item_has_stable_answer():
    _, analysis = _analysis(cue_letter=None)
    # No cue -> dummy always answers gold -> baseline distribution is one-hot.
    assert analysis.baseline_dist.max() == 1.0
    assert analysis.base_answer == "C"


def test_analysis_has_one_result_per_sentence():
    backend, analysis = _analysis(cue_letter="A")
    prompt = build_prompt(QUESTION, cue_letter="A", groundtruth_marker=True)
    n_sentences = len(split_sentences(backend.complete(prompt, seed=7)))
    assert len(analysis.per_sentence) == n_sentences


def test_evenly_spaced_indices_covers_endpoints_and_dedups():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(5, 1) == [0]
    assert evenly_spaced_indices(5, 100) == [0, 1, 2, 3, 4]


def test_max_workers_matches_sequential_result():
    backend = DummyBackend(n_steps=5, pivot=2, cue_strength=0.7)
    prompt = build_prompt(QUESTION, cue_letter="A", groundtruth_marker=True)
    cot = backend.complete(prompt, seed=7)
    kwargs = dict(
        backend=backend, embedder=HashEmbedder(), base_prompt=prompt, base_cot=cot,
        letters=LETTERS, k_sentence=60, k_baseline=40, cosine_max=0.85, seed=7,
        indices=[2],
    )
    sequential = analyse_cot(**kwargs, max_workers=1)
    concurrent = analyse_cot(**kwargs, max_workers=8)
    assert sequential.per_sentence[0].importance == concurrent.per_sentence[0].importance


def test_capitulation_curve_rises_at_the_pivot():
    # DummyBackend: before the pivot, resampling redraws (P(cue) ~ cue_strength);
    # once the pivot is in the prefix, the answer is locked to it.
    backend = DummyBackend(n_steps=6, pivot=3, cue_strength=0.7)
    prompt = build_prompt(QUESTION, cue_letter="A", groundtruth_marker=True)
    cot = backend.complete(prompt, seed=7)
    curve = capitulation_curve(
        backend, base_prompt=prompt, cued_cot=cot, cue_letter="A", letters=LETTERS,
        k=40, seed=1, stride=1,
    )
    assert curve.p_cue[0] < 0.9          # early: not yet committed
    assert curve.p_cue[-1] == 1.0        # late: pivot fixed, locked to cue
    assert curve.capitulation_index is not None
    assert 2 <= curve.capitulation_index <= 5   # pivot sentence sits at index 3


def test_capitulation_index_none_when_never_crosses():
    backend = DummyBackend(n_steps=5, pivot=2, cue_strength=0.7)
    prompt = build_prompt(QUESTION, groundtruth_marker=True)  # no cue
    cot = backend.complete(prompt, seed=7)
    curve = capitulation_curve(
        backend, base_prompt=prompt, cued_cot=cot, cue_letter="A", letters=LETTERS,
        k=20, seed=1,
    )
    # No cue -> dummy always answers gold (C), never A -> p_cue stays 0.
    assert curve.p_cue.max() == 0.0
    assert curve.capitulation_index is None


def test_indices_restricts_to_selected_positions():
    backend = DummyBackend(n_steps=5, pivot=2, cue_strength=0.7)
    prompt = build_prompt(QUESTION, cue_letter="A", groundtruth_marker=True)
    cot = backend.complete(prompt, seed=7)
    analysis = analyse_cot(
        backend, HashEmbedder(), base_prompt=prompt, base_cot=cot, letters=LETTERS,
        k_sentence=60, k_baseline=40, cosine_max=0.85, seed=7, indices=[0, 2],
    )
    assert [r.index for r in analysis.per_sentence] == [0, 2]
    # The pivot (index 2) is still found even though most sentences were skipped.
    pivot_result = analysis.per_sentence[1]
    assert pivot_result.index == 2 and pivot_result.importance > 0.5
