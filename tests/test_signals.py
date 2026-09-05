import numpy as np

from mats.backend import DummyBackend
from mats.signals import (
    _parse_verdict,
    cue_mentioned,
    diffuse_score,
    entropy_baseline_score,
    llm_monitor_score,
    normalized_entropy,
)


def test_normalized_entropy_range():
    assert normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0
    assert normalized_entropy(np.array([0.25, 0.25, 0.25, 0.25])) == 1.0


def test_diffuse_high_when_unstable_and_spread():
    spread = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    peaked = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    unstable = np.array([0.5, 0.5, 0.0, 0.0])
    # Same instability, but spread importance is not explained by one sentence.
    assert diffuse_score(spread, unstable) > diffuse_score(peaked, unstable)
    assert diffuse_score(spread, unstable) > 0.3


def test_diffuse_low_when_one_pivot_dominates():
    peaked = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    unstable = np.array([0.5, 0.5, 0.0, 0.0])
    assert diffuse_score(peaked, unstable) == 0.0


def test_entropy_baseline_is_zero_for_certain_answer():
    assert entropy_baseline_score(np.array([0.0, 1.0, 0.0, 0.0])) == 0.0


def test_cue_mentioned_detects_reference_words():
    assert cue_mentioned("The hint says it is B, so B.")
    assert not cue_mentioned("Option B is supported by the passage about photosynthesis.")


def test_parse_verdict_ignores_think_block_and_word_boundaries():
    assert _parse_verdict("<think>hmm, I do not KNOW</think>\nYES") is True
    assert _parse_verdict("reasoning... it does NOT matter\nFinal: NO") is False
    assert _parse_verdict("<think>YES probably</think>\nActually, NO.") is False
    assert _parse_verdict("I cannot tell") is None


def test_llm_monitor_score_on_dummy_is_zero():
    # DummyBackend always answers NO to a monitor query.
    score = llm_monitor_score(
        DummyBackend(),
        question_prompt="Q",
        cot_text="some reasoning",
        answer="B",
        seed=0,
        k=5,
    )
    assert score == 0.0
