from mats.backend import DummyBackend
from mats.cot import parse_answer, split_sentences

TRUTH_PROMPT = "Q: pick one. GROUNDTRUTH=(C)"
CUE_PROMPT = TRUTH_PROMPT + "\nHINT: the answer is (A)"


def test_dummy_is_deterministic():
    b = DummyBackend()
    assert b.complete(CUE_PROMPT, seed=1) == b.complete(CUE_PROMPT, seed=1)


def test_dummy_answers_groundtruth_without_cue():
    b = DummyBackend()
    for seed in range(20):
        assert parse_answer(b.complete(TRUTH_PROMPT, seed=seed)) == "C"


def test_dummy_flip_rate_tracks_cue_strength():
    b = DummyBackend(cue_strength=0.7)
    flips = sum(
        parse_answer(b.complete(CUE_PROMPT, seed=seed)) == "A" for seed in range(400)
    )
    assert 0.6 < flips / 400 < 0.8


def test_dummy_cot_never_mentions_the_hint():
    b = DummyBackend()
    for seed in range(20):
        assert "HINT" not in b.complete(CUE_PROMPT, seed=seed).upper()


def test_dummy_cot_has_expected_sentence_count():
    b = DummyBackend(n_steps=5)
    # 5 CoT sentences + 1 final-answer sentence.
    assert len(split_sentences(b.complete(TRUTH_PROMPT, seed=0))) == 6


def test_pivot_must_be_in_range():
    try:
        DummyBackend(n_steps=3, pivot=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range pivot")
