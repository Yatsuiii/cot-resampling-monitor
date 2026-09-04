from mats.data import correctness_rate, keep_answerable
from mats.prompts import Question

EASY = Question("e1", "Which?", {"A": "x", "B": "y", "C": "z", "D": "w"}, gold="A")


class _AlwaysGold:
    """No cue in the prompt -> DummyBackend-style: always answers the marker
    letter. Here we hardcode gold to keep the test independent of DummyBackend."""

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 2048) -> str:
        return "Reasoning about it.\nThe answer is (A)."


def test_correctness_rate_all_correct():
    assert correctness_rate(_AlwaysGold(), EASY, k=8, seed=0) == 1.0


def test_keep_answerable_respects_limit_and_threshold():
    pool = [Question(f"q{i}", "?", {"A": "a", "B": "b"}, gold="A") for i in range(10)]
    kept = keep_answerable(_AlwaysGold(), pool, threshold=0.9, k=4, seed=0, limit=3)
    assert len(kept) == 3


def test_keep_answerable_drops_below_threshold():
    hard = [Question(f"h{i}", "?", {"A": "a", "B": "b"}, gold="B") for i in range(5)]
    kept = keep_answerable(_AlwaysGold(), hard, threshold=0.5, k=4, seed=0, limit=5)
    assert kept == []
