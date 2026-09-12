from mats.backend import DummyBackend
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


def _pool(n):
    from mats.prompts import Question
    return [Question(qid=f"q{i}", stem=f"Q{i}?",
                     options={"A": "a", "B": "b", "C": "c", "D": "d"},
                     gold="ABCD"[i % 4]) for i in range(n)]


class _Recorder(DummyBackend):
    """Captures the kwargs complete() was called with."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def complete(self, prompt, **kw):
        self.calls.append(kw)
        return super().complete(prompt, **kw)


def test_parallel_filter_keeps_exactly_the_same_questions():
    """H31. `limit` makes order matter, so concurrency must not change which
    questions win the scan."""
    from mats.data import keep_answerable
    qs = _pool(40)
    serial = keep_answerable(_Recorder(), qs, threshold=0.5, k=4, seed=7,
                             limit=9, max_workers=1)
    parallel = keep_answerable(_Recorder(), qs, threshold=0.5, k=4, seed=7,
                               limit=9, max_workers=16)
    assert [q.qid for q in serial] == [q.qid for q in parallel]


def test_parallel_filter_matches_serial_when_limit_is_never_reached():
    from mats.data import keep_answerable
    qs = _pool(12)
    a = keep_answerable(_Recorder(), qs, threshold=0.99, k=4, seed=3,
                        limit=999, max_workers=1)
    b = keep_answerable(_Recorder(), qs, threshold=0.99, k=4, seed=3,
                        limit=999, max_workers=8)
    assert [q.qid for q in a] == [q.qid for q in b]


def test_filter_generates_at_the_experiment_token_cap():
    """H32. The complete() default is 1024 and chains run well past it; a filter
    left on the default truncates its own rollouts and marks items unanswerable
    for the wrong reason."""
    from mats.data import keep_answerable
    rec = _Recorder()
    keep_answerable(rec, _pool(6), threshold=0.5, k=2, seed=1, limit=3,
                    max_tokens=8192, max_workers=4)
    assert rec.calls, "filter never called the backend"
    assert all(c.get("max_tokens") == 8192 for c in rec.calls), \
        sorted({c.get("max_tokens") for c in rec.calls})
