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


def test_continuation_resumes_after_prefix():
    b = DummyBackend(n_steps=5, pivot=2)
    # Prefix already contains fillers 0 and 1; the continuation must start at the
    # pivot sentence, not re-emit filler 0.
    prefix = (
        "Consideration 0: this detail is weighed carefully at step 0.\n"
        "Consideration 1: this detail is weighed carefully at step 1."
    )
    out = b.complete(TRUTH_PROMPT, prefix=prefix, seed=3)
    first = split_sentences(out)[0]
    assert first.startswith("The third choice")  # pivot for answer C


def test_answer_locks_to_pivot_already_in_prefix():
    b = DummyBackend(n_steps=5, pivot=2)
    prefix = "The first choice is the one the evidence above points to."
    # Pivot names A; even with a cue for a different letter the answer stays A.
    assert parse_answer(b.complete(CUE_PROMPT, prefix=prefix, seed=9)) == "A"


def test_pivot_must_be_in_range():
    try:
        DummyBackend(n_steps=3, pivot=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range pivot")


def test_the_request_deadline_covers_the_tokens_it_asked_for():
    """Three MMLU cells of the 09-12 grid died on a client-side TimeoutError
    while the server was healthy and returning 200s.

    The deadline was a fixed 600s. A chain running to the 8,192-token cap at
    the ~15 tok/s per stream that 16-way concurrency leaves needs about 546s,
    so the deadline sat inside the noise band of normal operation. It has to be
    a function of the tokens requested, not a constant.
    """
    from mats.backend_vllm import VLLMBackend

    b = VLLMBackend(model="m")
    slowest_measured_rate = 14.0          # tok/s, one long chain draining
    assert b._deadline(8192) > 8192 / slowest_measured_rate
    assert b._deadline(8192) > b._deadline(1024), "deadline ignores the request size"


def test_a_short_request_is_not_given_a_long_chain_deadline():
    """The LLM monitor asks for 1,024 tokens. One constant cannot be right for
    both it and an 8,192-token chain, which is why the constant was wrong."""
    from mats.backend_vllm import VLLMBackend

    b = VLLMBackend(model="m")
    assert b._deadline(1024) < b._deadline(8192) / 4
