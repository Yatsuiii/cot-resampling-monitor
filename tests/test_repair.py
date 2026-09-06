from mats.backend import Completion, DummyBackend
from mats.cot import split_sentences
from mats.prompts import Question, build_prompt
from mats.repair import ConditionRates, gate_report, halfway_prefix

Q = Question("q1", "Why?", {"A": "a", "B": "b", "C": "c", "D": "d"}, gold="C")


def _completions(answers, truncated=()):
    return [
        Completion(
            text=f"reasoning here.\nThe answer is ({a})." if a else "reasoning, no verdict",
            finish_reason="length" if i in truncated else "stop",
            completion_tokens=10,
        )
        for i, a in enumerate(answers)
    ]


def test_halfway_prefix_takes_first_half_of_sentences():
    cot = "One.\nTwo.\nThree.\nFour."
    assert halfway_prefix(cot) == "One.\nTwo."


def test_halfway_prefix_of_single_sentence_is_empty():
    assert halfway_prefix("Only one sentence here.") == ""


def test_halfway_prefix_cuts_on_a_sentence_boundary():
    backend = DummyBackend(n_steps=6, pivot=3)
    cot = backend.complete(build_prompt(Q, groundtruth_marker=True), seed=3)
    prefix = halfway_prefix(cot)
    # every prefix line is a whole sentence of the original
    assert all(line in split_sentences(cot) for line in prefix.splitlines())


def test_rates_count_cue_gold_and_unusable():
    rates = ConditionRates.of(
        _completions(["A", "A", "C", None]), cue_letter="A", gold="C"
    )
    assert rates.n == 4
    assert rates.cue_rate == 0.5
    assert rates.gold_rate == 0.25
    assert rates.unusable == 0.25


def test_rates_separate_truncation_from_refusal():
    rates = ConditionRates.of(
        _completions([None, None], truncated={0}), cue_letter="A", gold="C"
    )
    assert rates.unusable == 1.0
    assert rates.truncated == 0.5


def test_rates_of_nothing_is_all_zero():
    assert ConditionRates.of([], cue_letter="A", gold="C").n == 0


def test_gate_passes_when_all_thresholds_met():
    clean = ConditionRates(20, cue_rate=0.05, gold_rate=0.90, unusable=0.0, truncated=0.0)
    cued = ConditionRates(20, cue_rate=0.70, gold_rate=0.25, unusable=0.0, truncated=0.0)
    transplant = ConditionRates(20, cue_rate=0.40, gold_rate=0.50, unusable=0.0, truncated=0.0)
    report = gate_report(clean, cued, transplant)
    assert abs(report["cue_effect"] - 0.65) < 1e-9
    assert abs(report["residual_pull"] - 0.35) < 1e-9
    assert report["passed"]


def test_gate_fails_when_the_prefix_stops_pulling():
    # The premise-killing case: removing the cue from the prompt fully repairs
    # the answer, so there is nothing left in the scratchpad to delete.
    clean = ConditionRates(20, cue_rate=0.05, gold_rate=0.90, unusable=0.0, truncated=0.0)
    cued = ConditionRates(20, cue_rate=0.70, gold_rate=0.25, unusable=0.0, truncated=0.0)
    transplant = ConditionRates(20, cue_rate=0.08, gold_rate=0.88, unusable=0.0, truncated=0.0)
    report = gate_report(clean, cued, transplant)
    assert not report["passed"]
    assert report["checks"]["residual_pull>=0.15"] is False
    assert report["checks"]["cue_effect>=0.25"] is True


def test_gate_fails_on_unusable_generations():
    rates = ConditionRates(20, cue_rate=0.70, gold_rate=0.20, unusable=0.20, truncated=0.20)
    clean = ConditionRates(20, cue_rate=0.05, gold_rate=0.90, unusable=0.0, truncated=0.0)
    report = gate_report(clean, rates, rates)
    assert report["checks"]["unusable<0.05"] is False
    assert not report["passed"]
