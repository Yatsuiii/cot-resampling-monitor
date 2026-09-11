from mats.backend import Completion, DummyBackend
from mats.cot import split_sentences
from mats.prompts import Question, build_prompt
from mats.repair import (
    ConditionRates,
    QuestionResult,
    dose_response,
    gate_report,
    halfway_prefix,
    run_question,
)

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


def test_rates_do_not_treat_a_truncated_tentative_answer_as_final():
    rates = ConditionRates.of(
        _completions(["A"], truncated={0}), cue_letter="A", gold="C"
    )
    assert rates.cue_rate == 0.0
    assert rates.gold_rate == 0.0
    assert rates.unusable == 1.0
    assert rates.truncated == 1.0


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


def _dummy_result(qid, cue_effect, residual):
    """A QuestionResult with the two derived quantities set to known values."""
    clean = ConditionRates(8, cue_rate=0.0, gold_rate=1.0, unusable=0.0, truncated=0.0)
    cued = ConditionRates(8, cue_rate=cue_effect, gold_rate=0.0, unusable=0.0, truncated=0.0)
    removal = ConditionRates(8, cue_rate=residual, gold_rate=0.0, unusable=0.0, truncated=0.0)
    return QuestionResult(
        qid=qid, gold="C", cue_letter="A", donor_prefix="", donor_text="",
        rates={"clean": clean, "cued": cued, "source_removal": removal,
               "explicit_correction": removal},
    )


# run_question builds real prompts, which carry no GROUNDTRUTH marker, so
# DummyBackend falls back to answering (A). Give it a question whose gold is A
# and cue toward a different letter.
QA = Question("qa", "Why?", {"A": "a", "B": "b", "C": "c", "D": "d"}, gold="A")


def test_run_question_covers_all_four_policies_without_a_gpu():
    backend = DummyBackend(n_steps=6, pivot=3, cue_strength=0.7)
    result = run_question(backend, QA, cue_letter="B", n_gen=4, seed=11)
    assert set(result.rates) == {"clean", "cued", "source_removal", "explicit_correction"}
    assert all(r.n == 4 for r in result.rates.values())
    # no cue -> the dummy answers gold every time, so the cue has to move it
    assert result.rates["clean"].gold_rate == 1.0
    assert result.rates["clean"].cue_rate == 0.0
    assert result.cue_effect > 0
    # the donor is a real prefix taken from the first cued generation
    assert result.donor_prefix and result.donor_prefix in result.donor_text
    assert set(result.completions) == set(result.rates)
    assert all(len(rows) == 4 for rows in result.completions.values())


def test_run_question_donor_is_the_first_cued_trace_not_a_flipped_one():
    backend = DummyBackend(n_steps=6, pivot=3, cue_strength=0.7)
    result = run_question(backend, QA, cue_letter="B", n_gen=4, seed=11)
    first_cued = backend.complete(
        build_prompt(QA, cue_letter="B"), seed=11 + 100, max_tokens=1200
    )
    assert result.donor_text == first_cued


def test_dose_response_recovers_a_planted_correlation():
    results = [
        _dummy_result(f"q{i}", cue_effect=e, residual=r)
        for i, (e, r) in enumerate(
            [(0.1, 0.0), (0.2, 0.0), (0.4, 0.13), (0.6, 0.25), (0.8, 0.5), (1.0, 0.75)]
        )
    ]
    out = dose_response(results, draws=2000)
    assert out["n_questions"] == 6
    assert out["r"] > 0.9
    assert out["excludes_zero"]
    assert out["fully_repaired"] == 2


def test_dose_response_reports_inconclusive_when_there_is_no_signal():
    results = [
        _dummy_result(f"q{i}", cue_effect=e, residual=r)
        for i, (e, r) in enumerate(
            [(0.2, 0.5), (0.4, 0.0), (0.6, 0.5), (0.8, 0.0), (1.0, 0.25), (0.3, 0.25)]
        )
    ]
    out = dose_response(results, draws=2000)
    assert not out["excludes_zero"]


def test_dose_response_handles_a_flat_variable():
    # every question fully repaired: residual has no spread, r is undefined
    results = [_dummy_result(f"q{i}", cue_effect=0.1 * i, residual=0.0) for i in range(5)]
    out = dose_response(results, draws=500)
    assert out["r"] is None
    assert out["fully_repaired"] == 5


def test_rates_record_mean_tokens_for_setting_the_next_cap():
    rows = _completions(["A", "C"])
    rows = [
        Completion(text=r.text, finish_reason=r.finish_reason, completion_tokens=n)
        for r, n in zip(rows, (100, 300))
    ]
    assert ConditionRates.of(rows, cue_letter="A", gold="C").mean_tokens == 200.0
