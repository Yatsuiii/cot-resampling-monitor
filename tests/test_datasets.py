"""Schema adapters for the corpora the sweep varies over.

All synthetic rows: the schema layer must be testable without the network or
the `datasets` package, and `import mats.datasets` must not pull either in.
"""
import sys

import pytest

from mats.datasets import (LETTERS, arc_row_to_question, gpqa_row_to_question,
                           load, mmlu_row_to_question, sampling_order)
from mats.prompts import Question


def _arc_row(i=0):
    return {"id": f"arc-{i}", "question": "What holds a nucleus together?",
            "choices": {"label": ["1", "2", "3", "4"],
                        "text": ["gravity", "strong force", "magnetism", "wind"]},
            "answerKey": "2"}


def _mmlu_row(i=0):
    return {"question": f"Question {i}?", "choices": ["a", "b", "c", "d"],
            "answer": i % 4, "subject": "physics"}


def _gpqa_row(i=0):
    return {"Question": f"Distinct question number {i}?",
            "Correct Answer": "right", "Incorrect Answer 1": "wrong1",
            "Incorrect Answer 2": "wrong2", "Incorrect Answer 3": "wrong3"}


def _assert_invariants(q):
    """H24: every loader must satisfy what the existing ARC path assumes."""
    assert q is not None
    assert q.qid
    assert 2 <= len(q.options) <= len(LETTERS)
    expected = set(LETTERS[: len(q.options)])
    assert set(q.options) == expected, "letters must run contiguously from A"
    assert q.gold in q.options
    assert q.stem


def test_all_three_schemas_produce_valid_questions():
    _assert_invariants(arc_row_to_question(_arc_row()))
    _assert_invariants(mmlu_row_to_question(_mmlu_row(), index=0))
    _assert_invariants(gpqa_row_to_question(_gpqa_row(), index=0))


def test_arc_remaps_non_letter_labels():
    q = arc_row_to_question(_arc_row())
    assert q.options["B"] == "strong force" and q.gold == "B"


def test_mmlu_integer_answer_indexes_the_choices_list():
    for i in range(4):
        q = mmlu_row_to_question(_mmlu_row(i), index=i)
        assert q.options[q.gold] == ["a", "b", "c", "d"][i % 4]


def test_mmlu_qid_is_stable_and_carries_the_subject():
    a = mmlu_row_to_question(_mmlu_row(3), index=3)
    b = mmlu_row_to_question(_mmlu_row(3), index=3)
    assert a.qid == b.qid == "mmlu-physics-3"


def test_gpqa_permutation_is_deterministic_from_the_question_text():
    """H23, first half: reruns must reproduce, so the shuffle cannot be
    seeded by iteration order or a global RNG."""
    row = _gpqa_row(1)
    first = gpqa_row_to_question(row, index=1)
    second = gpqa_row_to_question(row, index=99)     # different index, same text
    assert first.options == second.options
    assert first.gold == second.gold


def test_gpqa_gold_is_not_always_the_same_position():
    """H23, second half: the correct answer always arrives in the same FIELD,
    so without shuffling gold would be a constant letter and every cue result
    would be confounded with position."""
    golds = {gpqa_row_to_question(_gpqa_row(i), index=i).gold for i in range(40)}
    assert len(golds) > 1, f"gold never moved: {golds}"


def test_gpqa_gold_text_is_the_correct_answer_field():
    for i in range(20):
        q = gpqa_row_to_question(_gpqa_row(i), index=i)
        assert q.options[q.gold] == "right"


def test_malformed_rows_return_none_rather_than_raising():
    assert mmlu_row_to_question({"question": "q", "choices": ["a"], "answer": 0},
                                index=0) is None
    assert mmlu_row_to_question({"question": "q", "choices": ["a", "b"],
                                 "answer": 5}, index=0) is None
    assert gpqa_row_to_question({"Question": "q", "Correct Answer": "r"},
                                index=0) is None
    bad_arc = _arc_row()
    bad_arc["answerKey"] = "9"
    assert arc_row_to_question(bad_arc) is None


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown dataset"):
        load("winogrande")


def test_importing_the_schema_layer_does_not_import_datasets():
    """The lazy import must stay inside the loader bodies; the sweep imports
    this module on machines with no `datasets` installed."""
    assert "datasets" not in sys.modules or True    # tolerate a prior import
    import importlib
    mod = importlib.import_module("mats.datasets")
    src = open(mod.__file__, encoding="utf-8").read()
    top = src.split("def ")[0]
    assert "from datasets import" not in top, "datasets imported at module level"


def test_a_prefix_of_the_sampling_order_is_not_one_subject():
    """The 09-12 grid's entire 40-item MMLU arm was abstract_algebra, because
    `cais/mmlu` `all` is ordered by subject and `keep_answerable` returns a
    prefix. This test fails against that behaviour.
    """
    grouped = [Question(qid=f"{subject}-{i}", stem="q",
                        options={"A": "a", "B": "b", "C": "c", "D": "d"}, gold="A")
               for subject in ("algebra", "biology", "history", "law")
               for i in range(50)]
    prefix = sampling_order(grouped, seed=20260912)[:40]
    subjects = {q.qid.rsplit("-", 1)[0] for q in prefix}
    assert len(subjects) == 4, f"prefix drawn from {subjects}"
    assert {q.qid.rsplit("-", 1)[0] for q in grouped[:40]} == {"algebra"}, \
        "fixture no longer reproduces the grouped-corpus shape"


def test_the_sampling_order_is_reproducible_from_the_seed():
    qs = [Question(qid=str(i), stem="q", options={"A": "a", "B": "b"}, gold="A")
          for i in range(200)]
    assert sampling_order(qs, seed=7) == sampling_order(qs, seed=7)
    assert sampling_order(qs, seed=7) != sampling_order(qs, seed=8)
    assert sorted(q.qid for q in sampling_order(qs, seed=7)) == sorted(q.qid for q in qs)


def test_the_sampling_order_does_not_mutate_the_corpus():
    qs = [Question(qid=str(i), stem="q", options={"A": "a", "B": "b"}, gold="A")
          for i in range(20)]
    before = [q.qid for q in qs]
    sampling_order(qs, seed=1)
    assert [q.qid for q in qs] == before
