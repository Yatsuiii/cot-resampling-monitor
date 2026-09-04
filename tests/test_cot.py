from mats.cot import parse_answer, split_sentences


def test_split_on_newlines_and_punctuation():
    text = "First I read the question. Then I think.\nNext line here."
    assert split_sentences(text) == [
        "First I read the question.",
        "Then I think.",
        "Next line here.",
    ]


def test_split_keeps_decimals_and_abbreviations_intact():
    text = "The ratio is 9.11 not 9.8. We compare e.g. these two."
    assert split_sentences(text) == [
        "The ratio is 9.11 not 9.8.",
        "We compare e.g. these two.",
    ]


def test_split_drops_blank_chunks():
    assert split_sentences("\n\n  \nOne sentence.\n\n") == ["One sentence."]


def test_parse_answer_prefers_explicit_statement():
    assert parse_answer("Maybe A or C. The answer is (B).") == "B"


def test_parse_answer_takes_last_occurrence():
    assert parse_answer("answer is A\n...reconsidering...\nanswer: D") == "D"


def test_parse_answer_reads_boxed():
    assert parse_answer("... so \\boxed{C}") == "C"


def test_parse_answer_none_when_absent():
    assert parse_answer("I am not sure how to proceed here.") is None
