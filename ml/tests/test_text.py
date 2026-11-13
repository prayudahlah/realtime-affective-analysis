from affective_ml.text import handle_negation, normalize_repetitive_chars, normalize_slang


def test_normalize_repetitive_chars():
    assert normalize_repetitive_chars("bagussss") == "bagus"
    assert normalize_repetitive_chars("mantappp") == "mantap"


def test_normalize_slang_keeps_punctuation():
    slang = {"gak": "tidak", "bgt": "banget"}
    assert normalize_slang("gak bgt!", slang) == "tidak banget!"
    assert normalize_slang("Gak", slang) == "tidak"


def test_handle_negation_marks_next_token():
    assert handle_negation(["tidak", "bagus"]) == ["tidak", "NEG_bagus"]
    assert handle_negation(["bagus", "sekali"]) == ["bagus", "sekali"]
