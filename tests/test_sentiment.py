"""Test nhanh cho lab3.sentiment: không cần tải model."""

import pytest
import torch

from lab3.sentiment import (
    load_sentiment_model,
    logits_to_predictions,
    max_probability_difference,
    softmax,
    validate_max_length,
    validate_texts,
)

BINARY_LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


# ------------------------------------------------------------- validate_texts
def test_validate_texts_single_string_becomes_list():
    assert validate_texts("Great film") == ["Great film"]


@pytest.mark.parametrize("texts", [["a", "b"], ("a", "b")])
def test_validate_texts_accepts_list_and_tuple(texts):
    assert validate_texts(texts) == ["a", "b"]


def test_validate_texts_keeps_text_unchanged():
    text = "  Mixed CASE with spaces  "
    assert validate_texts(text) == [text]


def test_validate_texts_accepts_unicode_and_emoji():
    assert validate_texts(["Phim hay quá 😍", "最高の映画"]) == ["Phim hay quá 😍", "最高の映画"]


def test_validate_texts_none_raises_type_error():
    with pytest.raises(TypeError):
        validate_texts(None)


@pytest.mark.parametrize("texts", [123, 4.5, {"text": "a"}, b"bytes"])
def test_validate_texts_wrong_type_raises(texts):
    with pytest.raises(TypeError):
        validate_texts(texts)


@pytest.mark.parametrize("texts", [["ok", 5], ["ok", None]])
def test_validate_texts_wrong_element_type_names_index(texts):
    with pytest.raises(TypeError, match="thứ 1"):
        validate_texts(texts)


@pytest.mark.parametrize("texts", ["", "   ", "\n\t", [], ["ok", "  "]])
def test_validate_texts_empty_raises_value_error(texts):
    with pytest.raises(ValueError):
        validate_texts(texts)


# -------------------------------------------------------- validate_max_length
@pytest.mark.parametrize("value", [3, 128, 512])
def test_validate_max_length_accepts_valid_range(fake_tokenizer, value):
    assert validate_max_length(value, fake_tokenizer) == value


@pytest.mark.parametrize("value", [2, 1, 0, -5])
def test_validate_max_length_too_small_raises(fake_tokenizer, value):
    with pytest.raises(ValueError):
        validate_max_length(value, fake_tokenizer)


def test_validate_max_length_above_model_limit_raises(fake_tokenizer):
    with pytest.raises(ValueError):
        validate_max_length(513, fake_tokenizer)


@pytest.mark.parametrize("value", [2.5, "10", True, None])
def test_validate_max_length_wrong_type_raises(fake_tokenizer, value):
    with pytest.raises(TypeError):
        validate_max_length(value, fake_tokenizer)


# -------------------------------------------------------------------- softmax
def test_softmax_matches_torch_on_random_batch():
    torch.manual_seed(0)
    logits = torch.randn(5, 4) * 10
    assert torch.allclose(softmax(logits), torch.softmax(logits, dim=-1), atol=1e-6)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(torch.tensor([[1.0, 2.0, 3.0], [-4.0, 0.0, 9.0]]))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))


def test_softmax_one_dimensional_input():
    probabilities = softmax(torch.tensor([0.0, 0.0]))
    assert probabilities.shape == (2,)
    assert torch.allclose(probabilities, torch.tensor([0.5, 0.5]))


def test_softmax_large_logits_do_not_overflow():
    large = softmax(torch.tensor([1000.0, 1001.0, 1002.0]))
    small = softmax(torch.tensor([0.0, 1.0, 2.0]))
    assert torch.isfinite(large).all()
    assert torch.allclose(large, small, atol=1e-6)


def test_softmax_very_negative_logits_do_not_underflow_to_nan():
    probabilities = softmax(torch.tensor([-1000.0, -1000.0]))
    assert torch.allclose(probabilities, torch.tensor([0.5, 0.5]))


def test_softmax_single_class_is_certain():
    assert torch.allclose(softmax(torch.tensor([[7.3]])), torch.tensor([[1.0]]))


def test_softmax_integer_tensor_is_accepted():
    assert torch.allclose(softmax(torch.tensor([1, 1, 1])), torch.full((3,), 1 / 3))


@pytest.mark.parametrize("value", [[1.0, 2.0], None, 3.0])
def test_softmax_non_tensor_raises_type_error(value):
    with pytest.raises(TypeError):
        softmax(value)


def test_softmax_bool_tensor_raises_type_error():
    with pytest.raises(TypeError):
        softmax(torch.tensor([True, False]))


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(2, 2, 2),  # 3 chiều
        torch.tensor(1.0),  # 0 chiều
        torch.empty(0),  # rỗng
        torch.empty(0, 2),  # batch rỗng
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
        torch.tensor([1.0, float("-inf")]),
    ],
)
def test_softmax_invalid_tensor_raises_value_error(logits):
    with pytest.raises(ValueError):
        softmax(logits)


# ------------------------------------------------------- logits_to_predictions
def test_logits_to_predictions_picks_highest_probability():
    predictions = logits_to_predictions(torch.tensor([[-2.0, 3.0], [4.0, 1.0]]), BINARY_LABELS)
    assert [p["label"] for p in predictions] == ["POSITIVE", "NEGATIVE"]
    for prediction in predictions:
        assert prediction["score"] == pytest.approx(max(prediction["probabilities"].values()))
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_logits_to_predictions_one_dimensional_is_one_sentence():
    predictions = logits_to_predictions(torch.tensor([0.1, 0.9]), BINARY_LABELS)
    assert len(predictions) == 1
    assert predictions[0]["logits"] == pytest.approx([0.1, 0.9])


def test_logits_to_predictions_tie_picks_lowest_index():
    prediction = logits_to_predictions(torch.tensor([[2.0, 2.0]]), BINARY_LABELS)[0]
    assert prediction["label_id"] == 0
    assert prediction["score"] == pytest.approx(0.5)


def test_logits_to_predictions_works_for_more_than_two_classes():
    labels = {0: "neg", 1: "neu", 2: "pos"}
    prediction = logits_to_predictions(torch.tensor([[0.0, 5.0, 1.0]]), labels)[0]
    assert prediction["label"] == "neu"
    assert set(prediction["probabilities"]) == {"neg", "neu", "pos"}


def test_logits_to_predictions_missing_label_raises():
    with pytest.raises(ValueError, match="thiếu nhãn"):
        logits_to_predictions(torch.tensor([[0.0, 1.0, 2.0]]), BINARY_LABELS)


@pytest.mark.parametrize("labels", [None, ["NEGATIVE", "POSITIVE"]])
def test_logits_to_predictions_label_map_wrong_type_raises(labels):
    with pytest.raises(TypeError):
        logits_to_predictions(torch.tensor([[0.0, 1.0]]), labels)


# --------------------------------------------------- max_probability_difference
def _prediction(negative: float, positive: float) -> dict:
    return {"probabilities": {"NEGATIVE": negative, "POSITIVE": positive}}


def test_max_probability_difference_identical_is_zero():
    predictions = [_prediction(0.2, 0.8), _prediction(0.9, 0.1)]
    assert max_probability_difference(predictions, predictions) == 0.0


def test_max_probability_difference_returns_largest_gap():
    left = [_prediction(0.2, 0.8), _prediction(0.9, 0.1)]
    right = [_prediction(0.25, 0.75), _prediction(0.6, 0.4)]
    assert max_probability_difference(left, right) == pytest.approx(0.3)


def test_max_probability_difference_length_mismatch_raises():
    with pytest.raises(ValueError):
        max_probability_difference([_prediction(0.5, 0.5)], [])


def test_max_probability_difference_empty_raises():
    with pytest.raises(ValueError):
        max_probability_difference([], [])


def test_max_probability_difference_different_labels_raises():
    other = [{"probabilities": {"neg": 0.5, "pos": 0.5}}]
    with pytest.raises(ValueError):
        max_probability_difference([_prediction(0.5, 0.5)], other)


@pytest.mark.parametrize("bad", [None, "abc", ({"probabilities": {}},)])
def test_max_probability_difference_wrong_type_raises(bad):
    with pytest.raises(TypeError):
        max_probability_difference(bad, [])


# ------------------------------------------------ load_sentiment_model (input)
@pytest.mark.parametrize("name", [None, 123, ["model"]])
def test_load_model_name_wrong_type_raises(name):
    with pytest.raises(TypeError):
        load_sentiment_model(name, torch.device("cpu"))


@pytest.mark.parametrize("name", ["", "   "])
def test_load_model_empty_name_raises(name):
    with pytest.raises(ValueError):
        load_sentiment_model(name, torch.device("cpu"))


def test_load_model_device_wrong_type_raises():
    with pytest.raises(TypeError):
        load_sentiment_model("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "cpu")


def test_load_model_missing_local_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sentiment_model(str(tmp_path / "khong_ton_tai"), torch.device("cpu"))
