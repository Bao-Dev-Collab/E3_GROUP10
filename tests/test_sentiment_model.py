"""Test Bài 1 với model thật (đánh dấu slow vì cần tải model lần đầu)."""

import pytest
import torch

from lab3 import config
from lab3.sentiment import (
    load_sentiment_model,
    max_probability_difference,
    predict_sentiment,
    predict_with_pipeline,
    tokenize_texts,
)

pytestmark = pytest.mark.slow

MAX_LENGTH = config.EXERCISE1_MAX_LENGTH


# ------------------------------------------------------------------ tokenize
def test_tokenize_single_sentence_structure(sentiment_bundle):
    tokenizer, _ = sentiment_bundle
    result = tokenize_texts(tokenizer, "The movie was wonderful.", MAX_LENGTH)
    tokens = result["tokens"][0]
    ids = result["encoding"]["input_ids"][0].tolist()

    assert tokens[0] == tokenizer.cls_token
    assert tokens[-1] == tokenizer.sep_token
    assert tokenizer.convert_tokens_to_ids(tokens) == ids
    assert result["encoding"]["attention_mask"][0].tolist() == [1] * len(ids)


def test_tokenize_is_case_insensitive_for_uncased_model(sentiment_bundle):
    tokenizer, _ = sentiment_bundle
    upper = tokenize_texts(tokenizer, "GREAT MOVIE", MAX_LENGTH)["encoding"]["input_ids"]
    lower = tokenize_texts(tokenizer, "great movie", MAX_LENGTH)["encoding"]["input_ids"]
    assert torch.equal(upper, lower)


def test_tokenize_batch_pads_shorter_sentence(sentiment_bundle):
    tokenizer, _ = sentiment_bundle
    result = tokenize_texts(tokenizer, ["Short.", "This sentence is clearly a lot longer than the first one."], MAX_LENGTH)
    mask = result["encoding"]["attention_mask"]

    assert result["encoding"]["input_ids"].shape[0] == 2
    assert mask[0].sum() < mask[1].sum()
    assert mask[0][-1] == 0  # vị trí pad
    assert tokenizer.pad_token not in result["tokens"][0]
    assert len(result["tokens"][0]) == int(mask[0].sum())


def test_tokenize_truncates_long_text(sentiment_bundle):
    tokenizer, _ = sentiment_bundle
    result = tokenize_texts(tokenizer, "good " * 1000, 16)
    assert len(result["tokens"][0]) == 16
    assert result["tokens"][0][-1] == tokenizer.sep_token


def test_tokenize_rejects_max_length_below_special_tokens(sentiment_bundle):
    tokenizer, _ = sentiment_bundle
    with pytest.raises(ValueError):
        tokenize_texts(tokenizer, "hello world", 1)


# ------------------------------------------------------------------- predict
def test_predict_outputs_are_valid_probabilities(sentiment_bundle):
    tokenizer, model = sentiment_bundle
    texts = ["I loved it.", "Terrible.", "It was a film."]
    predictions = predict_sentiment(model, tokenizer, texts, MAX_LENGTH)

    assert [p["text"] for p in predictions] == texts
    for prediction in predictions:
        assert prediction["label"] in model.config.id2label.values()
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
        assert prediction["score"] >= 1 / len(model.config.id2label)
        assert prediction["num_tokens"] >= 3


def test_predict_clear_positive_and_negative_sentences(sentiment_bundle):
    tokenizer, model = sentiment_bundle
    positive, negative = predict_sentiment(
        model, tokenizer, ["An absolute masterpiece, I loved every minute.", "Awful, boring and a total waste of time."],
        MAX_LENGTH,
    )
    assert positive["label"] == "POSITIVE"
    assert negative["label"] == "NEGATIVE"


def test_predict_batch_equals_one_by_one(sentiment_bundle):
    """Pad trong batch không được làm đổi kết quả của từng câu."""
    tokenizer, model = sentiment_bundle
    texts = ["Fine.", "The story drags on far too long and the dialogue is clumsy throughout."]
    together = predict_sentiment(model, tokenizer, texts, MAX_LENGTH)
    separate = [predict_sentiment(model, tokenizer, text, MAX_LENGTH)[0] for text in texts]
    assert max_probability_difference(together, separate) < 1e-4


def test_predict_handles_emoji_unicode_and_very_long_text(sentiment_bundle):
    tokenizer, model = sentiment_bundle
    texts = ["Phim hay quá 😍🎬", "最高の映画", "boring " * 2000]
    predictions = predict_sentiment(model, tokenizer, texts, MAX_LENGTH)
    assert len(predictions) == 3
    assert predictions[2]["num_tokens"] == MAX_LENGTH


def test_manual_and_pipeline_agree(sentiment_bundle):
    tokenizer, model = sentiment_bundle
    texts = list(config.EXERCISE1_SAMPLE_TEXTS) + ["Phim hay quá 😍", "meh " * 700]
    manual = predict_sentiment(model, tokenizer, texts, MAX_LENGTH)
    piped = predict_with_pipeline(model, tokenizer, texts, MAX_LENGTH)

    assert [p["label"] for p in manual] == [p["label"] for p in piped]
    assert max_probability_difference(manual, piped) <= config.EXERCISE1_PROBABILITY_TOLERANCE


def test_load_on_cpu_gives_aligned_weights():
    """Nguyên nhân gốc của lỗi Bus error: trọng số nằm ở địa chỉ lệch căn hàng. Không tensor nào được lệch."""
    _, model = load_sentiment_model(config.SENTIMENT_MODEL_NAME, torch.device("cpu"))
    misaligned = [
        name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.data_ptr() % tensor.element_size() != 0
    ]
    assert misaligned == []


def test_load_unknown_hub_model_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Không tải được model"):
        load_sentiment_model("this-model-does-not-exist-xyz/abc123", torch.device("cpu"))
