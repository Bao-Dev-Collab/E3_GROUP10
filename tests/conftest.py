"""Fixture dùng chung cho các test.

Model thật chỉ tải một lần cho cả phiên test (scope="session") để test chạy nhanh hơn.
"""

import pytest

from lab3 import config
from lab3.sentiment import load_sentiment_model
from lab3.utils import get_device


class FakeTokenizer:
    """Tokenizer giả, chỉ có các thuộc tính mà validate_max_length cần. Không cần tải gì."""

    def __init__(self, num_special_tokens: int = 2, model_max_length: int = 512):
        self._num_special_tokens = num_special_tokens
        self.model_max_length = model_max_length

    def num_special_tokens_to_add(self) -> int:
        return self._num_special_tokens


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


class WhitespaceTokenizer:
    """Tokenizer giả: mỗi từ (tách theo khoảng trắng) là 1 token, cộng 2 token đặc biệt. Không cần tải gì."""

    def __call__(self, texts, add_special_tokens=True, truncation=False, verbose=True):
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(text.split()) + extra) for text in texts]}


@pytest.fixture
def whitespace_tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Model DistilBERT tí hon (1 lớp, 32 chiều, trọng số ngẫu nhiên) + tokenizer thật của base model.

    Dùng để test toàn bộ đường đi của Trainer trong vài giây trên CPU, không cần tải model 270 MB.
    """
    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification
    import torch

    folder = tmp_path_factory.mktemp("tiny_distilbert")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    model_config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size, dim=32, n_layers=1, n_heads=2, hidden_dim=64,
        num_labels=2, id2label=config.ID2LABEL, label2id={v: k for k, v in config.ID2LABEL.items()},
    )
    torch.manual_seed(0)
    DistilBertForSequenceClassification(model_config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


@pytest.fixture(scope="session")
def sentiment_bundle():
    """Tokenizer và model Bài 1 trên thiết bị tự chọn (MPS trên máy Mac M1)."""
    return load_sentiment_model(config.SENTIMENT_MODEL_NAME, get_device())
