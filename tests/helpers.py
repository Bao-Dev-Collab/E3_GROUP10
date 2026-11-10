"""Hàm tạo dữ liệu giả dùng chung cho các test (không phải fixture)."""

import random

import pandas as pd

POSITIVE_WORDS = ["good", "great", "excellent", "love", "wonderful", "brilliant"]
NEGATIVE_WORDS = ["bad", "awful", "terrible", "hate", "boring", "dull"]
FILLER_WORDS = ["movie", "film", "plot", "actor", "scene", "story", "director", "music"]


def synthetic_reviews(n_per_class: int, seed: int) -> pd.DataFrame:
    """Review giả tách được: mỗi review có 3 từ cảm xúc + 4 từ trung tính, thứ tự ngẫu nhiên theo seed."""
    rng = random.Random(seed)
    rows = []
    for label, words in ((1, POSITIVE_WORDS), (0, NEGATIVE_WORDS)):
        for _ in range(n_per_class):
            tokens = rng.sample(words, 3) + rng.sample(FILLER_WORDS, 4)
            rng.shuffle(tokens)
            rows.append({"text": " ".join(tokens), "label": label})
    return pd.DataFrame(rows)
