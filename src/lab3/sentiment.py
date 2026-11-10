"""Bài 1: phân tích cảm xúc bằng model đã train sẵn trên Hugging Face Hub.

Luồng xử lý: văn bản -> tokenizer -> input_ids, attention_mask -> model -> logits
-> softmax -> xác suất từng nhãn -> nhãn có xác suất cao nhất.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def validate_texts(texts: str | Sequence[str]) -> list[str]:
    """Kiểm tra và chuẩn hoá đầu vào văn bản thành list[str].

    Giả định:
    - texts là một chuỗi, hoặc list/tuple các chuỗi.
    - Mỗi chuỗi có ít nhất một ký tự không phải khoảng trắng.
    - Không sửa nội dung chuỗi (không strip, không đổi chữ thường): tokenizer tự xử lý.
    """
    if texts is None:
        raise TypeError("texts không được là None.")
    if isinstance(texts, str):
        items = [texts]
    elif isinstance(texts, (list, tuple)):
        items = list(texts)
    else:
        raise TypeError(f"texts phải là str hoặc list/tuple các str, nhận được {type(texts).__name__}.")

    if not items:
        raise ValueError("Danh sách texts rỗng.")
    for index, text in enumerate(items):
        if not isinstance(text, str):
            raise TypeError(f"Phần tử thứ {index} của texts phải là str, nhận được {type(text).__name__}.")
        if not text.strip():
            raise ValueError(f"Phần tử thứ {index} của texts rỗng hoặc chỉ có khoảng trắng.")
    return items


def validate_max_length(max_length: int, tokenizer: Any) -> int:
    """Kiểm tra số token tối đa cho mỗi câu.

    Giả định:
    - max_length là int (không nhận bool, float hay str).
    - Phải chứa được các token đặc biệt ([CLS], [SEP]) cộng ít nhất 1 token nội dung.
      Lý do: với max_length nhỏ hơn số token đặc biệt, tokenizer âm thầm bỏ qua việc cắt câu.
    - Không vượt quá tokenizer.model_max_length (số vị trí model nhận được).
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError(f"max_length phải là int, nhận được {type(max_length).__name__}.")

    minimum = tokenizer.num_special_tokens_to_add() + 1
    maximum = tokenizer.model_max_length
    if max_length < minimum:
        raise ValueError(f"max_length phải >= {minimum} (số token đặc biệt + 1), nhận được {max_length}.")
    if max_length > maximum:
        raise ValueError(f"max_length phải <= {maximum} (giới hạn của model), nhận được {max_length}.")
    return max_length


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Softmax tự cài đặt: p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z)).

    Giả định:
    - logits là torch.Tensor số thực hoặc số nguyên (không nhận bool), 1 chiều (số lớp)
      hoặc 2 chiều (batch, số lớp), có ít nhất 1 lớp và 1 dòng.
    - Mọi giá trị hữu hạn (không NaN, không ±inf).
    - Trừ max trước khi lấy exp để không bị tràn số khi logits lớn.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits phải là torch.Tensor, nhận được {type(logits).__name__}.")
    if logits.dtype == torch.bool or logits.is_complex():
        raise TypeError(f"logits phải là số thực hoặc số nguyên, nhận được dtype {logits.dtype}.")
    if logits.dim() not in (1, 2):
        raise ValueError(f"logits phải có 1 hoặc 2 chiều, nhận được {logits.dim()} chiều.")
    if logits.numel() == 0:
        raise ValueError("logits rỗng.")

    values = logits if logits.is_floating_point() else logits.to(torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("logits chứa NaN hoặc vô cực.")

    shifted = values - values.max(dim=-1, keepdim=True).values
    exponents = torch.exp(shifted)
    return exponents / exponents.sum(dim=-1, keepdim=True)


def logits_to_predictions(logits: torch.Tensor, id2label: dict[int, str]) -> list[dict[str, Any]]:
    """Chuyển logits thành nhãn dự đoán và xác suất của từng nhãn.

    Giả định:
    - logits 2 chiều (batch, số lớp); logits 1 chiều được coi là batch có 1 câu.
    - id2label là dict ánh xạ đủ mọi chỉ số lớp 0..C-1 sang tên nhãn.
    - Khi nhiều lớp có xác suất bằng nhau, chọn lớp có chỉ số nhỏ nhất (cách torch.argmax làm).
    """
    if not isinstance(id2label, dict):
        raise TypeError(f"id2label phải là dict, nhận được {type(id2label).__name__}.")

    probabilities = softmax(logits)
    raw_logits = logits.detach().to(torch.float32)
    if probabilities.dim() == 1:
        probabilities = probabilities.unsqueeze(0)
        raw_logits = raw_logits.unsqueeze(0)

    num_classes = probabilities.shape[-1]
    missing = [index for index in range(num_classes) if index not in id2label]
    if missing:
        raise ValueError(f"id2label thiếu nhãn cho các lớp: {missing}.")

    predictions = []
    for row_logits, row_probs in zip(raw_logits, probabilities):
        best = int(torch.argmax(row_probs))
        predictions.append(
            {
                "label": id2label[best],
                "label_id": best,
                "score": float(row_probs[best]),
                "logits": [float(value) for value in row_logits],
                "probabilities": {id2label[index]: float(row_probs[index]) for index in range(num_classes)},
            }
        )
    return predictions


def _looks_like_local_path(model_name: str) -> bool:
    """Đoán chuỗi model_name là đường dẫn local hay id trên Hub.

    Giả định: id trên Hub có dạng "tên" hoặc "tổ_chức/tên", không bắt đầu bằng ".", "/", "~"
    và không chứa dấu "\\". Chuỗi đã tồn tại trên đĩa luôn được coi là đường dẫn.
    """
    return Path(model_name).expanduser().exists() or model_name.startswith((".", "/", "~")) or "\\" in model_name


def load_sentiment_model(model_name: str, device: torch.device) -> tuple[Any, torch.nn.Module]:
    """Tải tokenizer và model phân loại văn bản, đặt model ở chế độ eval trên device.

    Giả định:
    - model_name là id trên Hugging Face Hub hoặc đường dẫn tới thư mục đã lưu bằng save_pretrained.
    - Lần đầu tải từ Hub cần internet; các lần sau dùng cache trên máy.
    - Model là model phân loại chuỗi (có config.id2label).
    - device là torch.device mà máy đang có.
    - Trọng số được đọc hẳn vào RAM (disable_mmap=True), không ánh xạ thẳng từ file. Lý do: phần
      dữ liệu trong file safetensors của model Bài 1 bắt đầu ở byte 12.518 (không chia hết cho 4).
      Nếu ánh xạ thẳng, mọi tensor float32 nằm ở địa chỉ lệch căn hàng, và phép nhân ma trận–vector
      của PyTorch trên CPU ARM64 (batch 1) bị lỗi căn hàng -> Bus error. Đọc vào RAM thì PyTorch
      tự cấp phát vùng nhớ căn hàng đúng.
    """
    if model_name is None or not isinstance(model_name, str):
        raise TypeError(f"model_name phải là str, nhận được {type(model_name).__name__}.")
    if not model_name.strip():
        raise ValueError("model_name không được rỗng.")
    if not isinstance(device, torch.device):
        raise TypeError(f"device phải là torch.device, nhận được {type(device).__name__}.")

    if _looks_like_local_path(model_name):
        local_dir = Path(model_name).expanduser()
        if not local_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục model '{local_dir}'.")
        source = str(local_dir)
    else:
        source = model_name

    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source, disable_mmap=True)
    except OSError as error:
        raise RuntimeError(
            f"Không tải được model '{model_name}'. Kiểm tra lại tên model hoặc kết nối internet."
        ) from error

    model.eval()
    model.to(device)
    return tokenizer, model


def tokenize_texts(tokenizer: Any, texts: str | Sequence[str], max_length: int) -> dict[str, Any]:
    """Tokenize một hoặc nhiều câu.

    Giả định:
    - tokenizer là tokenizer của Hugging Face.
    - Câu dài hơn max_length bị cắt (giữ phần đầu, token [SEP] vẫn ở cuối).
    - Các câu trong batch được pad tới độ dài của câu dài nhất; attention_mask = 0 ở vị trí pad.
    Trả về dict gồm:
    - "texts": list câu đầu vào,
    - "encoding": kết quả tokenizer (tensor input_ids, attention_mask),
    - "tokens": list token thật của từng câu (đã bỏ token pad).
    """
    items = validate_texts(texts)
    validate_max_length(max_length, tokenizer)

    encoding = tokenizer(items, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    tokens = [
        tokenizer.convert_ids_to_tokens(ids[mask.bool()].tolist())
        for ids, mask in zip(encoding["input_ids"], encoding["attention_mask"])
    ]
    return {"texts": items, "encoding": encoding, "tokens": tokens}


def predict_sentiment(
    model: torch.nn.Module, tokenizer: Any, texts: str | Sequence[str], max_length: int
) -> list[dict[str, Any]]:
    """Dự đoán cảm xúc theo từng bước: tokenize -> model -> logits -> softmax tự cài đặt.

    Giả định:
    - model và tokenizer đi cùng nhau (tải từ cùng một model_name).
    - model đã ở chế độ eval và nằm trên thiết bị muốn chạy; input được chuyển sang thiết bị đó.
    - Chỉ đưa input_ids và attention_mask vào model (DistilBERT không dùng token_type_ids).
    """
    tokenized = tokenize_texts(tokenizer, texts, max_length)
    encoding = tokenized["encoding"]
    device = next(model.parameters()).device

    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    predictions = logits_to_predictions(output.logits.detach().cpu(), model.config.id2label)
    for text, tokens, prediction in zip(tokenized["texts"], tokenized["tokens"], predictions):
        prediction["text"] = text
        prediction["num_tokens"] = len(tokens)
    return predictions


def predict_with_pipeline(
    model: torch.nn.Module, tokenizer: Any, texts: str | Sequence[str], max_length: int
) -> list[dict[str, Any]]:
    """Dự đoán bằng pipeline("sentiment-analysis") của Hugging Face, dùng để đối chiếu.

    Giả định:
    - Dùng chung model và tokenizer với predict_sentiment, cùng max_length, để hai cách so được với nhau.
    - top_k=None để pipeline trả về xác suất của mọi nhãn.
    - Xác suất được sắp theo thứ tự nhãn của model (id 0, 1, ...) cho dễ so với cách tự tính.
    """
    items = validate_texts(texts)
    validate_max_length(max_length, tokenizer)
    device = next(model.parameters()).device
    label_order = [model.config.id2label[index] for index in sorted(model.config.id2label)]

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    raw_results = classifier(items, top_k=None, truncation=True, max_length=max_length)

    predictions = []
    for text, scores in zip(items, raw_results):
        by_label = {item["label"]: float(item["score"]) for item in scores}
        if set(by_label) != set(label_order):
            raise ValueError(f"pipeline trả về nhãn {sorted(by_label)} khác với nhãn của model {label_order}.")
        best = max(label_order, key=lambda label: by_label[label])
        predictions.append(
            {
                "text": text,
                "label": best,
                "score": by_label[best],
                "probabilities": {label: by_label[label] for label in label_order},
            }
        )
    return predictions


def max_probability_difference(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> float:
    """Sai lệch xác suất lớn nhất giữa hai danh sách dự đoán của cùng các câu.

    Giả định:
    - Hai danh sách cùng độ dài, cùng thứ tự câu.
    - Mỗi phần tử có khoá "probabilities" là dict {nhãn: xác suất} với cùng tập nhãn.
    """
    if not isinstance(first, list) or not isinstance(second, list):
        raise TypeError("Cả hai đầu vào phải là list.")
    if not first:
        raise ValueError("Danh sách dự đoán rỗng.")
    if len(first) != len(second):
        raise ValueError(f"Hai danh sách khác độ dài: {len(first)} và {len(second)}.")

    largest = 0.0
    for index, (left, right) in enumerate(zip(first, second)):
        left_probs, right_probs = left["probabilities"], right["probabilities"]
        if set(left_probs) != set(right_probs):
            raise ValueError(f"Câu thứ {index}: hai bên có tập nhãn khác nhau.")
        for label, value in left_probs.items():
            largest = max(largest, abs(value - right_probs[label]))
    return largest
