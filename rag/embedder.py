"""
RAG Layer 1: Text Embedder

Dùng sentence-transformers chạy LOCAL — không tốn API, không tốn tiền.
Model: all-MiniLM-L6-v2
  - Size: ~80MB (download 1 lần, cache mãi)
  - Dim:  384
  - Tốc độ: ~14,000 sentences/giây trên CPU
  - Đủ tốt cho CV/JD matching (benchmark MTEB competitive)

Tại sao không dùng Gemini Embedding API?
  - Free tier có rate limit (1500 req/ngày)
  - Batch 200 CVs × nhiều chunks = dễ vượt limit
  - Local model chạy offline, không phụ thuộc internet
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """
    Load model 1 lần duy nhất, cache trong memory.
    lru_cache đảm bảo dù gọi 1000 lần cũng chỉ load 1 lần.
    """
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s (first time only)", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Embed 1 đoạn text thành vector 384 chiều.

    Args:
        text: Chuỗi text cần embed (CV skills, JD requirement, v.v.)

    Returns:
        List 384 floats — vector representation của text
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    model = _load_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed nhiều đoạn text cùng lúc — nhanh hơn gọi embed_text từng cái.
    sentence-transformers tự batch internally.

    Args:
        texts: List các đoạn text cần embed

    Returns:
        List các vectors tương ứng
    """
    if not texts:
        return []

    # Lọc empty strings, ghi nhớ index để map lại sau
    valid_texts  = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not valid_texts:
        raise ValueError("All texts are empty")

    indices, clean_texts = zip(*valid_texts)

    model  = _load_model()
    vectors = model.encode(
        list(clean_texts),
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )

    dim = vectors.shape[1]
    result = [[0.0] * dim] * len(texts)
    for original_idx, vec in zip(indices, vectors):
        result[original_idx] = vec.tolist()

    logger.debug("Embedded %d texts", len(valid_texts))
    return result