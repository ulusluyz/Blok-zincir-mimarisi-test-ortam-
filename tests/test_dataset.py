"""
Unit tests for expanded 10-category dataset.
"""

from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.features import normalize_turkish


def test_categories_10_defined():
    assert len(CATEGORIES) == 10
    assert "Fizik" in CATEGORIES
    assert "Biyoloji" in CATEGORIES
    assert "Tarih" in CATEGORIES
    assert "Edebiyat" in CATEGORIES


def test_no_exact_leakage_10cat():
    train_data, test_data, multi_data = get_full_10cat_dataset()
    train_texts = {normalize_turkish(t) for t, _ in train_data}
    test_texts = {normalize_turkish(t) for t, _ in test_data}

    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping texts in expanded 10-cat dataset: {overlap}"


def test_multi_topic_queries_defined():
    _, _, multi_data = get_full_10cat_dataset()
    assert len(multi_data) >= 10
    for text, topics in multi_data:
        assert len(topics) >= 2
        for t in topics:
            assert t in CATEGORIES
