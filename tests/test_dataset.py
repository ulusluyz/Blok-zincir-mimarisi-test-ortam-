"""
Unit tests for dataset module.
"""

from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA, get_expanded_dataset


def test_categories_defined():
    assert len(CATEGORIES) == 6
    assert "Selamlaşma" in CATEGORIES
    assert "Gezegenler" in CATEGORIES
    assert "Hayvanlar" in CATEGORIES
    assert "Elektronik" in CATEGORIES
    assert "Matematik" in CATEGORIES
    assert "Alakasız" in CATEGORIES


def test_small_dataset_structure():
    assert len(SMALL_TRAIN_DATA) >= 48
    assert len(SMALL_TEST_DATA) >= 24

    train_texts = {t.strip().lower() for t, _ in SMALL_TRAIN_DATA}
    test_texts = {t.strip().lower() for t, _ in SMALL_TEST_DATA}

    # Verify no exact overlap
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping texts in small dataset: {overlap}"


def test_expanded_dataset_no_leakage():
    train_data, test_data = get_expanded_dataset()
    assert len(train_data) > len(SMALL_TRAIN_DATA)
    assert len(test_data) > len(SMALL_TEST_DATA)

    train_texts = {t.strip().lower() for t, _ in train_data}
    test_texts = {t.strip().lower() for t, _ in test_data}

    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping texts in expanded dataset: {overlap}"
