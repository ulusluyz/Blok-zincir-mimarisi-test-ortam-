"""
Unit tests for dataset module, checking for exact and near-duplicate leakage.
"""

from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA, get_expanded_dataset
from src.features import normalize_turkish


def test_categories_defined():
    assert len(CATEGORIES) == 6
    assert "Selamlaşma" in CATEGORIES
    assert "Gezegenler" in CATEGORIES
    assert "Hayvanlar" in CATEGORIES
    assert "Elektronik" in CATEGORIES
    assert "Matematik" in CATEGORIES
    assert "Alakasız" in CATEGORIES


def test_no_exact_leakage():
    train_data, test_data = get_expanded_dataset()
    train_texts = {normalize_turkish(t) for t, _ in train_data}
    test_texts = {normalize_turkish(t) for t, _ in test_data}

    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping texts in expanded dataset: {overlap}"


def test_near_duplicate_overlap_analysis():
    """
    Check word Jaccard similarity between all train and test pairs to ensure no near-identical duplicates.
    """
    train_data, test_data = get_expanded_dataset()

    high_overlap_pairs = []
    for tr_text, tr_cat in train_data:
        tr_words = set(normalize_turkish(tr_text).split())
        for te_text, te_cat in test_data:
            te_words = set(normalize_turkish(te_text).split())
            if not tr_words or not te_words:
                continue
            jaccard = len(tr_words.intersection(te_words)) / len(tr_words.union(te_words))
            if jaccard > 0.8:  # Near duplicate threshold
                high_overlap_pairs.append((tr_text, te_text, jaccard))

    assert len(high_overlap_pairs) == 0, f"Found near-duplicate train-test pairs: {high_overlap_pairs}"
