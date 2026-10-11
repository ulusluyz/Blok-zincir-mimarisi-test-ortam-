import pytest
from src import dataset
from src.split_audit import audit_splits


def test_validation_leakage_is_rejected_at_dataset_entry(monkeypatch):
    monkeypatch.setattr(dataset, 'VAL_DATASET', [{'text': dataset.EXPANDED_TRAIN_DATA[0][0], 'category': 'Selamlaşma'}])
    with pytest.raises(ValueError, match='Cross-split leakage'):
        dataset.get_full_split_datasets()


def test_multi_leakage_is_rejected_at_dataset_entry(monkeypatch):
    monkeypatch.setattr(dataset, 'UNSEEN_TEST_MULTI_DATASET', [{'text': dataset.EXPANDED_TRAIN_DATA[0][0], 'expected_categories': ['Selamlaşma']}])
    with pytest.raises(ValueError, match='Cross-split leakage'):
        dataset.get_full_split_datasets()


def test_overlap_between_validation_and_single_detected():
    report = audit_splits([], [{'text': 'Bir Test!', 'category': 'x'}], [{'text': 'bir test', 'category': 'x'}], [])
    assert report['cross_split_collisions'][0]['splits'] == ['test_single', 'validation']
