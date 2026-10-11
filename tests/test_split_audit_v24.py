import pytest
from src.split_audit import normalize_text, audit_splits, assert_no_split_leakage
from src.dataset import get_full_split_datasets


def test_existing_splits_have_no_normalized_exact_overlap():
    assert assert_no_split_leakage(*get_full_split_datasets())['passed']


def test_punctuation_and_case_overlap_detected():
    result = audit_splits([('Merhaba, Dünya!', 'x')], [], [], [{'text':'merhaba dünya', 'expected_categories':['x']}])
    assert not result['passed']
    assert result['cross_split_collisions'][0]['splits'] == ['test_multi', 'train']


def test_unicode_compatibility_normalization():
    assert normalize_text('ＡＢＣ  123') == normalize_text('abc 123')


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        audit_splits([('...', 'x')], [], [], [])
