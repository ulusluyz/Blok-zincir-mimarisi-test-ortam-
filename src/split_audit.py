"""Fail-closed exact and normalized overlap audit across dataset splits."""
import re
import unicodedata
from collections import defaultdict


def normalize_text(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'\w+', value, flags=re.UNICODE))


def audit_splits(train, validation, test_single, test_multi):
    splits = {
        'train': [text for text, _ in train],
        'validation': [item['text'] for item in validation],
        'test_single': [item['text'] for item in test_single],
        'test_multi': [item['text'] for item in test_multi],
    }
    owners = defaultdict(set)
    for name, texts in splits.items():
        for text in texts:
            key = normalize_text(text)
            if not key:
                raise ValueError(f'Empty normalized text in {name}')
            owners[key].add(name)
    collisions = [{'text': key, 'splits': sorted(names)} for key, names in owners.items() if len(names) > 1]
    return {'passed': not collisions, 'split_sizes': {k: len(v) for k, v in splits.items()}, 'cross_split_collisions': collisions}


def assert_no_split_leakage(train, validation, test_single, test_multi):
    report = audit_splits(train, validation, test_single, test_multi)
    if not report['passed']:
        raise ValueError(f"Cross-split leakage: {report['cross_split_collisions'][:5]}")
    return report
