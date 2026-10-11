import json
import os
import subprocess
import sys

from src.features import FeatureExtractor


def test_tie_breaking_lexicographic():
    fe = FeatureExtractor(max_features=2).fit(["xyz abc", "abc xyz"])
    assert list(fe.vocab) == sorted(fe.vocab)


def test_vocabulary_stable_across_python_hash_seeds():
    code = "import json; from src.features import FeatureExtractor; f=FeatureExtractor(max_features=64).fit([\"ankara istanbul izmir\",\"izmir ankara bursa\",\"istanbul bursa ankara\"]); print(json.dumps(list(f.vocab), ensure_ascii=False))"
    results = []
    for seed in ("1", "42", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        results.append(json.loads(result.stdout))
    assert results[0] == results[1] == results[2]
