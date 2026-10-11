"""Cross-process reproducibility of learned encoder and G routing."""
import json
import os
import subprocess
import sys

SCRIPT = r'''
import hashlib, json
from src.features import FeatureExtractor
from src.encoder import TrainableBinaryEncoder
from src.learnable_g import LearnableMultiPrototypeGRepresentative
texts = ["ankara istanbul başkent", "istanbul boğaz ulaşım", "matematik matris cebir", "cebir matris determinant", "başkent ankara türkiye", "matematik determinant çarpım"]
labels = [0, 0, 1, 1, 0, 1]
f = FeatureExtractor(max_features=48).fit(texts)
X = [f.transform_dense(t) for t in texts]
e = TrainableBinaryEncoder(input_dim=len(f.vocab), bit_length=16, num_categories=2, seed=17)
e.fit(X, labels, epochs=2)
encoded = [e.encode(x).to_bool_list() for x in X]
groups = [LearnableMultiPrototypeGRepresentative(f'g{i}', f'group{i}', num_sub_prototypes=2, bit_length=16, seed=17+i) for i in range(2)]
for i in range(2):
    for j, bits in enumerate(encoded):
        groups[i].update_gradients(bits, is_target=(labels[j] == i), lr=0.05)
routes = [max(range(2), key=lambda i: (groups[i].evaluate_suitability(e.encode(x)), -i)) for x in X]
prototypes = [g.sub_prototypes for g in groups]
weights = hashlib.sha256(json.dumps(e.W, separators=(',', ':')).encode()).hexdigest()
print(json.dumps({'vocab':list(f.vocab),'encoded':encoded,'weights':weights,'routes':routes,'prototypes':prototypes}, sort_keys=True))
'''

def _run(hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    result = subprocess.run([sys.executable, '-c', SCRIPT], env=env, capture_output=True, text=True, check=True, timeout=45)
    return json.loads(result.stdout)

def test_encoder_training_reproducible_across_processes():
    assert _run(1) == _run(42) == _run('random')

def test_encoder_training_repeatable_same_process_configuration():
    assert _run(7) == _run(7)
