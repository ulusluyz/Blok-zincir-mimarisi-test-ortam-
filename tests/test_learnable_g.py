"""
Unit tests for Learnable G Representatives and Joint A/G Training Strategies.
"""

import pytest
from src.bitwise import BinaryVector
from src.features import FeatureExtractor
from src.encoder import TrainableBinaryEncoder
from src.learnable_g import (
    StaticGRepresentative,
    LearnableSingleGRepresentative,
    LearnableMultiPrototypeGRepresentative,
    DynamicMultiPrototypeGRepresentative,
    JointAGTrainer
)


def test_learnable_single_g_representative():
    g = LearnableSingleGRepresentative("g1", "Elektronik", bit_length=64)
    Q = BinaryVector(0b11110000, bit_length=64)

    s0 = g.evaluate_suitability(Q)
    assert 0.0 <= s0 <= 1.0

    q_bools = Q.to_bool_list()
    g.update_gradients(q_bools, is_target=True, lr=0.1)

    s1 = g.evaluate_suitability(Q)
    assert g.evaluation_count == 2


def test_learnable_multi_prototype_g_representative():
    g = LearnableMultiPrototypeGRepresentative("g1", "Elektronik", num_sub_prototypes=3, bit_length=64)
    Q = BinaryVector(0b11110000, bit_length=64)

    s0 = g.evaluate_suitability(Q, aggregation="max")
    assert 0.0 <= s0 <= 1.0
    assert len(g.sub_prototypes) == 3

    g.update_gradients(Q.to_bool_list(), is_target=True, lr=0.1)


def test_dynamic_multi_prototype_g_representative():
    g = DynamicMultiPrototypeGRepresentative("g1", "Elektronik", bit_length=64, max_sub_prototypes=4)
    assert len(g.sub_prototypes) == 1

    g.sub_errors[0] = 5.0
    g.adapt_topology(split_error_threshold=2.0)
    assert len(g.sub_prototypes) == 2


@pytest.mark.parametrize("strategy", [
    "a_only", "g_only", "simultaneous_joint", "staged", "contrastive_joint"
])
def test_five_training_strategies(strategy):
    texts = ["merhaba selam", "dünya güneş gezegen", "bc337 transistör devre", "12 çarpı 8"]
    labels = [["Selamlaşma"], ["Gezegenler"], ["Elektronik"], ["Matematik"]]
    all_cats = ["Selamlaşma", "Gezegenler", "Elektronik", "Matematik"]

    fe = FeatureExtractor(max_features=64)
    fe.fit(texts)
    X = [fe.transform_dense(t) for t in texts]

    encoder = TrainableBinaryEncoder(input_dim=64, bit_length=64, num_categories=len(all_cats))

    g_reps = {
        "Selamlaşma": LearnableSingleGRepresentative("g0", "Selamlaşma", bit_length=64),
        "Gezegenler": LearnableMultiPrototypeGRepresentative("g1", "Gezegenler", num_sub_prototypes=2, bit_length=64),
        "Elektronik": DynamicMultiPrototypeGRepresentative("g2", "Elektronik", bit_length=64),
        "Matematik": LearnableSingleGRepresentative("g3", "Matematik", bit_length=64)
    }

    trainer = JointAGTrainer(encoder, g_reps)
    res = trainer.fit_strategy(X, labels, all_cats, strategy=strategy, epochs=6, lr=0.05)

    assert res["strategy"] == strategy
    if strategy == "a_only":
        assert res["encoder_w_drift_norm"] > 0.0
        assert res["average_g_drift"] == 0.0
    elif strategy == "g_only":
        assert res["encoder_w_drift_norm"] == 0.0
        assert res["average_g_drift"] > 0.0
    else:
        assert res["encoder_w_drift_norm"] > 0.0
        assert res["average_g_drift"] > 0.0
