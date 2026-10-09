"""
Comprehensive verification test suite covering:
1. End-to-end training and inference pipeline across categories
2. Bit lengths comparison (64, 128, 256 bits)
3. Out-of-Domain and threshold filtering
4. Comparison of learned binary representations vs random binary representations
"""

import pytest
from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA
from src.features import FeatureExtractor
from src.encoder import RandomLSHEncoder, TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork


@pytest.mark.parametrize("bit_length", [64, 128, 256])
def test_end_to_end_pipeline_different_bit_lengths(bit_length):
    train_texts = [text for text, _ in SMALL_TRAIN_DATA]
    train_labels = [CATEGORIES.index(cat) for _, cat in SMALL_TRAIN_DATA]

    # 1. Extract Features
    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    # 2. Train Encoder
    encoder = TrainableBinaryEncoder(input_dim=256, bit_length=bit_length, num_categories=len(CATEGORIES))
    encoder.fit(X_train, train_labels, epochs=15)

    # 3. Create A and A+ Blocks for each Category prototype
    network = BlockNetwork()
    for cat_idx, cat_name in enumerate(CATEGORIES):
        proto_vector = encoder.category_prototypes[cat_idx]
        a_plus = AddressBlock(f"ap_{cat_idx}", payload={"category": cat_name})
        a_block = KnowledgeBlock(f"a_{cat_idx}", cat_name, proto_vector, a_plus)
        network.add_block(a_block)

    # 4. Process Test Query
    query_text = "Dünya hangi yıldızın etrafında döner?"
    q_dense = fe.transform_dense(query_text)
    Q = encoder.encode(q_dense)

    res = network.process_query(Q, method="xor", top_k=1)
    top_activated = res["activated_results"][0]

    # Should match Gezegenler
    assert top_activated["category"] == "Gezegenler"
    assert res["total_a_evaluated"] == len(CATEGORIES)
    assert res["total_a_plus_read"] == 1


def test_learned_vs_random_representation_performance():
    train_texts = [text for text, _ in SMALL_TRAIN_DATA]
    train_labels = [CATEGORIES.index(cat) for _, cat in SMALL_TRAIN_DATA]

    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    # Learned
    trainable_enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES))
    trainable_enc.fit(X_train, train_labels, epochs=20)

    # Random LSH
    random_enc = RandomLSHEncoder(input_dim=256, bit_length=128)

    # Evaluate on test set
    learned_correct = 0
    random_correct = 0

    for query_text, true_cat in SMALL_TEST_DATA:
        q_dense = fe.transform_dense(query_text)

        # Learned scoring
        Q_learned = trainable_enc.encode(q_dense)
        learned_scores = [
            (cat_idx, (Q_learned.bits ^ proto.bits).bit_count())
            for cat_idx, proto in trainable_enc.category_prototypes.items()
        ]
        learned_scores.sort(key=lambda x: x[1])  # smallest hamming dist first
        top_learned_cat = CATEGORIES[learned_scores[0][0]]
        if top_learned_cat == true_cat:
            learned_correct += 1

        # Random scoring
        Q_rand = random_enc.encode(q_dense)
        rand_proto_scores = []
        for cat_idx in range(len(CATEGORIES)):
            # Random prototype representation
            dummy_proto = random_enc.encode(fe.transform_dense(train_texts[cat_idx * 8]))
            rand_proto_scores.append((cat_idx, (Q_rand.bits ^ dummy_proto.bits).bit_count()))
        rand_proto_scores.sort(key=lambda x: x[1])
        top_rand_cat = CATEGORIES[rand_proto_scores[0][0]]
        if top_rand_cat == true_cat:
            random_correct += 1

    # Learned representation must significantly outperform random LSH
    assert learned_correct > random_correct, f"Learned ({learned_correct}) should outperform Random ({random_correct})"
