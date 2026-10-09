"""
Test script to evaluate Full Scan A performance with Supervised Contrastive STE Encoder.
"""

from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.features import FeatureExtractor
from src.bitwise import similarity_xor_popcount
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork

def test_full_scan_contrastive():
    train_data, test_data, _ = get_full_10cat_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    # Train Encoder with Contrastive Loss & Hard Negative Mining
    enc = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    enc.fit(X_train, train_labels, epochs=30, lr=0.05, contrastive_weight=0.6)

    # Build Knowledge Blocks for each category sample
    all_blocks = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:3]):
            vec = enc.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            all_blocks.append(kb)

    flat_net = BlockNetwork(all_blocks)

    top1_correct = 0
    top3_correct = 0
    total = len(test_data)

    for q_text, true_cat in test_data:
        q_dense = fe.transform_dense(q_text)
        Q = enc.encode(q_dense)
        res = flat_net.process_query(Q, method="xor", top_k=3)
        # Deduplicate categories in ranked order
        ranked_cats = list(dict.fromkeys([cat for _, cat, _ in res["ranked_all_scores"]]))

        if ranked_cats and ranked_cats[0] == true_cat:
            top1_correct += 1
        if true_cat in ranked_cats[:3]:
            top3_correct += 1

    top1_acc = top1_correct / total
    top3_acc = top3_correct / total
    print(f"=== FULL SCAN A CONTRASTIVE ENCODER EVALUATION ===")
    print(f"Categories Count   : {len(CATEGORIES)}")
    print(f"Total Test Queries : {total}")
    print(f"Top-1 Accuracy     : {top1_acc * 100:.2f}% ({top1_correct}/{total})")
    print(f"Top-3 Accuracy     : {top3_acc * 100:.2f}% ({top3_correct}/{total})")

if __name__ == "__main__":
    test_full_scan_contrastive()
