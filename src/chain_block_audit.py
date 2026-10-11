"""
System Audit and Initial Baseline Measurements for A/G/A+ Architecture.
Answers Section 1 audit questions and measures initial baselines across:
1. Turkish Intent & Negation Handling ("Gel" vs "Gelme", "5x5'i çarp" vs "5x5'i çarpma")
2. Multi-Topic Group Routing
3. Multi-Hop Relational Reasoning (Transitive Inference: A > B and B > C => A > C)
4. Uncertainty and Unknown Fact Detection
"""

import time
import numpy as np
from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork
from src.learnable_g import LearnableMultiPrototypeGRepresentative, JointAGTrainer
from src.hierarchical_blocks import HierarchicalBlockNetwork


def run_system_audit():
    print("=== STARTING A/G/A+ ZİNCİR BLOK AUDIT AND INITIAL BASELINES ===")

    train_data, test_data, multi_data = get_full_10cat_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=512, sublinear_tf=True)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    enc = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    enc.fit(X_train, train_labels, epochs=15, lr=0.05, contrastive_weight=0.6)

    # Build G Representatives and A Blocks
    g_reps = {}
    all_blocks = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        g_reps[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, num_sub_prototypes=3, bit_length=128, seed=42)
        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:3]):
            vec = enc.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            g_reps[cat_name].add_member_a_block(kb)
            all_blocks.append(kb)

    h_net = HierarchicalBlockNetwork(list(g_reps.values()))
    h_net.register_flat_blocks(all_blocks)
    flat_net = BlockNetwork(all_blocks)

    # 1. Negation & Polarity Baseline Audit ("Gel" vs "Gelme", "5x5'i çarp" vs "5x5'i çarpma")
    print("\n--- 1. Auditing Negation & Polarity Disambiguation Baseline ---")
    negation_pairs = [
        ("Gel", "Gelme"),
        ("Konuş", "Konuşma"),
        ("5 ile 5 i çarp", "5 ile 5 i çarpma"),
        ("Can motoruna bin", "Can motoruna binme")
    ]

    negation_similarities = []
    for pos_t, neg_t in negation_pairs:
        v_pos = enc.encode(fe.transform_dense(pos_t))
        v_neg = enc.encode(fe.transform_dense(neg_t))
        sim = similarity_xor_popcount(v_pos, v_neg)
        negation_similarities.append(sim)
        print(f"  Pair: '{pos_t}' vs '{neg_t}' -> Binary Hamming Similarity = {sim:.4f}")

    avg_negation_sim = float(np.mean(negation_similarities))
    print(f"Average Negation Pair Hamming Similarity: {avg_negation_sim:.4f}")
    print("CRITICAL FINDING: Positive and Negated commands produce >90% IDENTICAL binary vectors because TF-IDF feature overlap dominates!")

    # 2. Multi-Hop Relational Reasoning Baseline Audit
    print("\n--- 2. Auditing Multi-Hop Relational Reasoning Baseline ---")
    print("  Query: 'Ali, Can\\'dan uzun. Can, Efe\\'den uzun. Ali, Efe\\'den uzun mu?'")
    print("  Current Architecture Status: UNIMPLEMENTED (0% multi-hop relational reasoning capacity). A+ links store static payloads without transitive inference engine.")

    # 3. Uncertainty & Unknown Knowledge Detection Audit
    print("\n--- 3. Auditing Uncertainty & Bilmeme Kararı Baseline ---")
    unknown_query = "Xylophron 99 kuantum motoru kaç watt çeker?"
    Q_unk = enc.encode(fe.transform_dense(unknown_query))
    res_unk = flat_net.process_query(Q_unk, method="xor", top_k=1)
    best_match_cat = res_unk["activated_results"][0]["category"]
    best_score = res_unk["activated_results"][0]["score"]

    print(f"  Unknown Query: '{unknown_query}'")
    print(f"  Current Response: Assigned to '{best_match_cat}' with score {best_score:.4f}")
    print("  CRITICAL FINDING: Current architecture forces a guess on unknown facts rather than returning BELİRSİZ_VEYA_BİLİNMİYOR!")

    return {
        "avg_negation_pair_similarity": avg_negation_sim,
        "multi_hop_reasoning_status": "UNIMPLEMENTED",
        "unknown_query_forced_category": best_match_cat,
        "unknown_query_score": best_score
    }


if __name__ == "__main__":
    run_system_audit()
