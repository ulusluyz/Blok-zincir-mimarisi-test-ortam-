"""
Verification Script for Sparse Activation and Method Invocation Counters.
Verifies real G evaluation_count, A evaluation_count, and A+ read_count across different query types.
"""

import random
from src.dataset import CATEGORIES, get_expanded_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def audit_sparse_counters():
    train_data, test_data = get_expanded_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    encoder.fit(X_train, train_labels, epochs=30, lr=0.05)

    # Build 100,000 Scaled Network with 6 Groups
    scale_groups = {}
    all_a_blocks = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = encoder.category_prototypes[cat_i]
        scale_groups[cat_name] = GroupRepresentative(f"g_{cat_i}", cat_name, proto)

    rng = random.Random(42)
    # Add 100,000 blocks distributed across 6 groups (~16,666 blocks per group)
    for i in range(100000):
        cat_name = CATEGORIES[i % len(CATEGORIES)]
        vec = BinaryVector(rng.getrandbits(128), bit_length=128)
        ap = AddressBlock(f"ap_{i}")
        kb = KnowledgeBlock(f"kb_{i}", cat_name, vec, ap)
        scale_groups[cat_name].add_member_a_block(kb)
        all_a_blocks.append(kb)

    h_net = HierarchicalBlockNetwork(list(scale_groups.values()))
    h_net.register_flat_blocks(all_a_blocks)

    print("=== AUDITING 100,000 SCALED NETWORK COUNTERS ===")

    # Reset all counters
    for g in h_net.groups:
        g.evaluation_count = 0
    for a in all_a_blocks:
        a.evaluation_count = 0
        a.address_block.read_count = 0

    # Test Query 1: Single-Topic High Confidence (Top-1 Group Active)
    q1_text = "Dünya hangi yıldızın etrafında döner?"
    Q1 = encoder.encode(fe.transform_dense(q1_text))

    res1 = h_net.process_query_hierarchical_2level(
        Q1, method="xor", group_routing_strategy="top_k", group_top_k=1, block_top_k=1
    )

    g_eval_real_1 = sum(g.evaluation_count for g in h_net.groups)
    a_eval_real_1 = sum(a.evaluation_count for a in all_a_blocks)
    ap_read_real_1 = sum(a.address_block.read_count for a in all_a_blocks)

    print(f"Query 1 ('{q1_text}') [Strategy: Top-1 Group]:")
    print(f"  Network Total G Groups: {len(h_net.groups)}")
    print(f"  Network Total A Blocks: {len(all_a_blocks)}")
    print(f"  Real G Evaluated Count: {g_eval_real_1}")
    print(f"  Real A Evaluated Count: {a_eval_real_1} (Elenen / Değerlendirilmeyen: {len(all_a_blocks) - a_eval_real_1})")
    print(f"  Real A+ Read Count     : {ap_read_real_1}")
    print(f"  Unselected G Member A Method Invocations: {sum(a.evaluation_count for g in h_net.groups if g.group_id != res1['active_groups'][0][0] for a in g.member_a_blocks)}\n")

    # Reset counters
    for g in h_net.groups:
        g.evaluation_count = 0
    for a in all_a_blocks:
        a.evaluation_count = 0
        a.address_block.read_count = 0

    # Test Query 2: Fallback Triggered (3 Groups Active)
    q2_text = "Transistörün sıcaklıkla değişen davranışını matematiksel türevle açıkla"
    Q2 = encoder.encode(fe.transform_dense(q2_text))

    res2 = h_net.process_query_hierarchical_2level(
        Q2, method="xor", group_routing_strategy="confidence_margin", confidence_margin=0.08, block_top_k=1
    )

    g_eval_real_2 = sum(g.evaluation_count for g in h_net.groups)
    a_eval_real_2 = sum(a.evaluation_count for a in all_a_blocks)
    ap_read_real_2 = sum(a.address_block.read_count for a in all_a_blocks)

    print(f"Query 2 ('{q2_text}') [Strategy: Margin Fallback]:")
    print(f"  Fallback Triggered    : {res2['fallback_triggered']} ({res2['fallback_reason']})")
    print(f"  Active G Groups Count : {res2['active_g_count']}")
    print(f"  Real G Evaluated Count: {g_eval_real_2}")
    print(f"  Real A Evaluated Count: {a_eval_real_2} (Elenen / Değerlendirilmeyen: {len(all_a_blocks) - a_eval_real_2})")
    print(f"  Real A+ Read Count     : {ap_read_real_2}")

if __name__ == "__main__":
    audit_sparse_counters()
