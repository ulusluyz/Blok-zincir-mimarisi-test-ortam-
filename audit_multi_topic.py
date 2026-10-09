"""
Detailed Multi-Topic & Single-Topic Recall Audit Script.
"""

from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA, MULTI_TOPIC_TEST_DATA, get_expanded_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork

def audit_recalls():
    train_data, test_data = get_expanded_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    encoder.fit(X_train, train_labels, epochs=30, lr=0.05)

    category_groups = {}
    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = encoder.category_prototypes[cat_i]
        g_rep = GroupRepresentative(f"g_{cat_i}", cat_name, proto)
        category_groups[cat_name] = g_rep

        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:5]):
            vec = encoder.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            g_rep.add_member_a_block(kb)

    h_net = HierarchicalBlockNetwork(list(category_groups.values()))

    print("=== SINGLE-TOPIC GROUP RECALL AUDIT ===")
    single_topic_correct = 0
    single_topic_total = len(test_data)
    for q_text, true_cat in test_data:
        q_dense = fe.transform_dense(q_text)
        Q = encoder.encode(q_dense)
        res = h_net.process_query_hierarchical_2level(Q, method="xor", group_routing_strategy="confidence_margin")
        active_cats = [item[1] for item in res["active_groups"]]
        if true_cat in active_cats:
            single_topic_correct += 1

    single_topic_recall = single_topic_correct / single_topic_total
    print(f"Single-Topic Group Recall: {single_topic_recall * 100:.2f}% ({single_topic_correct}/{single_topic_total})\n")

    print("=== MULTI-TOPIC GROUP RECALL AUDIT ===")
    multi_details = []

    # 1. Top-1 Only (Before Fallback)
    all_expected_before_fallback = 0
    found_before_fallback = 0

    # 2. Flexible Margin / Fallback
    all_expected_after_fallback = 0
    found_after_fallback = 0
    all_groups_jointly_selected = 0
    total_missing_groups = 0
    total_false_positive_groups = 0
    total_activated_groups = 0

    for idx, (query_text, expected_topics) in enumerate(MULTI_TOPIC_TEST_DATA):
        q_dense = fe.transform_dense(query_text)
        Q = encoder.encode(q_dense)

        # Before fallback (Top-1 group only)
        res_top1 = h_net.process_query_hierarchical_2level(
            Q, method="xor", group_routing_strategy="top_k", group_top_k=1
        )
        selected_top1 = [item[1] for item in res_top1["active_groups"]]
        found_top1 = [t for t in expected_topics if t in selected_top1]
        all_expected_before_fallback += len(expected_topics)
        found_before_fallback += len(found_top1)

        # After fallback / flexible margin
        res_flex = h_net.process_query_hierarchical_2level(
            Q, method="xor", group_routing_strategy="confidence_margin", confidence_margin=0.08
        )
        selected_flex = [item[1] for item in res_flex["active_groups"]]
        found_flex = [t for t in expected_topics if t in selected_flex]
        missing_flex = [t for t in expected_topics if t not in selected_flex]
        fp_flex = [g for g in selected_flex if g not in expected_topics]

        all_expected_after_fallback += len(expected_topics)
        found_after_fallback += len(found_flex)
        total_missing_groups += len(missing_flex)
        total_false_positive_groups += len(fp_flex)
        total_activated_groups += len(selected_flex)

        is_joint_success = (len(found_flex) == len(expected_topics))
        if is_joint_success:
            all_groups_jointly_selected += 1

        print(f"Query {idx+1}: '{query_text}'")
        print(f"  Expected Topics : {expected_topics}")
        print(f"  Top-1 Selected  : {selected_top1} -> Found {len(found_top1)}/{len(expected_topics)}")
        print(f"  Flex Selected   : {selected_flex}")
        print(f"  Missing Topics  : {missing_flex}")
        print(f"  False Positives : {fp_flex}\n")

    recall_before = found_before_fallback / all_expected_before_fallback
    recall_after = found_after_fallback / all_expected_after_fallback
    joint_selection_rate = all_groups_jointly_selected / len(MULTI_TOPIC_TEST_DATA)
    fp_rate = total_false_positive_groups / total_activated_groups if total_activated_groups > 0 else 0.0

    print("=== SUMMARY METRICS ===")
    print(f"Multi-Topic Recall (Before Fallback - Top-1 Only): {recall_before * 100:.2f}%")
    print(f"Multi-Topic Recall (After Fallback - Flex Margin): {recall_after * 100:.2f}%")
    print(f"Joint Selection Success Rate (All Topics Found): {joint_selection_rate * 100:.2f}%")
    print(f"Total Missing Expected Topics: {total_missing_groups}")
    print(f"False Positive Group Activation Rate: {fp_rate * 100:.2f}%")

if __name__ == "__main__":
    audit_recalls()
