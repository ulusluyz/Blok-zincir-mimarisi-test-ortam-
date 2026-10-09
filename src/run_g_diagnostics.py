"""
Mandatory Comparative Diagnostic Experiments Script (Experiments A, B, and C).
Executes:
- Experiment A: Full Scan A (No G Routing)
- Experiment B: Oracle G (0% Routing Error Ground Truth G Routing)
- Experiment C: Learnable G Routing (Static G, Learnable Single G, Learnable Multi G M=3, Dynamic Multi G)

Exports diagnostic findings to G_DIAGNOSTIC_RESULTS.json and G_DIAGNOSTIC_RAW_LOGS.txt.
"""

import sys
import os
import io
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Any

from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork
from src.learnable_g import (
    StaticGRepresentative,
    LearnableSingleGRepresentative,
    LearnableMultiPrototypeGRepresentative,
    DynamicMultiPrototypeGRepresentative,
    JointAGTrainer
)
from src.hierarchical_blocks import HierarchicalBlockNetwork


def run_g_diagnostics(num_seeds: int = 3) -> Dict[str, Any]:
    print("=== STARTING MANDATORY DIAGNOSTIC EXPERIMENTS (A, B, C) ===")
    diagnostic_results = {}

    train_data, single_test_data, multi_test_data = get_full_10cat_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]
    y_train_categories = [[cat] for _, cat in train_data]

    # Pre-encode test queries using trained encoder
    enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    enc.fit(X_train, train_labels, epochs=15, lr=0.05)

    encoded_single_test = []
    for q_text, true_cat in single_test_data:
        q_dense = fe.transform_dense(q_text)
        Q = enc.encode(q_dense)
        encoded_single_test.append((q_text, true_cat, Q))

    encoded_multi_test = []
    for q_text, expected_topics in multi_test_data:
        q_dense = fe.transform_dense(q_text)
        Q = enc.encode(q_dense)
        encoded_multi_test.append((q_text, expected_topics, Q))

    # Build Knowledge Blocks and Groups
    category_groups = {}
    all_real_a_blocks = []

    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = enc.category_prototypes[cat_i]
        g_rep = DynamicMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)
        category_groups[cat_name] = g_rep

        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:3]):
            vec = enc.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            g_rep.add_member_a_block(kb)
            all_real_a_blocks.append(kb)

    h_net = HierarchicalBlockNetwork(list(category_groups.values()))
    h_net.register_flat_blocks(all_real_a_blocks)
    flat_net = BlockNetwork(all_real_a_blocks)

    # Train G representatives
    trainer = JointAGTrainer(enc, category_groups)
    trainer.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=15, lr=0.05)

    # -------------------------------------------------------------------------
    # EXPERIMENT A: FULL SCAN A (No G Routing)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment A: Full Scan A (No G Routing) ---")
    exp_a_latencies = []
    exp_a_top1 = 0
    exp_a_top3 = 0
    exp_a_ranks = []
    exp_a_num = len(encoded_single_test)

    for q_text, true_cat, Q in encoded_single_test:
        t0 = time.perf_counter()
        res = flat_net.process_query(Q, method="xor", top_k=len(all_real_a_blocks))
        t1 = time.perf_counter()
        exp_a_latencies.append((t1 - t0) * 1000.0)

        ranked_cats = [item[1] for item in res["ranked_all_scores"]]
        top1 = ranked_cats[0] if ranked_cats else None
        top3 = ranked_cats[:3]

        if top1 == true_cat:
            exp_a_top1 += 1
        if true_cat in top3:
            exp_a_top3 += 1

        # Calculate exact rank of first correct A block
        rank = ranked_cats.index(true_cat) + 1 if true_cat in ranked_cats else len(ranked_cats)
        exp_a_ranks.append(rank)

    diagnostic_results["experiment_a_full_scan"] = {
        "top1_accuracy": exp_a_top1 / exp_a_num,
        "top1_numerator": exp_a_top1,
        "top1_denominator": exp_a_num,
        "top3_accuracy": exp_a_top3 / exp_a_num,
        "top3_numerator": exp_a_top3,
        "top3_denominator": exp_a_num,
        "avg_rank_of_correct_block": float(np.mean(exp_a_ranks)),
        "avg_latency_ms": float(np.mean(exp_a_latencies)),
        "total_a_evaluated": len(all_real_a_blocks)
    }

    # -------------------------------------------------------------------------
    # EXPERIMENT B: ORACLE G (0% Routing Error Ground Truth Routing)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment B: Oracle G (Ground-Truth Routing) ---")
    exp_b_latencies = []
    exp_b_top1 = 0
    exp_b_top3 = 0
    exp_b_ranks = []
    exp_b_a_eval_counts = []

    for q_text, true_cat, Q in encoded_single_test:
        t0 = time.perf_counter()
        # Oracle G: Directly evaluate ONLY A blocks belonging to true_cat group
        oracle_g = category_groups[true_cat]
        oracle_a_blocks = oracle_g.member_a_blocks

        scores = [
            (a_blk, a_blk.evaluate_suitability(Q, similarity_xor_popcount))
            for a_blk in oracle_a_blocks
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        t1 = time.perf_counter()

        exp_b_latencies.append((t1 - t0) * 1000.0)
        exp_b_a_eval_counts.append(len(oracle_a_blocks))

        ranked_cats = [b.category for b, _ in scores]
        top1 = ranked_cats[0] if ranked_cats else None
        top3 = ranked_cats[:3]

        if top1 == true_cat:
            exp_b_top1 += 1
        if true_cat in top3:
            exp_b_top3 += 1

        rank = ranked_cats.index(true_cat) + 1 if true_cat in ranked_cats else len(ranked_cats)
        exp_b_ranks.append(rank)

    diagnostic_results["experiment_b_oracle_g"] = {
        "top1_accuracy": exp_b_top1 / exp_a_num,
        "top1_numerator": exp_b_top1,
        "top1_denominator": exp_a_num,
        "top3_accuracy": exp_b_top3 / exp_a_num,
        "top3_numerator": exp_b_top3,
        "top3_denominator": exp_a_num,
        "avg_rank_of_correct_block": float(np.mean(exp_b_ranks)),
        "avg_latency_ms": float(np.mean(exp_b_latencies)),
        "avg_a_evaluated": float(np.mean(exp_b_a_eval_counts))
    }

    # -------------------------------------------------------------------------
    # EXPERIMENT C: LEARNABLE G (Normal G Routing Across 4 Architectures)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment C: Learnable G Routing Across 4 Architectures ---")
    exp_c_results = {}

    architectures = [
        ("static_g", "Static G"),
        ("learnable_single_g", "Learnable Single G"),
        ("learnable_multi_g", "Learnable Multi-Prototype G (M=3)"),
        ("dynamic_multi_g", "Dynamic Multi-Prototype G")
    ]

    for arch_key, arch_label in architectures:
        g_reps = {}
        for cat_i, cat_name in enumerate(CATEGORIES):
            proto = enc.category_prototypes[cat_i]
            if arch_key == "static_g":
                g_reps[cat_name] = StaticGRepresentative(f"g_{cat_i}", cat_name, proto)
            elif arch_key == "learnable_single_g":
                g_reps[cat_name] = LearnableSingleGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)
            elif arch_key == "learnable_multi_g":
                g_reps[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, num_sub_prototypes=3, bit_length=128, seed=42)
            elif arch_key == "dynamic_multi_g":
                g_reps[cat_name] = DynamicMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)

            for s_i, kb in enumerate(category_groups[cat_name].member_a_blocks):
                g_reps[cat_name].add_member_a_block(kb)

        trainer_c = JointAGTrainer(enc, g_reps)
        trainer_c.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=15, lr=0.05)

        h_net_c = HierarchicalBlockNetwork(list(g_reps.values()))
        h_net_c.register_flat_blocks(all_real_a_blocks)

        # Single-topic metrics
        st_top1, st_top3, st_group_recalled = 0, 0, 0
        c_latencies = []

        for q_text, true_cat, Q in encoded_single_test:
            t0 = time.perf_counter()
            res = h_net_c.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="dynamic_multi_label", confidence_margin=0.08, block_top_k=3
            )
            t1 = time.perf_counter()
            c_latencies.append((t1 - t0) * 1000.0)

            active_cats = [item[1] for item in res["active_groups"]]
            if true_cat in active_cats:
                st_group_recalled += 1

            act_res = res["activated_results"]
            ranked_cats = [b["category"] for b in act_res]
            if ranked_cats and ranked_cats[0] == true_cat:
                st_top1 += 1
            if true_cat in ranked_cats[:3]:
                st_top3 += 1

        # Multi-topic metrics
        mt_expected, mt_found, mt_joint_success, mt_fps, mt_active_total = 0, 0, 0, 0, 0

        for q_text, expected_topics, Q in encoded_multi_test:
            res = h_net_c.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="dynamic_multi_label", confidence_margin=0.08, block_top_k=3
            )
            active_cats = [item[1] for item in res["active_groups"]]
            found = [t for t in expected_topics if t in active_cats]
            fps = [g for g in active_cats if g not in expected_topics]

            mt_expected += len(expected_topics)
            mt_found += len(found)
            mt_fps += len(fps)
            mt_active_total += len(active_cats)

            if len(found) == len(expected_topics):
                mt_joint_success += 1

        exp_c_results[arch_key] = {
            "single_topic_top1_accuracy": st_top1 / exp_a_num,
            "single_topic_top1_numerator": st_top1,
            "single_topic_top1_denominator": exp_a_num,
            "single_topic_top3_accuracy": st_top3 / exp_a_num,
            "single_topic_group_recall": st_group_recalled / exp_a_num,
            "multi_topic_group_recall": mt_found / max(1, mt_expected),
            "multi_topic_joint_selection_recall": mt_joint_success / max(1, len(encoded_multi_test)),
            "multi_topic_precision": mt_found / max(1, mt_active_total),
            "multi_topic_fp_rate": mt_fps / max(1, mt_active_total),
            "avg_latency_ms": float(np.mean(c_latencies))
        }

    diagnostic_results["experiment_c_learnable_g"] = exp_c_results

    # Save findings
    with open("G_DIAGNOSTIC_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)

    print("\nSaved diagnostic findings to G_DIAGNOSTIC_RESULTS.json!")
    return diagnostic_results


if __name__ == "__main__":
    run_g_diagnostics()
