"""
Comprehensive Comparative Benchmark Framework for AG Learning Improvement.

Measures:
1. Mandatory Experiments A (Full Scan A), B (Oracle G), C (Learnable G Architectures).
2. Comparison of 5 Training Strategies (A-Only, G-Only, Simultaneous Joint, Staged, Contrastive Margin Joint).
3. Ablation Study on Sub-Prototype Count (M in {1, 2, 3, 4, 8}).
4. Scalability Benchmark across 1,000, 10,000, and 100,000 blocks.

Outputs results to AG_LEARNING_RESULTS.json.
"""

import time
import tracemalloc
import random
import math
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


def run_ag_improvement_benchmark(num_seeds: int = 2) -> Dict[str, Any]:
    print("=== STARTING AG LEARNING IMPROVEMENT BENCHMARK SUITE ===")
    results = {}

    train_data, single_test_data, multi_test_data = get_full_10cat_dataset()
    train_texts = [text for text, _ in train_data]

    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]
    y_train_categories = [[cat] for _, cat in train_data]

    # Helper function to evaluate network performance using the specific trained encoder
    def evaluate_network_with_encoder(h_net: HierarchicalBlockNetwork, encoder: TrainableBinaryEncoder) -> Dict[str, Any]:
        # Pre-encode single and multi test queries with the SPECIFIC trained encoder
        single_encoded = []
        for q_text, true_cat in single_test_data:
            q_dense = fe.transform_dense(q_text)
            Q = encoder.encode(q_dense)
            single_encoded.append((q_text, true_cat, Q))

        multi_encoded = []
        for q_text, expected_topics in multi_test_data:
            q_dense = fe.transform_dense(q_text)
            Q = encoder.encode(q_dense)
            multi_encoded.append((q_text, expected_topics, Q))

        # Single-topic evaluation
        st_top1_correct = 0
        st_top3_correct = 0
        st_group_recalled = 0
        st_total = len(single_encoded)

        for q_text, true_cat, Q in single_encoded:
            res = h_net.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="dynamic_multi_label", confidence_margin=0.08, block_top_k=3
            )
            active_cats = [item[1] for item in res["active_groups"]]
            if true_cat in active_cats:
                st_group_recalled += 1

            act_res = res["activated_results"]
            ranked_cats = list(dict.fromkeys([cat for _, cat, _ in res["ranked_candidate_a_scores"]]))
            if ranked_cats and ranked_cats[0] == true_cat:
                st_top1_correct += 1
            if true_cat in ranked_cats[:3]:
                st_top3_correct += 1

        # Multi-topic evaluation
        mt_expected_total = 0
        mt_found_total = 0
        mt_joint_success_count = 0
        mt_fp_count = 0
        mt_total_active = 0

        for q_text, expected_topics, Q in multi_encoded:
            res = h_net.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="dynamic_multi_label", confidence_margin=0.08
            )
            active_cats = [item[1] for item in res["active_groups"]]
            found = [t for t in expected_topics if t in active_cats]
            fps = [g for g in active_cats if g not in expected_topics]

            mt_expected_total += len(expected_topics)
            mt_found_total += len(found)
            mt_fp_count += len(fps)
            mt_total_active += len(active_cats)

            if len(found) == len(expected_topics):
                mt_joint_success_count += 1

        return {
            "single_topic_top1_acc": st_top1_correct / max(1, st_total),
            "single_topic_top3_acc": st_top3_correct / max(1, st_total),
            "single_topic_group_recall": st_group_recalled / max(1, st_total),
            "multi_topic_group_recall": mt_found_total / max(1, mt_expected_total),
            "multi_topic_joint_selection_recall": mt_joint_success_count / max(1, len(multi_encoded)),
            "multi_topic_fp_rate": mt_fp_count / max(1, mt_total_active)
        }

    # Base Encoder trained with Contrastive STE Loss
    enc_base = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    enc_base.fit(X_train, [CATEGORIES.index(c[0]) for c in y_train_categories], epochs=15, lr=0.05, contrastive_weight=0.6)

    # Build Knowledge Blocks for base encoder
    all_real_blocks = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:3]):
            vec = enc_base.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            all_real_blocks.append(kb)

    flat_net = BlockNetwork(all_real_blocks)

    # Pre-encode single test queries for base encoder
    base_single_test = []
    for q_text, true_cat in single_test_data:
        q_dense = fe.transform_dense(q_text)
        Q = enc_base.encode(q_dense)
        base_single_test.append((q_text, true_cat, Q))

    # Experiment A: Full Scan A
    exp_a_top1, exp_a_top3 = 0, 0
    exp_a_num = len(base_single_test)
    for q_text, true_cat, Q in base_single_test:
        res = flat_net.process_query(Q, method="xor", top_k=3)
        ranked_cats = list(dict.fromkeys([cat for _, cat, _ in res["ranked_all_scores"]]))
        if ranked_cats and ranked_cats[0] == true_cat:
            exp_a_top1 += 1
        if true_cat in ranked_cats[:3]:
            exp_a_top3 += 1

    results["experiment_a_full_scan"] = {
        "top1_acc": exp_a_top1 / exp_a_num,
        "top3_acc": exp_a_top3 / exp_a_num
    }

    # Experiment B: Oracle G
    exp_b_top1, exp_b_top3 = 0, 0
    for q_text, true_cat, Q in base_single_test:
        oracle_blocks = [b for b in all_real_blocks if b.category == true_cat]
        scores = [(b.category, similarity_xor_popcount(b.representation, Q)) for b in oracle_blocks]
        scores.sort(key=lambda x: x[1], reverse=True)
        if scores and scores[0][0] == true_cat:
            exp_b_top1 += 1
        if true_cat in [s[0] for s in scores[:3]]:
            exp_b_top3 += 1

    results["experiment_b_oracle_g"] = {
        "top1_acc": exp_b_top1 / exp_a_num,
        "top3_acc": exp_b_top3 / exp_a_num
    }

    # Experiment C: 4 G Architectures
    exp_c_archs = {}
    architectures = [
        ("static_g", "Static G"),
        ("learnable_single_g", "Learnable Single G"),
        ("learnable_multi_g", "Learnable Multi-Prototype G (M=3)"),
        ("dynamic_multi_g", "Dynamic Multi-Prototype G")
    ]

    for arch_key, arch_label in architectures:
        enc_arch = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
        g_reps = {}
        arch_blocks = []

        for cat_i, cat_name in enumerate(CATEGORIES):
            proto = enc_arch.category_prototypes[cat_i]
            if arch_key == "static_g":
                g_reps[cat_name] = StaticGRepresentative(f"g_{cat_i}", cat_name, proto)
            elif arch_key == "learnable_single_g":
                g_reps[cat_name] = LearnableSingleGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)
            elif arch_key == "learnable_multi_g":
                g_reps[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, num_sub_prototypes=3, bit_length=128, seed=42)
            elif arch_key == "dynamic_multi_g":
                g_reps[cat_name] = DynamicMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)

            cat_samples = [text for text, c in train_data if c == cat_name]
            for s_i, sample_text in enumerate(cat_samples[:3]):
                vec = enc_arch.encode(fe.transform_dense(sample_text))
                ap = AddressBlock(f"ap_{cat_name}_{s_i}")
                kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
                g_reps[cat_name].add_member_a_block(kb)
                arch_blocks.append(kb)

        trainer = JointAGTrainer(enc_arch, g_reps)
        trainer.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=15, lr=0.05)

        # Re-encode KnowledgeBlocks with trained encoder
        for kb in arch_blocks:
            sample_text = kb.address_block._payload.get("sample", "") if hasattr(kb.address_block, "_payload") else ""
            if sample_text:
                kb.representation = enc_arch.encode(fe.transform_dense(sample_text))

        h_net = HierarchicalBlockNetwork(list(g_reps.values()))
        h_net.register_flat_blocks(arch_blocks)
        exp_c_archs[arch_key] = evaluate_network_with_encoder(h_net, enc_arch)

    results["experiment_c_learnable_g"] = exp_c_archs

    # 2. Benchmark 5 Training Strategies
    print("\n--- 2. Benchmarking 5 Training Strategies ---")
    strategy_results = {}
    strategies = ["a_only", "g_only", "simultaneous_joint", "staged", "contrastive_joint"]

    for strat in strategies:
        enc_strat = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
        g_reps_strat = {}
        strat_blocks = []

        for i, cat_name in enumerate(CATEGORIES):
            g_reps_strat[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{i}", cat_name, num_sub_prototypes=3, bit_length=128, seed=42)
            cat_samples = [text for text, c in train_data if c == cat_name]
            for s_i, sample_text in enumerate(cat_samples[:3]):
                vec = enc_strat.encode(fe.transform_dense(sample_text))
                ap = AddressBlock(f"ap_{cat_name}_{s_i}", payload={"sample": sample_text})
                kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
                g_reps_strat[cat_name].add_member_a_block(kb)
                strat_blocks.append(kb)

        trainer_strat = JointAGTrainer(enc_strat, g_reps_strat)
        fit_res = trainer_strat.fit_strategy(X_train, y_train_categories, CATEGORIES, strategy=strat, epochs=15, lr=0.05)

        # Re-encode KnowledgeBlocks with the SPECIFIC trained encoder enc_strat
        for kb in strat_blocks:
            sample_text = kb.address_block._payload.get("sample", "")
            if sample_text:
                kb.representation = enc_strat.encode(fe.transform_dense(sample_text))

        h_net_strat = HierarchicalBlockNetwork(list(g_reps_strat.values()))
        h_net_strat.register_flat_blocks(strat_blocks)
        eval_metrics = evaluate_network_with_encoder(h_net_strat, enc_strat)

        strategy_results[strat] = {
            **eval_metrics,
            "encoder_w_drift": fit_res["encoder_w_drift_norm"],
            "average_g_drift": fit_res["average_g_drift"]
        }

    results["training_strategies_comparison"] = strategy_results

    # 3. Sub-prototype Ablation Study (M in {1, 2, 3, 4, 8})
    print("\n--- 3. Sub-prototype Ablation Study (M = 1, 2, 3, 4, 8) ---")
    ablation_results = {}
    for M in [1, 2, 3, 4, 8]:
        enc_m = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
        g_reps_m = {}
        m_blocks = []

        for i, cat_name in enumerate(CATEGORIES):
            g_reps_m[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{i}", cat_name, num_sub_prototypes=M, bit_length=128, seed=42)
            cat_samples = [text for text, c in train_data if c == cat_name]
            for s_i, sample_text in enumerate(cat_samples[:3]):
                vec = enc_m.encode(fe.transform_dense(sample_text))
                ap = AddressBlock(f"ap_{cat_name}_{s_i}", payload={"sample": sample_text})
                kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
                g_reps_m[cat_name].add_member_a_block(kb)
                m_blocks.append(kb)

        trainer_m = JointAGTrainer(enc_m, g_reps_m)
        trainer_m.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=15, lr=0.05)

        for kb in m_blocks:
            sample_text = kb.address_block._payload.get("sample", "")
            if sample_text:
                kb.representation = enc_m.encode(fe.transform_dense(sample_text))

        h_net_m = HierarchicalBlockNetwork(list(g_reps_m.values()))
        h_net_m.register_flat_blocks(m_blocks)
        ablation_results[M] = evaluate_network_with_encoder(h_net_m, enc_m)

    results["sub_prototype_ablation"] = ablation_results

    # 4. Scalability Benchmark (1,000, 10,000, 100,000 blocks)
    print("\n--- 4. Scalability Benchmark across 1,000, 10,000, 100,000 blocks ---")
    scale_results = {}
    rng = random.Random(42)

    for target_count in [1000, 10000, 100000]:
        tracemalloc.start()
        enc_scale = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
        g_reps_scale = {
            cat_name: DynamicMultiPrototypeGRepresentative(f"g_{i}", cat_name, bit_length=128, seed=42)
            for i, cat_name in enumerate(CATEGORIES)
        }

        all_scale_blocks = []
        for i in range(target_count):
            cat_name = CATEGORIES[i % len(CATEGORIES)]
            vec = BinaryVector(rng.getrandbits(128), bit_length=128)
            ap = AddressBlock(f"scale_ap_{i}")
            kb = KnowledgeBlock(f"scale_kb_{i}", cat_name, vec, ap)
            g_reps_scale[cat_name].add_member_a_block(kb)
            all_scale_blocks.append(kb)

        h_net_scale = HierarchicalBlockNetwork(list(g_reps_scale.values()))
        h_net_scale.register_flat_blocks(all_scale_blocks)
        flat_net_scale = BlockNetwork(all_scale_blocks)

        curr_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        full_lats, g_lats = [], []
        test_samples = base_single_test[:20]

        for q_text, true_cat, Q in test_samples:
            t0 = time.perf_counter()
            res_full = flat_net_scale.process_query(Q, method="xor", top_k=1)
            t1 = time.perf_counter()
            full_lats.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            res_g = h_net_scale.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="top_k", group_top_k=2, block_top_k=1
            )
            t3 = time.perf_counter()
            g_lats.append((t3 - t2) * 1000.0)

        avg_full = float(np.mean(full_lats))
        avg_g = float(np.mean(g_lats))

        scale_results[target_count] = {
            "total_blocks": target_count,
            "full_scan_latency_ms": avg_full,
            "learnable_g_latency_ms": avg_g,
            "p95_g_latency_ms": float(np.percentile(g_lats, 95)),
            "speedup": avg_full / avg_g if avg_g > 0 else 1.0,
            "peak_ram_mb": peak_ram / (1024.0 * 1024.0)
        }

        print(f"  Scale {target_count:6d} blocks -> Full: {avg_full:8.3f} ms | Learnable G: {avg_g:8.3f} ms (Speedup: {avg_full/avg_g:5.1f}x) | Peak RAM: {peak_ram/(1024*1024):6.2f} MB")

    results["scalability"] = scale_results

    # Save to AG_LEARNING_RESULTS.json
    with open("AG_LEARNING_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved benchmark results to AG_LEARNING_RESULTS.json!")
    return results


if __name__ == "__main__":
    run_ag_improvement_benchmark()
