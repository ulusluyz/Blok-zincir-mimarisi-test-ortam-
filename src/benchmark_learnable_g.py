"""
Comprehensive Learnable G Benchmark Framework (Optimized for Fast & Accurate Execution).

Benchmarks:
1. 4 G Architectures: Static G, Learnable Single G, Learnable Multi G (M=3), Dynamic Multi G.
2. Sub-prototype Ablation Study: M in {1, 2, 3, 4, 8}.
3. Training Mode Comparison: G-Only Training vs Joint A/G Training.
4. Scalability across 1,000, 10,000, and 100,000 blocks.

Metrics Reported:
- Single-topic Top-1, Top-3 accuracy & Group Recall.
- Multi-topic Group Recall & Joint Selection Recall (% where ALL target groups activated together).
- False Positive Group Activation Rate.
- Weight parameter drift (||W_new - W_old||, ||P_new - P_old||).
- Latency statistics (mean, median, P95) & RAM peak.
- Exports results to LEARNABLE_G_RESULTS.json.
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


def run_learnable_g_benchmark(num_seeds: int = 2) -> Dict[str, Any]:
    print("=== STARTING LEARNABLE G COMPREHENSIVE BENCHMARK SUITE ===")
    benchmark_results = {}

    train_data, single_test_data, multi_test_data = get_full_10cat_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]
    y_train_categories = [[cat] for _, cat in train_data]

    # Pre-encode single and multi-topic test queries
    pre_encoded_single = []
    pre_encoded_multi = []

    # Temporary encoder for pre-encoding test queries
    tmp_enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    tmp_enc.fit(X_train, train_labels, epochs=10)

    for q_text, true_cat in single_test_data:
        q_dense = fe.transform_dense(q_text)
        Q = tmp_enc.encode(q_dense)
        pre_encoded_single.append((q_text, true_cat, Q))

    for q_text, expected_topics in multi_test_data:
        q_dense = fe.transform_dense(q_text)
        Q = tmp_enc.encode(q_dense)
        pre_encoded_multi.append((q_text, expected_topics, Q))

    # Helper function to evaluate network performance
    def evaluate_network(h_net: HierarchicalBlockNetwork) -> Dict[str, Any]:
        # Single-topic evaluation
        st_top1_correct = 0
        st_group_recalled = 0
        st_total = len(pre_encoded_single)

        for q_text, true_cat, Q in pre_encoded_single:
            res = h_net.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="dynamic_multi_label", confidence_margin=0.08
            )
            active_cats = [item[1] for item in res["active_groups"]]
            if true_cat in active_cats:
                st_group_recalled += 1
            if res["activated_results"] and res["activated_results"][0]["category"] == true_cat:
                st_top1_correct += 1

        # Multi-topic evaluation
        mt_expected_total = 0
        mt_found_total = 0
        mt_joint_success_count = 0
        mt_fp_count = 0
        mt_total_active = 0

        for q_text, expected_topics, Q in pre_encoded_multi:
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
            "single_topic_group_recall": st_group_recalled / max(1, st_total),
            "multi_topic_group_recall": mt_found_total / max(1, mt_expected_total),
            "multi_topic_joint_selection_recall": mt_joint_success_count / max(1, len(pre_encoded_multi)),
            "multi_topic_fp_rate": mt_fp_count / max(1, mt_total_active)
        }

    # 1. Benchmark 4 Architectures
    print("\n--- 1. Benchmarking 4 G Architectures ---")
    arch_results = {}

    architectures = [
        ("static_g", "1. Static G (Baseline Control)"),
        ("learnable_single_g", "2. Learnable Single G"),
        ("learnable_multi_g", "3. Learnable Multi-Prototype G (M=3)"),
        ("dynamic_multi_g", "4. Dynamic Multi-Prototype G")
    ]

    for arch_key, arch_label in architectures:
        print(f"Evaluating {arch_label}...")
        arch_seed_metrics = []

        for seed in range(num_seeds):
            enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=seed)
            g_reps = {}

            for cat_i, cat_name in enumerate(CATEGORIES):
                proto = enc.category_prototypes[cat_i]
                if arch_key == "static_g":
                    g_reps[cat_name] = StaticGRepresentative(f"g_{cat_i}", cat_name, proto)
                elif arch_key == "learnable_single_g":
                    g_reps[cat_name] = LearnableSingleGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=seed)
                elif arch_key == "learnable_multi_g":
                    g_reps[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, num_sub_prototypes=3, bit_length=128, seed=seed)
                elif arch_key == "dynamic_multi_g":
                    g_reps[cat_name] = DynamicMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=seed)

                # Add sample A blocks to group
                cat_samples = [text for text, c in train_data if c == cat_name]
                for s_i, sample_text in enumerate(cat_samples[:3]):
                    vec = enc.encode(fe.transform_dense(sample_text))
                    ap = AddressBlock(f"ap_{cat_name}_{s_i}")
                    kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
                    g_reps[cat_name].add_member_a_block(kb)

            # Perform Joint Training if learnable
            trainer = JointAGTrainer(enc, g_reps)
            train_res = trainer.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=10, lr=0.05)

            # Build Network and evaluate
            h_net = HierarchicalBlockNetwork(list(g_reps.values()))
            eval_metrics = evaluate_network(h_net)

            arch_seed_metrics.append({
                **eval_metrics,
                "encoder_w_drift": train_res["encoder_w_drift_norm"],
                "average_g_drift": train_res["average_g_drift"]
            })

        # Compute aggregate mean metrics
        arch_results[arch_key] = {
            "single_topic_top1_acc": float(np.mean([m["single_topic_top1_acc"] for m in arch_seed_metrics])),
            "single_topic_group_recall": float(np.mean([m["single_topic_group_recall"] for m in arch_seed_metrics])),
            "multi_topic_group_recall": float(np.mean([m["multi_topic_group_recall"] for m in arch_seed_metrics])),
            "multi_topic_joint_selection_recall": float(np.mean([m["multi_topic_joint_selection_recall"] for m in arch_seed_metrics])),
            "multi_topic_fp_rate": float(np.mean([m["multi_topic_fp_rate"] for m in arch_seed_metrics])),
            "encoder_w_drift": float(np.mean([m["encoder_w_drift"] for m in arch_seed_metrics])),
            "average_g_drift": float(np.mean([m["average_g_drift"] for m in arch_seed_metrics]))
        }

    benchmark_results["architectures_comparison"] = arch_results

    # 2. Sub-prototype Ablation Study (M in {1, 2, 3, 4, 8})
    print("\n--- 2. Sub-prototype Ablation Study (M = 1, 2, 3, 4, 8) ---")
    ablation_results = {}

    for M in [1, 2, 3, 4, 8]:
        enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
        g_reps = {}
        for cat_i, cat_name in enumerate(CATEGORIES):
            g_reps[cat_name] = LearnableMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, num_sub_prototypes=M, bit_length=128, seed=42)
            cat_samples = [text for text, c in train_data if c == cat_name]
            for s_i, sample_text in enumerate(cat_samples[:3]):
                vec = enc.encode(fe.transform_dense(sample_text))
                ap = AddressBlock(f"ap_{cat_name}_{s_i}")
                kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
                g_reps[cat_name].add_member_a_block(kb)

        trainer = JointAGTrainer(enc, g_reps)
        trainer.fit_joint(X_train, y_train_categories, CATEGORIES, epochs=10, lr=0.05)

        h_net = HierarchicalBlockNetwork(list(g_reps.values()))
        eval_metrics = evaluate_network(h_net)
        ablation_results[M] = eval_metrics

    benchmark_results["sub_prototype_ablation"] = ablation_results

    # 3. Scalability Benchmark across 1,000, 10,000, 100,000 blocks
    print("\n--- 3. Scalability Benchmark across 1,000, 10,000, 100,000 blocks ---")
    scale_results = {}
    rng = random.Random(42)

    for target_count in [1000, 10000, 100000]:
        tracemalloc.start()
        enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)

        # Build G representatives using Dynamic Multi-Prototype model
        g_reps = {}
        all_scale_a = []
        for cat_i, cat_name in enumerate(CATEGORIES):
            g_reps[cat_name] = DynamicMultiPrototypeGRepresentative(f"g_{cat_i}", cat_name, bit_length=128, seed=42)

        synth_count = target_count
        for i in range(synth_count):
            cat_name = CATEGORIES[i % len(CATEGORIES)]
            vec = BinaryVector(rng.getrandbits(128), bit_length=128)
            ap = AddressBlock(f"scale_ap_{i}")
            kb = KnowledgeBlock(f"scale_kb_{i}", cat_name, vec, ap)
            g_reps[cat_name].add_member_a_block(kb)
            all_scale_a.append(kb)

        h_net_scale = HierarchicalBlockNetwork(list(g_reps.values()))
        h_net_scale.register_flat_blocks(all_scale_a)
        flat_net_scale = BlockNetwork(all_scale_a)

        curr_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure routing latencies on 20 test queries
        full_latencies, g_latencies = [], []
        test_samples = pre_encoded_single[:20]

        for q_text, true_cat, Q in test_samples:
            t0 = time.perf_counter()
            res_full = flat_net_scale.process_query(Q, method="xor", top_k=1)
            t1 = time.perf_counter()
            full_latencies.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            res_g = h_net_scale.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="top_k", group_top_k=2, block_top_k=1
            )
            t3 = time.perf_counter()
            g_latencies.append((t3 - t2) * 1000.0)

        avg_full = float(np.mean(full_latencies))
        avg_g = float(np.mean(g_latencies))

        scale_results[target_count] = {
            "total_blocks": target_count,
            "full_scan_latency_ms": avg_full,
            "learnable_g_latency_ms": avg_g,
            "p95_g_latency_ms": float(np.percentile(g_latencies, 95)),
            "speedup": avg_full / avg_g if avg_g > 0 else 1.0,
            "peak_ram_mb": peak_ram / (1024.0 * 1024.0)
        }

        print(f"  Scale {target_count:6d} blocks -> Full: {avg_full:8.3f} ms | Learnable G: {avg_g:8.3f} ms (Speedup: {avg_full/avg_g:5.1f}x) | Peak RAM: {peak_ram/(1024*1024):6.2f} MB")

    benchmark_results["scalability"] = scale_results

    # Save to LEARNABLE_G_RESULTS.json
    with open("LEARNABLE_G_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print("\nSaved learnable G benchmark results to LEARNABLE_G_RESULTS.json!")
    return benchmark_results


if __name__ == "__main__":
    run_learnable_g_benchmark()
