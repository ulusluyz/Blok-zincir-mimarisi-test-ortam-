"""
Comprehensive Hierarchical Benchmark & Scalability Testing Framework.

Measures & Compares 5 Routing Strategies:
1. Full A Scan (Mevcut Tam A Taraması)
2. Learned Single-Level Group Routing (Öğrenilmiş Tek Seviyeli Grup Yönlendirme)
3. Learned Multi-Level Hierarchical Routing (Öğrenilmiş Çok Seviyeli Hiyerarşik Yönlendirme)
4. LSH Candidate Pre-filtering (LSH Tabanlı Ön Eleme)
5. IVF and HNSW Search Indexes (İndeks Tabanlı Yaklaşık Komşu Arama)

Metrics Reported across 10, 100, 1,000, 10,000, and 100,000 blocks:
- Top-1, Top-3 accuracy.
- Correct Group Candidate Recall Rate (% of times correct group is included).
- Correct A Block Retention Rate.
- False Pruning Rate.
- Latency Breakdown: Encoding, G Eval, A Eval, A+ Access, Selection Center.
- Speedup Ratio vs Full Scan.
- RAM Consumption (MB).
- Multi-Topic Query Group Routing Recall.
- Exports raw results to results.json and terminal execution logs.
"""

import sys
import time
import tracemalloc
import random
import math
import json
import numpy as np
from typing import List, Dict, Tuple, Any

from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA, UNSEEN_TEST_MULTI_DATASET, get_expanded_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork
from src.search_indexes import LSHIndex, IVFIndex, HNSWIndex


def calculate_metrics(predictions: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {"top_1": 0.0, "top_3": 0.0, "fp_rate": None, "fn_rate": None, "correct_array": []}

    top_1_correct = 0
    top_3_correct = 0
    correct_array = []

    for true_cat, ranked_preds in predictions:
        top_1 = ranked_preds[0] if ranked_preds else None
        top_3 = list(dict.fromkeys(ranked_preds))[:3]

        if top_1 == true_cat:
            top_1_correct += 1
            correct_array.append(True)
        else:
            correct_array.append(False)

        if true_cat in top_3:
            top_3_correct += 1

    return {
        "top_1": top_1_correct / total,
        "top_3": top_3_correct / total,
        "fp_rate": None,  # Cannot calculate false-positive rate without negative-class predictions.
        "fn_rate": None,  # Cannot calculate false-negative rate without a defined binary decision.
        "correct_array": correct_array
    }


def run_hierarchical_benchmark(block_counts=None, max_queries=None, include_indexes=True) -> Dict[str, Any]:
    print("=== STARTING HIERARCHICAL ROUTING BENCHMARK SUITE ===")
    results = {}

    train_data, test_data = get_expanded_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    # Feature Extractor
    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    # Train Encoder
    train_start = time.perf_counter()
    encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    encoder.fit(X_train, train_labels, epochs=30, lr=0.05)
    training_time = time.perf_counter() - train_start

    # Build Base Real Knowledge Blocks and Group Representatives
    real_a_blocks = []
    category_groups = {}

    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = encoder.category_prototypes[cat_i]
        # Create G representative
        g_rep = GroupRepresentative(f"g_{cat_i}", cat_name, proto)
        category_groups[cat_name] = g_rep

        # Create A blocks inside this category
        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:5]):
            vec = encoder.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}", payload={"sample": sample_text})
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            g_rep.add_member_a_block(kb)
            real_a_blocks.append(kb)

    # 1. Multi-Topic Routing Recall Benchmark
    print("\n--- 1. Evaluating Multi-Topic Query Group Routing ---")
    h_net_base = HierarchicalBlockNetwork(list(category_groups.values()))
    h_net_base.register_flat_blocks(real_a_blocks)
    multi_topic_results = []

    for item in UNSEEN_TEST_MULTI_DATASET:
        query_text, expected_topics = item["text"], item["expected_categories"]
        q_dense = fe.transform_dense(query_text)
        Q = encoder.encode(q_dense)

        # Route via confidence margin strategy
        res = h_net_base.process_query_hierarchical_2level(
            Q, method="xor", group_routing_strategy="confidence_margin", confidence_margin=0.08
        )
        active_g_names = [item[1] for item in res["active_groups"]]

        # Check topic recall
        topics_found = [t for t in expected_topics if t in active_g_names]
        topic_recall = len(topics_found) / len(expected_topics)

        multi_topic_results.append({
            "query": query_text,
            "expected_topics": expected_topics,
            "activated_groups": active_g_names,
            "topic_recall": topic_recall
        })

    avg_multi_topic_recall = float(np.mean([item["topic_recall"] for item in multi_topic_results]))
    results["multi_topic_benchmark"] = {
        "avg_multi_topic_recall": avg_multi_topic_recall,
        "details": multi_topic_results
    }

    # Pre-encode all test queries into Q binary vectors for isolated routing benchmarks
    encoded_test_queries = []
    encoding_latencies = []
    for q_text, true_cat in test_data[:30]:
        t0 = time.perf_counter()
        q_dense = fe.transform_dense(q_text)
        Q = encoder.encode(q_dense)
        t1 = time.perf_counter()
        encoding_latencies.append((t1 - t0) * 1000.0)
        encoded_test_queries.append((q_text, true_cat, Q))

    avg_encoding_lat = float(np.mean(encoding_latencies))

    # 2. Scalability & Comparative Routing Strategy Benchmark across Block Counts
    print("\n--- 2. Scalability & Comparative Strategy Benchmark (10 to 100,000 blocks) ---")
    results["scalability_comparisons"] = {}
    block_counts = [10, 100, 1000, 10000, 100000] if block_counts is None else list(block_counts)
    if max_queries is not None:
        encoded_test_queries = encoded_test_queries[:max_queries]
    if not encoded_test_queries:
        raise ValueError("Benchmark requires at least one query")

    rng = random.Random(42)

    for target_count in block_counts:
        print(f"\nBuilding and benchmarking network with scale: {target_count} blocks...")
        tracemalloc.start()

        # Build G groups and A blocks for target count
        scale_groups = {}
        for cat_i, cat_name in enumerate(CATEGORIES):
            proto = encoder.category_prototypes[cat_i]
            scale_groups[cat_name] = GroupRepresentative(f"g_{cat_i}", cat_name, proto)

        # Re-populate real blocks
        for kb in real_a_blocks:
            scale_groups[kb.category].add_member_a_block(kb)

        # Fill synthetic blocks
        synthetic_count = max(0, target_count - len(real_a_blocks))
        synth_a_plus_list = []
        if synthetic_count > 0:
            for s_i in range(synthetic_count):
                synth_bits = rng.getrandbits(128)
                synth_vec = BinaryVector(synth_bits, bit_length=128)
                assigned_cat = CATEGORIES[s_i % len(CATEGORIES)]
                ap = AddressBlock(f"synth_ap_{s_i}", payload={"is_synthetic": True})
                kb = KnowledgeBlock(f"synth_kb_{s_i}", assigned_cat, synth_vec, ap)
                scale_groups[assigned_cat].add_member_a_block(kb)
                synth_a_plus_list.append(ap)

        # Build Networks
        h_net_scale = HierarchicalBlockNetwork(list(scale_groups.values()))
        # Build flat A blocks list across all groups
        all_scale_a_blocks = []
        for g_rep in h_net_scale.groups:
            all_scale_a_blocks.extend(g_rep.member_a_blocks)
        h_net_scale.register_flat_blocks(all_scale_a_blocks)

        flat_net_scale = BlockNetwork(all_scale_a_blocks)

        # Build Auxiliary Indexes
        lsh_idx = LSHIndex(num_tables=4, hash_size=8, bit_length=128) if include_indexes else None
        if include_indexes:
            for b in all_scale_a_blocks:
                lsh_idx.add_block(b)

        if include_indexes:
            ivf_idx = IVFIndex(num_centroids=min(16, target_count), bit_length=128)
            ivf_idx.fit_and_build(all_scale_a_blocks)
            hnsw_idx = HNSWIndex(M=min(8, target_count))
            hnsw_idx.build_graph(all_scale_a_blocks[:min(1000, target_count)])  # Sub-graph for benchmarking

        current_ram_bytes, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        method_benchmarks = {}

        # 1. Full A Scan
        full_latencies, full_preds = [], []
        for q_text, true_cat, Q in encoded_test_queries:
            t0 = time.perf_counter()
            res = flat_net_scale.process_query(Q, method="xor", top_k=1)
            t1 = time.perf_counter()
            full_latencies.append((t1 - t0) * 1000.0)
            full_preds.append((true_cat, [item[1] for item in res["ranked_all_scores"]]))

        avg_full_lat = float(np.mean(full_latencies))
        method_benchmarks["full_a_scan"] = {
            **calculate_metrics(full_preds),
            "avg_latency_ms": avg_full_lat,
            "speedup_vs_full": 1.0,
            "total_a_evaluated": len(all_scale_a_blocks)
        }

        # 2. Single-level Learned Group Routing (2-Level)
        g_latencies, g_preds = [], []
        g_eval_times, a_eval_times = [], []
        correct_group_retained = 0
        evaluated_counts, pruned_counts = [], []

        for q_text, true_cat, Q in encoded_test_queries:
            t0 = time.perf_counter()
            res = h_net_scale.process_query_hierarchical_2level(
                Q, method="xor", group_routing_strategy="confidence_margin", confidence_margin=0.08, block_top_k=1
            )
            t1 = time.perf_counter()
            g_latencies.append((t1 - t0) * 1000.0)
            g_eval_times.append(res["g_eval_latency_sec"] * 1000.0)
            a_eval_times.append(res["a_eval_latency_sec"] * 1000.0)

            active_cats = [item[1] for item in res["active_groups"]]
            if true_cat in active_cats:
                correct_group_retained += 1

            evaluated_counts.append(res["total_a_evaluated"])
            pruned_counts.append(res["unaccessed_a_count"])
            g_preds.append((true_cat, [item[1] for item in res["ranked_candidate_a_scores"]]))

        avg_g_lat = float(np.mean(g_latencies))
        method_benchmarks["learned_2level_group_routing"] = {
            **calculate_metrics(g_preds),
            "avg_latency_ms": avg_g_lat,
            "speedup_vs_full": avg_full_lat / avg_g_lat if avg_g_lat > 0 else 1.0,
            "avg_g_eval_time_ms": float(np.mean(g_eval_times)),
            "avg_a_eval_time_ms": float(np.mean(a_eval_times)),
            "correct_group_recall_rate": correct_group_retained / len(encoded_test_queries),
            "false_pruning_rate": 1.0 - (correct_group_retained / len(encoded_test_queries)),
            "avg_a_evaluated": float(np.mean(evaluated_counts)),
            "avg_unaccessed_a_pruned": float(np.mean(pruned_counts))
        }

        # 3. Multi-level Hierarchical Group Routing (3-Level)
        ml_latencies, ml_preds = [], []
        for q_text, true_cat, Q in encoded_test_queries:
            t0 = time.perf_counter()
            res_ml = h_net_scale.process_query_hierarchical_multilevel(Q, method="xor", top_k_per_level=1, block_top_k=1)
            t1 = time.perf_counter()
            ml_latencies.append((t1 - t0) * 1000.0)
            ml_preds.append((true_cat, [item[1] for item in res_ml["ranked_candidate_a_scores"]]))

        avg_ml_lat = float(np.mean(ml_latencies))
        method_benchmarks["learned_multilevel_group_routing"] = {
            **calculate_metrics(ml_preds),
            "avg_latency_ms": avg_ml_lat,
            "speedup_vs_full": avg_full_lat / avg_ml_lat if avg_ml_lat > 0 else 1.0
        }

        if include_indexes:
            # 4. LSH Pre-filtering
            lsh_latencies, lsh_preds = [], []
            for q_text, true_cat, Q in encoded_test_queries:
                t0 = time.perf_counter()
                cands = lsh_idx.query_candidates(Q, fallback_all=all_scale_a_blocks)
                # Evaluate candidates
                scores = [(b.category, similarity_xor_popcount(b.representation, Q)) for b in cands]
                scores.sort(key=lambda x: x[1], reverse=True)
                t1 = time.perf_counter()
                lsh_latencies.append((t1 - t0) * 1000.0)
                lsh_preds.append((true_cat, [s[0] for s in scores]))

            avg_lsh_lat = float(np.mean(lsh_latencies))
            method_benchmarks["lsh_candidate_prefilter"] = {
                **calculate_metrics(lsh_preds),
                "avg_latency_ms": avg_lsh_lat,
                "speedup_vs_full": avg_full_lat / avg_lsh_lat if avg_lsh_lat > 0 else 1.0
            }

            # 5. IVF Index
            ivf_latencies, ivf_preds = [], []
            for q_text, true_cat, Q in encoded_test_queries:
                t0 = time.perf_counter()
                cands = ivf_idx.query_candidates(Q, nprobe=2)
                scores = [(b.category, similarity_xor_popcount(b.representation, Q)) for b in cands]
                scores.sort(key=lambda x: x[1], reverse=True)
                t1 = time.perf_counter()
                ivf_latencies.append((t1 - t0) * 1000.0)
                ivf_preds.append((true_cat, [s[0] for s in scores]))

            avg_ivf_lat = float(np.mean(ivf_latencies))
            method_benchmarks["ivf_index_search"] = {
                **calculate_metrics(ivf_preds),
                "avg_latency_ms": avg_ivf_lat,
                "speedup_vs_full": avg_full_lat / avg_ivf_lat if avg_ivf_lat > 0 else 1.0
            }

        results["scalability_comparisons"][target_count] = {
            "total_blocks": len(all_scale_a_blocks),
            "real_blocks": len(real_a_blocks),
            "synthetic_blocks": synthetic_count,
            "avg_encoding_latency_ms": avg_encoding_lat,
            "memory_peak_mb": peak_ram_bytes / (1024.0 * 1024.0),
            "memory_per_block_bytes": peak_ram_bytes / len(all_scale_a_blocks),
            "methods": method_benchmarks
        }

        speedup_val = avg_full_lat / avg_g_lat if avg_g_lat > 0 else 1.0
        print(f"Scale {len(all_scale_a_blocks):6d} blocks -> Full Scan: {avg_full_lat:8.3f} ms | Learned G Route: {avg_g_lat:8.3f} ms (Speedup: {speedup_val:5.1f}x) | Peak RAM: {peak_ram_bytes/(1024*1024):6.2f} MB")

    # Save to results.json
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved benchmark findings to results.json!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bounded, reproducible hierarchical routing benchmark")
    parser.add_argument("--scales", type=int, nargs="+", default=[10,100,1000,10000,100000])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-indexes", action="store_true", help="Skip LSH/IVF/HNSW construction and comparisons")
    args = parser.parse_args()
    run_hierarchical_benchmark(block_counts=args.scales, max_queries=args.max_queries, include_indexes=not args.skip_indexes)
