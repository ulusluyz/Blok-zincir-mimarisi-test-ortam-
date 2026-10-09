"""
Statistical Scalability Benchmark Script.
Executes repeated benchmark runs across different random seeds to calculate:
Mean, Median, Std Dev, P95, P99, Min, Max latencies and RAM peak.
"""

import time
import tracemalloc
import random
import json
import numpy as np
from typing import List, Dict, Any

from src.dataset import CATEGORIES, get_expanded_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def run_statistical_benchmark(num_seeds: int = 5, queries_per_seed: int = 10) -> Dict[str, Any]:
    print(f"=== RUNNING STATISTICAL BENCHMARK (Seeds: {num_seeds}, Queries/Seed: {queries_per_seed}) ===")

    train_data, test_data = get_expanded_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    encoder.fit(X_train, train_labels, epochs=30, lr=0.05)

    base_groups = {}
    real_a_blocks = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = encoder.category_prototypes[cat_i]
        base_groups[cat_name] = GroupRepresentative(f"g_{cat_i}", cat_name, proto)
        cat_samples = [text for text, c in train_data if c == cat_name]
        for s_i, sample_text in enumerate(cat_samples[:5]):
            vec = encoder.encode(fe.transform_dense(sample_text))
            ap = AddressBlock(f"ap_{cat_name}_{s_i}")
            kb = KnowledgeBlock(f"kb_{cat_name}_{s_i}", cat_name, vec, ap)
            base_groups[cat_name].add_member_a_block(kb)
            real_a_blocks.append(kb)

    block_counts = [10, 100, 1000, 10000, 100000]
    statistical_results = {}

    for target_count in block_counts:
        print(f"\nBenchmarking target scale: {target_count} blocks across {num_seeds} seeds...")

        full_scan_latencies = []
        learned_g_latencies = []
        ram_peaks = []
        a_eval_counts = []
        a_pruned_counts = []

        for seed in range(num_seeds):
            tracemalloc.start()
            rng = random.Random(seed * 1000 + 42)

            # Build G groups and A blocks for target count
            scale_groups = {}
            for cat_i, cat_name in enumerate(CATEGORIES):
                proto = encoder.category_prototypes[cat_i]
                scale_groups[cat_name] = GroupRepresentative(f"g_{cat_i}", cat_name, proto)

            for kb in real_a_blocks:
                scale_groups[kb.category].add_member_a_block(kb)

            synth_count = max(0, target_count - len(real_a_blocks))
            for s_i in range(synth_count):
                synth_vec = BinaryVector(rng.getrandbits(128), bit_length=128)
                assigned_cat = CATEGORIES[s_i % len(CATEGORIES)]
                ap = AddressBlock(f"synth_ap_{s_i}")
                kb = KnowledgeBlock(f"synth_kb_{s_i}", assigned_cat, synth_vec, ap)
                scale_groups[assigned_cat].add_member_a_block(kb)

            h_net = HierarchicalBlockNetwork(list(scale_groups.values()))
            all_a = []
            for g in h_net.groups:
                all_a.extend(g.member_a_blocks)
            h_net.register_flat_blocks(all_a)
            flat_net = BlockNetwork(all_a)

            curr_ram, peak_ram = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_peaks.append(peak_ram / (1024.0 * 1024.0))

            # Test queries
            test_samples = test_data[:queries_per_seed]
            for q_text, true_cat in test_samples:
                q_dense = fe.transform_dense(q_text)
                Q = encoder.encode(q_dense)

                # Full scan timing
                t0 = time.perf_counter()
                res_full = flat_net.process_query(Q, method="xor", top_k=1)
                t1 = time.perf_counter()
                full_scan_latencies.append((t1 - t0) * 1000.0)

                # Learned G route timing
                t2 = time.perf_counter()
                res_g = h_net.process_query_hierarchical_2level(
                    Q, method="xor", group_routing_strategy="top_k", group_top_k=1, block_top_k=1
                )
                t3 = time.perf_counter()
                learned_g_latencies.append((t3 - t2) * 1000.0)

                a_eval_counts.append(res_g["total_a_evaluated"])
                a_pruned_counts.append(res_g["unaccessed_a_count"])

        # Compute statistics helper
        def calc_stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std_dev": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "sample_count": len(arr)
            }

        total_a_size = len(all_a)
        avg_a_eval = float(np.mean(a_eval_counts))
        avg_a_pruned = float(np.mean(a_pruned_counts))

        statistical_results[target_count] = {
            "total_blocks": total_a_size,
            "real_blocks": len(real_a_blocks),
            "synthetic_blocks": max(0, target_count - len(real_a_blocks)),
            "pruning_efficiency_percent": (avg_a_pruned / total_a_size) * 100.0,
            "avg_a_evaluated": avg_a_eval,
            "avg_a_pruned": avg_a_pruned,
            "peak_ram_mb": calc_stats(ram_peaks),
            "full_scan_latency_ms": calc_stats(full_scan_latencies),
            "learned_g_latency_ms": calc_stats(learned_g_latencies),
            "speedup_mean": calc_stats(full_scan_latencies)["mean"] / calc_stats(learned_g_latencies)["mean"]
        }

        print(f"  Scale {target_count:6d} blocks -> Full Scan Mean: {np.mean(full_scan_latencies):8.3f} ms | G Route Mean: {np.mean(learned_g_latencies):8.3f} ms (Speedup: {statistical_results[target_count]['speedup_mean']:5.1f}x) | RAM Mean: {np.mean(ram_peaks):6.2f} MB")

    return statistical_results


if __name__ == "__main__":
    stats = run_statistical_benchmark(num_seeds=3, queries_per_seed=10)
    with open("VERIFIED_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\nSaved statistical verification metrics to VERIFIED_RESULTS.json!")
