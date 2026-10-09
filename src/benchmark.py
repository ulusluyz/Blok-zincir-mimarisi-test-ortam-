"""
Comprehensive Benchmark & Scalability Testing Framework.

Measures:
1. Top-1 and Top-3 accuracy on unseen Turkish test queries.
2. False Positive (FP) and False Negative (FN) rates.
3. Granular latency breakdown: Query Encoding, Block Evaluation, Top-K Selection, A+ Access.
4. Statistical significance (McNemar test / p-value) comparing Bitwise XOR vs Baselines.
5. Training vs inference costs.
6. Memory consumption (RAM in KB / MB).
7. A blocks evaluated vs A+ layers actually read.
8. Bit length comparison (64, 128, 256 bits).
9. Method comparison: XOR+POPCOUNT vs Binarized Dot vs Sign Cosine vs Keyword & TF-IDF Baselines.
10. Representation comparison: Learned vs Static/Random Binary Representation.
11. Scalability across 10, 50, 200, 1,000, 10,000, 100,000 blocks (real vs synthetic reported separately).
12. Pre-training vs Post-training score distributions.
"""

import sys
import time
import tracemalloc
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Any

from src.dataset import CATEGORIES, SMALL_TRAIN_DATA, SMALL_TEST_DATA, get_expanded_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine
from src.encoder import RandomLSHEncoder, TrainableBinaryEncoder
from src.baselines import KeywordBaselineMatcher, TFIDFBaselineMatcher
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork


def calculate_metrics(predictions: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    """
    Given predictions: list of (true_category, [ranked_predicted_categories]),
    calculates Top-1 accuracy, Top-3 accuracy, FP rate, FN rate, and boolean correctness array.
    """
    total = len(predictions)
    if total == 0:
        return {"top_1": 0.0, "top_3": 0.0, "fp_rate": 0.0, "fn_rate": 0.0, "correct_array": []}

    top_1_correct = 0
    top_3_correct = 0
    fp_count = 0
    fn_count = 0
    correct_array = []

    for true_cat, ranked_preds in predictions:
        top_1 = ranked_preds[0] if ranked_preds else None
        top_3 = ranked_preds[:3]

        if top_1 == true_cat:
            top_1_correct += 1
            correct_array.append(True)
        else:
            fp_count += 1
            fn_count += 1
            correct_array.append(False)

        if true_cat in top_3:
            top_3_correct += 1

    return {
        "top_1": top_1_correct / total,
        "top_3": top_3_correct / total,
        "fp_rate": fp_count / total,
        "fn_rate": fn_count / total,
        "correct_array": correct_array
    }


def compute_mcnemar_p_value(correct_a: List[bool], correct_b: List[bool]) -> float:
    """
    Computes McNemar's test p-value between two classifiers on the same test set.
    """
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and cb)

    if b + c == 0:
        return 1.0  # Identical predictions

    # McNemar chi-squared statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    # Survival function approximation for 1 degree of freedom chi2
    # p ≈ exp(-chi2 / 2) for simple approximation or error function
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return p_val


def run_benchmark() -> Dict[str, Any]:
    print("=== STARTING JULES POC BENCHMARK SUITE ===")
    results = {}

    train_data, test_data = get_expanded_dataset()
    train_texts = [text for text, _ in train_data]
    train_labels = [CATEGORIES.index(cat) for _, cat in train_data]

    # Feature Extractor
    fe = FeatureExtractor(max_features=512)
    fe.fit(train_texts)
    X_train = [fe.transform_dense(t) for t in train_texts]

    # 1. Baseline Evaluation
    print("\n--- 1. Evaluating Baselines ---")
    cat_keywords = {}
    for text, cat in train_data:
        if cat not in cat_keywords:
            cat_keywords[cat] = []
        cat_keywords[cat].append(text)
    kw_baseline = KeywordBaselineMatcher(cat_keywords)

    kw_preds = []
    kw_start = time.perf_counter()
    for q_text, true_cat in test_data:
        ranked = [cat for cat, _ in kw_baseline.rank_blocks(q_text)]
        kw_preds.append((true_cat, ranked))
    kw_latency = (time.perf_counter() - kw_start) / len(test_data)
    kw_metrics = calculate_metrics(kw_preds)
    results["baseline_keyword"] = {
        **kw_metrics,
        "avg_latency_ms": kw_latency * 1000.0
    }

    # TF-IDF Baseline
    tfidf_baseline = TFIDFBaselineMatcher(fe, cat_keywords)
    tfidf_preds = []
    tfidf_start = time.perf_counter()
    for q_text, true_cat in test_data:
        ranked = [cat for cat, _ in tfidf_baseline.rank_blocks(q_text)]
        tfidf_preds.append((true_cat, ranked))
    tfidf_latency = (time.perf_counter() - tfidf_start) / len(test_data)
    tfidf_metrics = calculate_metrics(tfidf_preds)
    results["baseline_tfidf"] = {
        **tfidf_metrics,
        "avg_latency_ms": tfidf_latency * 1000.0
    }

    # 2. Pre-training vs Post-training Score Distributions
    print("\n--- 2. Pre-training vs Post-training Analysis ---")
    pre_train_encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)

    # Train encoder
    train_start = time.perf_counter()
    post_train_encoder = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    post_train_encoder.fit(X_train, train_labels, epochs=30, lr=0.05)
    training_time = time.perf_counter() - train_start

    pre_scores_same = []
    pre_scores_diff = []
    post_scores_same = []
    post_scores_diff = []

    for q_text, true_cat in test_data:
        cat_idx = CATEGORIES.index(true_cat)
        q_dense = fe.transform_dense(q_text)

        # Pre-train
        Q_pre = pre_train_encoder.encode(q_dense)
        for c_i, proto in pre_train_encoder.category_prototypes.items():
            sim = similarity_xor_popcount(proto, Q_pre)
            if c_i == cat_idx:
                pre_scores_same.append(sim)
            else:
                pre_scores_diff.append(sim)

        # Post-train
        Q_post = post_train_encoder.encode(q_dense)
        for c_i, proto in post_train_encoder.category_prototypes.items():
            sim = similarity_xor_popcount(proto, Q_post)
            if c_i == cat_idx:
                post_scores_same.append(sim)
            else:
                post_scores_diff.append(sim)

    results["score_distribution"] = {
        "pre_train_same_category_avg": float(np.mean(pre_scores_same)),
        "pre_train_diff_category_avg": float(np.mean(pre_scores_diff)),
        "post_train_same_category_avg": float(np.mean(post_scores_same)),
        "post_train_diff_category_avg": float(np.mean(post_scores_diff)),
        "training_time_sec": training_time
    }

    # 3. Bit Length Comparison (64, 128, 256 bits) & Math Similarity Functions Comparison
    print("\n--- 3. Bit Length & Similarity Method Comparison ---")
    results["bit_length_experiments"] = {}

    for bit_len in [64, 128, 256]:
        enc = TrainableBinaryEncoder(input_dim=512, bit_length=bit_len, num_categories=len(CATEGORIES), seed=42)
        enc.fit(X_train, train_labels, epochs=30, lr=0.05)

        # Build Network
        net = BlockNetwork()
        a_plus_blocks = []
        for cat_i, cat_name in enumerate(CATEGORIES):
            proto = enc.category_prototypes[cat_i]
            ap = AddressBlock(f"ap_{cat_i}", payload={"cat": cat_name})
            kb = KnowledgeBlock(f"kb_{cat_i}", cat_name, proto, ap)
            net.add_block(kb)
            a_plus_blocks.append(ap)

        results["bit_length_experiments"][bit_len] = {}

        for math_method in ["xor", "binarized_dot", "sign_cosine"]:
            preds = []
            encoding_times = []
            eval_times = []
            select_times = []
            access_times = []
            total_times = []

            # Reset read counts
            for ap in a_plus_blocks:
                ap.read_count = 0

            for q_text, true_cat in test_data:
                t0 = time.perf_counter()
                q_dense = fe.transform_dense(q_text)
                Q = enc.encode(q_dense)
                t1 = time.perf_counter()

                res = net.process_query(Q, method=math_method, top_k=1)
                t2 = time.perf_counter()

                encoding_times.append(t1 - t0)
                eval_times.append(res["eval_latency_sec"])
                select_times.append(res["select_latency_sec"])
                access_times.append(res["read_latency_sec"])
                total_times.append(t2 - t0)

                ranked_cats = [item[1] for item in res["ranked_all_scores"]]
                preds.append((true_cat, ranked_cats))

            metrics = calculate_metrics(preds)
            total_a_plus_reads = sum(ap.read_count for ap in a_plus_blocks)

            # McNemar test against TF-IDF baseline
            p_val_vs_tfidf = compute_mcnemar_p_value(metrics["correct_array"], tfidf_metrics["correct_array"])

            results["bit_length_experiments"][bit_len][math_method] = {
                **metrics,
                "avg_encoding_time_ms": float(np.mean(encoding_times)) * 1000.0,
                "avg_eval_time_ms": float(np.mean(eval_times)) * 1000.0,
                "avg_select_time_ms": float(np.mean(select_times)) * 1000.0,
                "avg_access_time_ms": float(np.mean(access_times)) * 1000.0,
                "per_block_eval_time_us": (float(np.mean(eval_times)) / len(CATEGORIES)) * 1e6,
                "avg_total_time_ms": float(np.mean(total_times)) * 1000.0,
                "total_queries": len(test_data),
                "total_a_evaluated_per_query": len(CATEGORIES),
                "total_a_plus_read": total_a_plus_reads,
                "mcnemar_p_val_vs_tfidf": p_val_vs_tfidf,
            }

    # 4. Learned vs Random Representation Comparison
    print("\n--- 4. Learned vs Random Binary Representation ---")
    rand_enc = RandomLSHEncoder(input_dim=512, bit_length=128, seed=42)
    rand_preds = []

    # Random prototypes created by encoding category representative text
    rand_protos = {}
    for cat_i, cat_name in enumerate(CATEGORIES):
        cat_sample = [text for text, cat in train_data if cat == cat_name][0]
        rand_protos[cat_name] = rand_enc.encode(fe.transform_dense(cat_sample))

    for q_text, true_cat in test_data:
        Q_rand = rand_enc.encode(fe.transform_dense(q_text))
        scores = [
            (cat_name, similarity_xor_popcount(proto, Q_rand))
            for cat_name, proto in rand_protos.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        rand_preds.append((true_cat, [s[0] for s in scores]))

    results["learned_vs_random"] = {
        "random_lsh_128bit": calculate_metrics(rand_preds),
        "learned_ste_128bit": results["bit_length_experiments"][128]["xor"]
    }

    # 5. Scalability Benchmark (10, 50, 200, 1,000, 10,000, 100,000 blocks)
    print("\n--- 5. Scalability & Memory Benchmark across Block Counts ---")
    results["scalability"] = {}

    enc128 = TrainableBinaryEncoder(input_dim=512, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    enc128.fit(X_train, train_labels, epochs=30)

    # Real blocks: 6 categories
    real_blocks = []
    real_a_plus_list = []
    for cat_i, cat_name in enumerate(CATEGORIES):
        proto = enc128.category_prototypes[cat_i]
        ap = AddressBlock(f"real_ap_{cat_i}", payload={"category": cat_name, "is_synthetic": False})
        kb = KnowledgeBlock(f"real_kb_{cat_i}", cat_name, proto, ap)
        real_blocks.append(kb)
        real_a_plus_list.append(ap)

    # Test scale counts
    block_counts = [10, 50, 200, 1000, 10000, 100000]

    rng = random.Random(1234)

    for target_count in block_counts:
        tracemalloc.start()

        # Build scale network
        net_scale = BlockNetwork()
        for b in real_blocks:
            net_scale.add_block(b)

        synthetic_count = target_count - len(real_blocks)
        synth_a_plus_list = []

        if synthetic_count > 0:
            for s_i in range(synthetic_count):
                # Generate random synthetic 128-bit vector
                synth_bits = rng.getrandbits(128)
                synth_vec = BinaryVector(synth_bits, bit_length=128)
                synth_ap = AddressBlock(f"synth_ap_{s_i}", payload={"is_synthetic": True})
                synth_kb = KnowledgeBlock(f"synth_kb_{s_i}", f"Synthetic_{s_i % 10}", synth_vec, synth_ap)
                net_scale.add_block(synth_kb)
                synth_a_plus_list.append(synth_ap)

        current_ram_bytes, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Reset read counts
        for ap in real_a_plus_list:
            ap.read_count = 0
        for ap in synth_a_plus_list:
            ap.read_count = 0

        # Benchmark 20 queries on scaled network
        query_latencies = []
        eval_latencies = []
        select_latencies = []
        access_latencies = []
        real_top1_correct = 0

        sample_test_queries = test_data[:20]

        for q_text, true_cat in sample_test_queries:
            q_dense = fe.transform_dense(q_text)
            Q = enc128.encode(q_dense)

            t0 = time.perf_counter()
            res = net_scale.process_query(Q, method="xor", top_k=1)
            t1 = time.perf_counter()

            query_latencies.append((t1 - t0) * 1000.0)  # ms
            eval_latencies.append(res["eval_latency_sec"] * 1000.0)
            select_latencies.append(res["select_latency_sec"] * 1000.0)
            access_latencies.append(res["read_latency_sec"] * 1000.0)

            # Check if top activated block was the correct real category
            top_activated = res["activated_results"][0]
            if top_activated["category"] == true_cat:
                real_top1_correct += 1

        total_real_a_plus_read = sum(ap.read_count for ap in real_a_plus_list)
        total_synth_a_plus_read = sum(ap.read_count for ap in synth_a_plus_list)

        results["scalability"][target_count] = {
            "total_blocks": len(net_scale.blocks),
            "real_blocks": len(real_blocks),
            "synthetic_blocks": synthetic_count,
            "avg_query_latency_ms": float(np.mean(query_latencies)),
            "avg_eval_latency_ms": float(np.mean(eval_latencies)),
            "avg_select_latency_ms": float(np.mean(select_latencies)),
            "avg_access_latency_ms": float(np.mean(access_latencies)),
            "per_block_eval_time_ns": (float(np.mean(eval_latencies)) * 1e6) / len(net_scale.blocks),
            "memory_peak_mb": peak_ram_bytes / (1024.0 * 1024.0),
            "memory_per_block_bytes": peak_ram_bytes / len(net_scale.blocks) if len(net_scale.blocks) > 0 else 0,
            "real_category_top1_acc": real_top1_correct / len(sample_test_queries),
            "total_real_a_plus_read": total_real_a_plus_read,
            "total_synthetic_a_plus_read": total_synth_a_plus_read,
            "unaccessed_a_plus_count": len(net_scale.blocks) - (total_real_a_plus_read + total_synth_a_plus_read)
        }

        print(f"Scale {target_count:6d} blocks -> Latency: {np.mean(query_latencies):8.3f} ms | RAM Peak: {peak_ram_bytes / (1024*1024):6.2f} MB | Real A+ Read: {total_real_a_plus_read} | Synth A+ Read: {total_synth_a_plus_read}")

    return results


if __name__ == "__main__":
    run_benchmark()
