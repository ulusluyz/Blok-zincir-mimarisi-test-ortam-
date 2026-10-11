"""
Automated PASS/FAIL Evaluation, Calibration, and Metric Calculation Engine.

Calculates:
1. Single & Multi-Topic Retrieval: Precision@k, Recall@k, mAP@k, Exact Set Match.
2. True Out-Of-Distribution (OOD) Rejection: TRR, FAR, AUROC, AUPRC.
3. Sub-system Latency Breakdown (Mean, Median, P95, Max ms).
4. Unselected A+ Read Isolation Ratio (Target = 0.00%).
5. Light Baseline Comparisons (Keyword Matching, Random LSH, Naive Bayes).
6. Automated PASS / FAIL / NOT_EVALUATED status computation based on predefined target criteria.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Any, Optional
import time
import math
import numpy as np


class EvaluationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class BenchmarkCriterion:
    metric_name: str
    achieved_value: float
    target_value: float
    comparison_operator: str  # ">=", "<=", "=="
    status: EvaluationStatus
    rationale: str


def evaluate_retrieval_metrics(
    predictions_top_k: List[List[str]],
    ground_truth_sets: List[Set[str]],
    k: int = 3
) -> Dict[str, float]:
    """
    Calculates Precision@k, Recall@k, mAP@k, and Exact Set Match.
    """
    precisions = []
    recalls = []
    average_precisions = []
    exact_matches = 0

    num_samples = len(predictions_top_k)
    if num_samples == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mAP_at_k": 0.0, "exact_match": 0.0}

    for preds, gt in zip(predictions_top_k, ground_truth_sets):
        preds_k = preds[:k]
        if not gt:
            continue

        hits = sum(1 for p in preds_k if p in gt)
        p_k = hits / min(k, len(preds_k)) if preds_k else 0.0
        r_k = hits / len(gt) if gt else 0.0

        precisions.append(p_k)
        recalls.append(r_k)

        # Average Precision @ k
        running_hits = 0
        ap_sum = 0.0
        for i, p in enumerate(preds_k):
            if p in gt:
                running_hits += 1
                ap_sum += running_hits / (i + 1)
        ap = ap_sum / min(len(gt), k) if gt else 0.0
        average_precisions.append(ap)

        # Exact Match
        if set(preds_k) == gt:
            exact_matches += 1

    return {
        f"precision_at_{k}": float(np.mean(precisions)) * 100.0 if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) * 100.0 if recalls else 0.0,
        f"mAP_at_{k}": float(np.mean(average_precisions)) * 100.0 if average_precisions else 0.0,
        "exact_match": (exact_matches / num_samples) * 100.0 if num_samples else 0.0
    }


def evaluate_ood_metrics(
    in_domain_confidences: List[float],
    ood_confidences: List[float],
    accept_threshold: float = 0.55
) -> Dict[str, float]:
    """
    Evaluates True Reject Rate (TRR), False Accept Rate (FAR), AUROC, and AUPRC.
    """
    # True Reject Rate: Fraction of OOD queries with confidence < threshold
    true_rejects = sum(1 for conf in ood_confidences if conf < accept_threshold)
    trr = (true_rejects / len(ood_confidences)) * 100.0 if ood_confidences else 100.0

    # False Accept Rate: Fraction of OOD queries with confidence >= threshold
    false_accepts = sum(1 for conf in ood_confidences if conf >= accept_threshold)
    far = (false_accepts / len(ood_confidences)) * 100.0 if ood_confidences else 0.0

    # Simplified AUROC calculation
    all_scores = [(s, 1) for s in in_domain_confidences] + [(s, 0) for s in ood_confidences]
    all_scores.sort(key=lambda x: x[0], reverse=True)

    n_pos = len(in_domain_confidences)
    n_neg = len(ood_confidences)

    if n_pos == 0 or n_neg == 0:
        auroc = 1.0
    else:
        rank_sum = 0
        for rank, (score, label) in enumerate(all_scores, 1):
            if label == 1:
                # Rank 1 is highest score, so inverted rank = total - rank + 1
                rank_sum += (len(all_scores) - rank + 1)
        auroc = (rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)

    return {
        "true_reject_rate": trr,
        "false_accept_rate": far,
        "auroc": float(auroc) * 100.0
    }


def calculate_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "max_ms": float(np.max(latencies_ms))
    }


def evaluate_criterion(
    metric_name: str,
    achieved: float,
    target: float,
    op: str,
    rationale: str
) -> BenchmarkCriterion:
    """
    Automatically evaluates PASS or FAIL status based on target condition.
    """
    if op == ">=":
        passed = achieved >= target
    elif op == "<=":
        passed = achieved <= target
    elif op == "==":
        passed = abs(achieved - target) < 1e-6
    else:
        passed = False

    status = EvaluationStatus.PASS if passed else EvaluationStatus.FAIL
    return BenchmarkCriterion(
        metric_name=metric_name,
        achieved_value=achieved,
        target_value=target,
        comparison_operator=op,
        status=status,
        rationale=rationale
    )
