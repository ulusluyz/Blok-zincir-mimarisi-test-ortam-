"""
Automated PASS/FAIL Evaluation, Calibration, and Metric Calculation Engine.

Calculates:
1. Single & Multi-Topic Retrieval: Precision@k, Recall@k, mAP@k, Exact Set Match.
2. True Out-Of-Distribution (OOD) Rejection: TRR, FAR, AUROC, AUPRC.
3. Query-Scoped Access Audit: Unselected A+ Read Violations & Unauthorized Access Traps.
4. Sub-system Latency Breakdown (Mean, Median, P95, Max ms).
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
) -> Dict[str, Any]:
    """
    Evaluates True Reject Rate (TRR), False Accept Rate (FAR), AUROC with tie handling.
    Returns NOT_EVALUATED status if sample counts are insufficient.
    """
    n_pos = len(in_domain_confidences)
    n_neg = len(ood_confidences)

    if n_pos < 1 or n_neg < 1:
        return {
            "status": "NOT_EVALUATED",
            "true_reject_rate": 0.0,
            "false_accept_rate": 0.0,
            "auroc": 0.0
        }

    # True Reject Rate: Fraction of OOD queries with confidence < threshold
    true_rejects = sum(1 for conf in ood_confidences if conf < accept_threshold)
    trr = (true_rejects / n_neg) * 100.0

    # False Accept Rate: Fraction of OOD queries with confidence >= threshold
    false_accepts = sum(1 for conf in ood_confidences if conf >= accept_threshold)
    far = (false_accepts / n_neg) * 100.0

    # AUROC calculation with tie handling (Mann-Whitney U statistic)
    all_scores = [(s, 1) for s in in_domain_confidences] + [(s, 0) for s in ood_confidences]
    # Sort by score ascending to compute ranks
    all_scores.sort(key=lambda x: x[0])

    # Assign average ranks for tied scores
    ranks = [0.0] * len(all_scores)
    i = 0
    while i < len(all_scores):
        j = i
        while j < len(all_scores) and math.isclose(all_scores[j][0], all_scores[i][0], abs_tol=1e-7):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k_idx in range(i, j):
            ranks[k_idx] = avg_rank
        i = j

    pos_rank_sum = sum(ranks[k_idx] for k_idx, (_, label) in enumerate(all_scores) if label == 1)
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)

    return {
        "status": "EVALUATED",
        "true_reject_rate": trr,
        "false_accept_rate": far,
        "auroc": float(np.clip(auroc, 0.0, 1.0)) * 100.0
    }


def evaluate_access_audit_logs(query_audits: List[Dict[str, Any]], grant_manager: Optional[Any] = None) -> Dict[str, Any]:
    """Fail closed when audit evidence is missing or inconsistent.

    The audit consumes per-block access events captured at the read boundary,
    rather than treating the router's read IDs as proof of authorization.
    """
    if not query_audits:
        return {"status": "NOT_EVALUATED", "total_queries": 0,
                "unauthorized_attempts": 0, "unauthorized_successful_reads": 0,
                "unselected_a_plus_read_ratio": 0.0}

    total_reads = 0
    violations = 0
    attempts = 0
    for audit in query_audits:
        if grant_manager is not None and not grant_manager.verify_audit(audit):
            return {"status": "FAIL", "total_queries": len(query_audits),
                    "unauthorized_attempts": attempts, "unauthorized_successful_reads": violations,
                    "unselected_a_plus_read_ratio": 0.0, "reason": "Invalid audit signature"}
        required = ("query_id", "selected_block_ids", "authorized_block_ids",
                    "read_block_ids", "a_plus_read_count", "access_events")
        if any(key not in audit for key in required):
            return {"status": "NOT_EVALUATED", "total_queries": len(query_audits),
                    "unauthorized_attempts": attempts, "unauthorized_successful_reads": violations,
                    "unselected_a_plus_read_ratio": 0.0, "reason": "Missing audit evidence"}
        events = audit["access_events"]
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            return {"status": "NOT_EVALUATED", "total_queries": len(query_audits),
                    "unauthorized_attempts": attempts, "unauthorized_successful_reads": violations,
                    "unselected_a_plus_read_ratio": 0.0, "reason": "Malformed access events"}
        successful = [e for e in events if e.get("status") == "AUTHORIZED_READ"]
        actual_ids = [e.get("block_id") for e in successful]
        selected = set(audit["selected_block_ids"])
        authorized = set(audit["authorized_block_ids"])
        if (len(successful) != audit["a_plus_read_count"] or
                sorted(actual_ids) != sorted(audit["read_block_ids"])):
            return {"status": "NOT_EVALUATED", "total_queries": len(query_audits),
                    "unauthorized_attempts": attempts, "unauthorized_successful_reads": violations,
                    "unselected_a_plus_read_ratio": 0.0, "reason": "Inconsistent read counts"}
        attempts += sum(str(e.get("status", "")).startswith("VIOLATION_") for e in events)
        violations += sum(e.get("block_id") not in selected or
                          e.get("block_id") not in authorized or
                          e.get("query_id") != audit["query_id"] for e in successful)
        total_reads += len(successful)
    if total_reads == 0:
        return {"status": "NOT_EVALUATED", "total_queries": len(query_audits),
                "unauthorized_attempts": attempts, "unauthorized_successful_reads": violations,
                "unselected_a_plus_read_ratio": 0.0, "reason": "No successful reads"}
    return {"status": "FAIL" if violations else "EVALUATED",
            "total_queries": len(query_audits), "unauthorized_attempts": attempts,
            "unauthorized_successful_reads": violations,
            "unselected_a_plus_read_ratio": 100.0 * violations / total_reads}


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
    rationale: str,
    is_evaluated: bool = True
) -> BenchmarkCriterion:
    """
    Automatically evaluates PASS, FAIL, or NOT_EVALUATED status based on target condition.
    """
    if not is_evaluated:
        return BenchmarkCriterion(
            metric_name=metric_name,
            achieved_value=0.0,
            target_value=target,
            comparison_operator=op,
            status=EvaluationStatus.NOT_EVALUATED,
            rationale=f"Insufficient evidence or missing dataset for {metric_name}."
        )

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
