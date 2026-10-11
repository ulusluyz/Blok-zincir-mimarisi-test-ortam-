"""
Unit tests for Evaluation Engine, AUROC Tie Handling, and Automatic PASS/FAIL Status Computation.
"""

import pytest
from src.evaluation_engine import (
    evaluate_retrieval_metrics, evaluate_ood_metrics, evaluate_access_audit_logs,
    calculate_latency_stats, evaluate_criterion, EvaluationStatus
)


def test_retrieval_metrics_calculation():
    preds = [["cat1", "cat2", "cat3"], ["cat2", "cat4", "cat5"]]
    ground_truth = [{"cat1", "cat2"}, {"cat2"}]

    metrics = evaluate_retrieval_metrics(preds, ground_truth, k=3)
    assert metrics["precision_at_3"] > 0.0
    assert metrics["recall_at_3"] > 0.0
    assert metrics["mAP_at_3"] > 0.0


def test_ood_metrics_calculation():
    in_domain = [0.85, 0.92, 0.78, 0.88]
    ood = [0.42, 0.35, 0.50, 0.48]

    metrics = evaluate_ood_metrics(in_domain, ood, accept_threshold=0.55)
    assert metrics["status"] == "EVALUATED"
    assert metrics["true_reject_rate"] == 100.0
    assert metrics["false_accept_rate"] == 0.0
    assert metrics["auroc"] == 100.0


def test_auroc_tied_scores_calculation():
    # Test tied scores between in-domain and OOD
    in_domain = [0.80, 0.70, 0.60]
    ood = [0.70, 0.60, 0.50]

    metrics = evaluate_ood_metrics(in_domain, ood, accept_threshold=0.55)
    assert metrics["status"] == "EVALUATED"
    assert 0.0 <= metrics["auroc"] <= 100.0


def test_empty_ood_sample_set_returns_not_evaluated():
    metrics = evaluate_ood_metrics(in_domain_confidences=[0.85], ood_confidences=[])
    assert metrics["status"] == "NOT_EVALUATED"


def test_latency_stats():
    lats = [1.2, 2.5, 3.8, 4.1, 10.5]
    stats = calculate_latency_stats(lats)
    assert stats["mean_ms"] > 0.0
    assert stats["p95_ms"] > 0.0


def test_criterion_auto_pass_fail():
    c_pass = evaluate_criterion(
        "Top1_Accuracy", 92.5, 80.0, ">=", "Target category accuracy >= 80%"
    )
    assert c_pass.status == EvaluationStatus.PASS

    c_fail = evaluate_criterion(
        "Unselected_Reads", 2.5, 0.0, "==", "Target unselected read ratio == 0%"
    )
    assert c_fail.status == EvaluationStatus.FAIL

    c_not_eval = evaluate_criterion(
        "OOD_AUROC", 0.0, 90.0, ">=", "Target AUROC >= 90%", is_evaluated=False
    )
    assert c_not_eval.status == EvaluationStatus.NOT_EVALUATED
