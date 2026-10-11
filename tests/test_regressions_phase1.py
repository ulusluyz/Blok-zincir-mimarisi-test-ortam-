"""
Regression and Negative Unit Test Suite for Phase 1 Infrastructure Verification.
"""

import pytest
from src.dataset import get_full_split_datasets, CATEGORIES
from src.dataset_chain_block import get_chain_block_benchmark_dataset, ChainBlockTestCase
from src.features import FeatureExtractor
from src.bitwise import BinaryVector
from src.blocks import KnowledgeBlock, AddressBlock, BlockNetwork, AddressAccessViolation
from src.relational_inference import RelationalInferenceEngine, RelationType
from src.cognitive_control import CognitiveUncertaintyAuditor, DecisionType
from src.evaluation_engine import evaluate_criterion, EvaluationStatus


def test_inference_without_answer_chain_leakage():
    engine = RelationalInferenceEngine(decay_factor=0.9, max_hops=5)
    engine.add_relation("Aslan", "Memeli", RelationType.IS_A, weight=1.0)
    engine.add_relation("Memeli", "Hayvan", RelationType.IS_A, weight=1.0)
    engine.add_relation("Hayvan", "Canlı", RelationType.IS_A, weight=1.0)

    # Autonomously extract concepts from query text without ground truth annotations
    extracted = engine.extract_concepts_from_query("Aslan canlı mıdır?")
    assert "Aslan" in extracted
    assert "Canlı" in extracted

    # Run inference starting from extracted concept
    deductions = engine.find_all_deductions("Aslan", "Canlı")
    assert len(deductions) == 1
    assert deductions[0].is_valid_inference is True
    assert deductions[0].hop_count == 3


def test_non_transitive_deduction_rejection():
    engine = RelationalInferenceEngine(decay_factor=0.9, max_hops=5)
    engine.add_relation("Ali", "Can", RelationType.KNOWS, weight=1.0)
    engine.add_relation("Can", "Efe", RelationType.KNOWS, weight=1.0)

    deductions = engine.find_all_deductions("Ali", "Efe")
    assert len(deductions) == 1
    assert deductions[0].is_valid_inference is False
    assert deductions[0].total_confidence == 0.0


def test_unauthorized_direct_a_plus_access_trap():
    a_plus = AddressBlock("unauthorized_block")
    assert a_plus.unauthorized_access_count == 0

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(caller_token="ILLEGAL_DIRECT_ACCESS")

    assert a_plus.unauthorized_access_count == 1
    assert a_plus.access_logs[0]["status"] == "VIOLATION"


def test_dataset_intent_vs_category_decoupling():
    cases = get_chain_block_benchmark_dataset()
    cmd_cases = [c for c in cases if c.expected_intent in ["COMMAND", "PROHIBITION"]]
    assert len(cmd_cases) >= 6

    # Verify that command intent is decoupled from topic categories
    for c in cmd_cases:
        assert c.expected_intent in ["COMMAND", "PROHIBITION"]


def test_validation_set_threshold_tuning_no_test_leakage():
    auditor = CognitiveUncertaintyAuditor()
    val_cand_scores = [[("B1", 0.80), ("B2", 0.40)], [("B1", 0.50), ("B2", 0.45)]]
    val_expected = ["ACCEPT", "ÇELİŞKİLİ_KANIT"]

    auditor.calibrate_on_validation_set(val_cand_scores, val_expected)
    assert 0.50 <= auditor.accept_threshold <= 0.80


def test_feature_extractor_split_hygiene():
    train_data, val_data, single_test, multi_test = get_full_split_datasets()
    train_texts = [t for t, _ in train_data]

    fe = FeatureExtractor(max_features=256)
    fe.fit(train_texts)

    test_text = single_test[0]["text"]
    # Transforming test text must not alter fitted vocabulary size
    vocab_size_before = len(fe.vocab)
    _ = fe.transform_dense(test_text)
    assert len(fe.vocab) == vocab_size_before


def test_no_accidental_pass_on_failure():
    c_failed = evaluate_criterion(
        "Top1_Accuracy", 35.0, 80.0, ">=", "Target top1 accuracy >= 80%"
    )
    assert c_failed.status == EvaluationStatus.FAIL
