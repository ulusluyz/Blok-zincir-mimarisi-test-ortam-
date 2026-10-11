"""
Exhaustive Fault Injection and Security Test Suite for A+ Security Model.
"""

import pytest
import time
from src.bitwise import BinaryVector
from src.blocks import (
    AddressBlock, AddressState, AccessGrantToken, AccessGrantManager, AddressAccessViolation,
    KnowledgeBlock, BlockNetwork
)
from src.evaluation_engine import (
    evaluate_ood_metrics, evaluate_access_audit_logs, evaluate_criterion, EvaluationStatus
)


def test_forged_token_hmac_signature_rejected():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_sec_01")
    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    forged = AccessGrantToken(
        query_id="Q_ATTACK",
        target_block_id="block_sec_01",
        token_id="T_FORGED",
        created_at=time.perf_counter(),
        signature="invalid_signature_hash"
    )

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=forged, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 1
    assert a_plus.access_logs[-1]["status"] == "VIOLATION_FORGED_SIGNATURE"


def test_invalid_state_transition_and_read_in_selected_state_rejected():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_sec_02")

    # Invalid direct transition: NOT_SELECTED -> READ
    with pytest.raises(AddressAccessViolation):
        a_plus.set_state(AddressState.READ)

    # Valid transition to SELECTED
    a_plus.set_state(AddressState.SELECTED)

    token = mgr.issue_grant("Q_ATTACK", "block_sec_02")

    # Attempt to read in SELECTED state (NOT AUTHORIZED YET) -> MUST REJECT!
    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 1


def test_token_double_spend_rejection():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_sec_03")

    token = mgr.issue_grant("Q_1001", "block_sec_03")

    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    _ = a_plus.read_layer(grant_token=token, grant_manager=mgr)
    assert token.is_used is True

    # Attempt second read with same token -> DOUBLE SPEND REJECTION
    a_plus.set_state(AddressState.CLOSED)
    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 1
    assert a_plus.access_logs[-1]["status"] == "VIOLATION_TOKEN_DOUBLE_SPEND"


def test_closed_expired_token_rejection():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_sec_04")

    token = mgr.issue_grant("Q_1002", "block_sec_04")
    token.revoke()

    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 1
    assert a_plus.access_logs[-1]["status"] == "VIOLATION_TOKEN_CLOSED"


def test_cross_block_token_transfer_rejection():
    mgr = AccessGrantManager()
    a_plus_a = AddressBlock("block_A")
    a_plus_b = AddressBlock("block_B")

    token_a = mgr.issue_grant("Q_1003", "block_A")

    a_plus_b.set_state(AddressState.SELECTED)
    a_plus_b.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus_b.read_layer(grant_token=token_a, grant_manager=mgr)

    assert a_plus_b.unauthorized_access_count == 1
    assert a_plus_b.access_logs[-1]["status"] == "VIOLATION_BLOCK_MISMATCH"


def test_try_finally_session_cleanup_on_read_fault():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("fault_block")
    b1 = KnowledgeBlock("kb_fault", "Cat", BinaryVector.from_bool_list([True]*128), a_plus)

    network = BlockNetwork([b1], grant_manager=mgr)

    # Induce artificial fault inside read_layer by corrupting state
    a_plus.set_state(AddressState.NOT_SELECTED)

    # Process query will attempt read and raise violation, but try...finally MUST clean up
    res = network.process_query(BinaryVector.from_bool_list([True]*128), top_k=1, query_id="Q_FAULT")
    assert res["total_a_plus_read"] == 1
    assert a_plus.read_count == 1
    assert a_plus.current_state == AddressState.CLOSED


def test_empty_dataset_returns_not_evaluated():
    ood_res = evaluate_ood_metrics(in_domain_confidences=[], ood_confidences=[])
    assert ood_res["status"] == "NOT_EVALUATED"

    c = evaluate_criterion("OOD_AUROC", 0.0, 90.0, ">=", "Target AUROC >= 90%", is_evaluated=False)
    assert c.status == EvaluationStatus.NOT_EVALUATED


def test_access_audit_violation_triggers_fail():
    audit_logs = [{
        "query_id": "Q_BAD",
        "selected_block_ids": ["B1"],
        "read_block_ids": ["B1", "B2_UNSELECTED"],  # Injected violation: B2 was read without being selected!
        "a_plus_read_count": 2
    }]

    eval_res = evaluate_access_audit_logs(audit_logs)
    assert eval_res["unauthorized_successful_reads"] == 1
    assert eval_res["unselected_a_plus_read_ratio"] > 0.0

    c = evaluate_criterion(
        "Unselected_APlus_Read_Ratio", eval_res["unselected_a_plus_read_ratio"], 0.0, "==", "Target == 0%"
    )
    assert c.status == EvaluationStatus.FAIL
