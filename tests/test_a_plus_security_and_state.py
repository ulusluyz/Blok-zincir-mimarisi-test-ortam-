"""
Unit tests for A+ Address Layer 5-State Machine & HMAC Authorization Security.
"""

import pytest
import time
from src.bitwise import BinaryVector
from src.blocks import (
    AddressBlock, AddressState, AccessGrantToken, AccessGrantManager, AddressAccessViolation,
    KnowledgeBlock, BlockNetwork
)


def test_address_state_machine_and_authorized_read():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_01")
    assert a_plus.current_state == AddressState.NOT_SELECTED
    assert a_plus.unauthorized_access_count == 0

    token = mgr.issue_grant(query_id="Q_1001", target_block_id="block_01")

    # State transition: NOT_SELECTED -> SELECTED -> AUTHORIZED
    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    # Authorized Read
    data = a_plus.read_layer(grant_token=token, grant_manager=mgr)
    assert data["block_id"] == "block_01"
    assert a_plus.current_state == AddressState.READ
    assert a_plus.read_count == 1
    assert a_plus.unauthorized_access_count == 0


def test_unauthorized_direct_access_and_forged_token_rejection():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_02")

    # 1. Direct call with None token
    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=None, grant_manager=mgr)
    assert a_plus.unauthorized_access_count == 1

    # 2. Forged token instantiated publicly without valid HMAC signature
    forged_token = AccessGrantToken(
        query_id="Q_1002",
        target_block_id="block_02",
        token_id="T_FORGED",
        created_at=time.perf_counter(),
        signature="fake_hmac_signature"
    )
    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=forged_token, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 2
    assert a_plus.access_logs[-1]["status"] == "VIOLATION_FORGED_SIGNATURE"


def test_cross_block_token_reuse_rejection():
    mgr = AccessGrantManager()
    a_plus_a = AddressBlock("block_A")
    a_plus_b = AddressBlock("block_B")

    token_a = mgr.issue_grant(query_id="Q_1002", target_block_id="block_A")

    a_plus_b.set_state(AddressState.SELECTED)
    a_plus_b.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus_b.read_layer(grant_token=token_a, grant_manager=mgr)

    assert a_plus_b.unauthorized_access_count == 1
    assert a_plus_b.access_logs[-1]["status"] == "VIOLATION_BLOCK_MISMATCH"


def test_double_spend_token_rejection():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_03")

    token = mgr.issue_grant(query_id="Q_1003", target_block_id="block_03")

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


def test_invalid_state_machine_transitions():
    a_plus = AddressBlock("block_04")
    assert a_plus.current_state == AddressState.NOT_SELECTED

    # Direct transition NOT_SELECTED -> READ is invalid
    with pytest.raises(AddressAccessViolation):
        a_plus.set_state(AddressState.READ)

    # Direct transition NOT_SELECTED -> AUTHORIZED is invalid
    with pytest.raises(AddressAccessViolation):
        a_plus.set_state(AddressState.AUTHORIZED)


def test_read_in_selected_state_rejected():
    mgr = AccessGrantManager()
    a_plus = AddressBlock("block_05")
    token = mgr.issue_grant(query_id="Q_1005", target_block_id="block_05")

    # Set state to SELECTED (not AUTHORIZED yet)
    a_plus.set_state(AddressState.SELECTED)

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token, grant_manager=mgr)

    assert a_plus.unauthorized_access_count == 1
    assert "VIOLATION_INVALID_STATE" in a_plus.access_logs[-1]["status"]
