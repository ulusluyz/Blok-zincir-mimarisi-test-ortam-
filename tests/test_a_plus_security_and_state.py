"""
Unit tests for A+ Address Layer 5-State Machine & Authorization Security.
"""

import pytest
import time
from src.bitwise import BinaryVector
from src.blocks import (
    AddressBlock, AddressState, AccessGrantToken, AddressAccessViolation,
    KnowledgeBlock, BlockNetwork
)


def test_address_state_machine_and_authorized_read():
    a_plus = AddressBlock("block_01")
    assert a_plus.current_state == AddressState.NOT_SELECTED
    assert a_plus.unauthorized_access_count == 0

    # Grant token
    token = AccessGrantToken(
        query_id="Q_1001",
        target_block_id="block_01",
        token_id="T_1001",
        created_at=time.perf_counter()
    )

    # State transition: NOT_SELECTED -> SELECTED -> AUTHORIZED
    a_plus.set_state(AddressState.SELECTED)
    a_plus.set_state(AddressState.AUTHORIZED)

    # Authorized Read
    data = a_plus.read_layer(grant_token=token)
    assert data["block_id"] == "block_01"
    assert a_plus.current_state == AddressState.READ
    assert a_plus.read_count == 1
    assert a_plus.unauthorized_access_count == 0


def test_unauthorized_direct_access_rejection():
    a_plus = AddressBlock("block_02")

    # 1. Direct call without token
    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=None)
    assert a_plus.unauthorized_access_count == 1

    # 2. Legacy string token spoofing
    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token="AUTHORIZED_ROUTER")  # Strings rejected!
    assert a_plus.unauthorized_access_count == 2


def test_cross_block_token_reuse_rejection():
    a_plus_a = AddressBlock("block_A")
    a_plus_b = AddressBlock("block_B")

    # Token issued specifically for block_A
    token_a = AccessGrantToken(
        query_id="Q_1002",
        target_block_id="block_A",
        token_id="T_A",
        created_at=time.perf_counter()
    )

    a_plus_b.set_state(AddressState.AUTHORIZED)

    # Attempt to use token_A on block_B -> MUST REJECT!
    with pytest.raises(AddressAccessViolation):
        a_plus_b.read_layer(grant_token=token_a)

    assert a_plus_b.unauthorized_access_count == 1


def test_expired_closed_token_rejection():
    a_plus = AddressBlock("block_03")

    token = AccessGrantToken(
        query_id="Q_1003",
        target_block_id="block_03",
        token_id="T_03",
        created_at=time.perf_counter()
    )

    # Revoke / close token
    token.revoke()

    a_plus.set_state(AddressState.AUTHORIZED)

    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token)

    assert a_plus.unauthorized_access_count == 1


def test_unselected_state_read_rejection():
    a_plus = AddressBlock("block_04")
    assert a_plus.current_state == AddressState.NOT_SELECTED

    token = AccessGrantToken(
        query_id="Q_1004",
        target_block_id="block_04",
        token_id="T_04",
        created_at=time.perf_counter()
    )

    # State is NOT_SELECTED => MUST REJECT!
    with pytest.raises(AddressAccessViolation):
        a_plus.read_layer(grant_token=token)

    assert a_plus.unauthorized_access_count == 1


def test_block_network_query_session_and_token_cleanup():
    a_plus_1 = AddressBlock("B1")
    a_plus_2 = AddressBlock("B2")

    b1 = KnowledgeBlock("B1", "Selamlaşma", BinaryVector.from_bool_list([True]*128), a_plus_1)
    b2 = KnowledgeBlock("B2", "Gezegenler", BinaryVector.from_bool_list([False]*128), a_plus_2)

    network = BlockNetwork([b1, b2])

    Q = BinaryVector.from_bool_list([True]*128)
    res = network.process_query(Q, method="xor", top_k=1, query_id="Q_SESSION_1")

    assert res["total_a_plus_read"] == 1
    assert a_plus_1.read_count == 1
    assert a_plus_2.read_count == 0

    # Verify state cleanup post-query
    assert a_plus_1.current_state == AddressState.CLOSED
    assert a_plus_2.current_state == AddressState.NOT_SELECTED

    # Verify session audit logs
    audit = res["audit_session"]
    assert audit["query_id"] == "Q_SESSION_1"
    assert audit["selected_block_ids"] == ["B1"]
    assert audit["unauthorized_attempts"] == 0
    assert audit["unauthorized_successful_reads"] == 0
