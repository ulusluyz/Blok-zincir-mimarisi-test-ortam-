"""Independent adversarial regression tests for local A+ hardening."""
from concurrent.futures import ThreadPoolExecutor
from src.blocks import BlockNetwork, KnowledgeBlock, AddressBlock
from src.bitwise import BinaryVector
from src.evaluation_engine import evaluate_access_audit_logs


def network():
    bits = BinaryVector.from_bool_list([True] * 128)
    return BlockNetwork([KnowledgeBlock("B1", "Category", bits, AddressBlock("B1"))]), bits


def test_concurrent_queries_are_isolated():
    net, bits = network()
    def run(i):
        return [net.process_query(bits, top_k=1, query_id=f"T{i}-{j}") for j in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(8)))
    assert sum(len(r) for r in results) == 320
    assert all(item["audit_session"]["query_id"] == item["query_id"] for group in results for item in group)
    assert all(evaluate_access_audit_logs([item["audit_session"]])["status"] == "EVALUATED" for group in results for item in group)


def test_missing_audit_evidence_is_not_pass():
    audit = {"query_id": "x", "selected_block_ids": ["B1"], "authorized_block_ids": [],
             "read_block_ids": ["B1"], "a_plus_read_count": 1}
    assert evaluate_access_audit_logs([audit])["status"] == "NOT_EVALUATED"


def test_selected_but_unauthorized_read_detected():
    audit = {"query_id": "x", "selected_block_ids": ["B1"], "authorized_block_ids": [],
             "read_block_ids": ["B1"], "a_plus_read_count": 1,
             "access_events": [{"query_id": "x", "block_id": "B1", "status": "AUTHORIZED_READ"}]}
    result = evaluate_access_audit_logs([audit])
    assert result["status"] == "FAIL"
    assert result["unauthorized_successful_reads"] == 1


def test_audit_attempts_are_query_scoped():
    net, bits = network()
    net.blocks[0].address_block.read_layer() if False else None
    first = net.process_query(bits, top_k=1, query_id="first")
    net.blocks[0].address_block.unauthorized_access_count += 42
    second = net.process_query(bits, top_k=1, query_id="second")
    assert first["audit_session"]["unauthorized_attempts"] == 0
    assert second["audit_session"]["unauthorized_attempts"] == 0


def test_audit_signature_detects_modification_and_missing_records():
    net, bits = network()
    result = net.process_query(bits, top_k=1)
    audit = result["audit_session"]
    assert net.grant_manager.verify_audit(audit)
    audit["access_events"].clear()
    assert not net.grant_manager.verify_audit(audit)
    assert net.grant_manager.verify_audit(net.last_query_audit)
    net.last_query_audit["read_block_ids"].clear()
    assert not net.grant_manager.verify_audit(net.last_query_audit)


def test_expired_and_replayed_token_rejected():
    import time
    import copy
    from src.blocks import AccessGrantManager, AddressState, AddressAccessViolation
    import pytest
    manager = AccessGrantManager(token_ttl_seconds=0.002)
    addr = AddressBlock("B")
    token = manager.issue_grant("Q", "B")
    addr.set_state(AddressState.SELECTED)
    addr.set_state(AddressState.AUTHORIZED)
    time.sleep(0.01)
    with pytest.raises(AddressAccessViolation):
        addr.read_layer(token, manager)
    assert addr.read_count == 0
    manager2 = AccessGrantManager()
    token2 = manager2.issue_grant("Q", "B")
    addr.read_layer(token2, manager2)
    replay = copy.deepcopy(token2)
    replay.is_used = False
    addr.set_state(AddressState.CLOSED)
    addr.set_state(AddressState.SELECTED)
    addr.set_state(AddressState.AUTHORIZED)
    with pytest.raises(AddressAccessViolation):
        addr.read_layer(replay, manager2)
    assert addr.read_count == 1
