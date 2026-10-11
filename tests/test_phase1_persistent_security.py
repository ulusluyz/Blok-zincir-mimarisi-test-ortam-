"""Independent checks for token lifecycle and optional persistent audit chain."""
import copy
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from src.blocks import AccessGrantManager, BlockNetwork, KnowledgeBlock, AddressBlock
from src.bitwise import BinaryVector


def build(manager):
    bits = BinaryVector.from_bool_list([True] * 128)
    return BlockNetwork([KnowledgeBlock('B1','C',bits,AddressBlock('B1'))], manager), bits


def test_pruning_keeps_replays_invalid():
    manager = AccessGrantManager()
    token = manager.issue_grant('Q', 'B1')
    assert manager.consume_grant(token)
    replay = copy.deepcopy(token)
    replay.is_used = False
    assert manager.prune_grants() == 1
    assert not manager.verify_grant(replay)
    assert not manager.consume_grant(replay)


def test_audit_chain_detects_tampering(tmp_path):
    db = tmp_path/'audit.sqlite'
    key = b'static-test-secret'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db))
    net, bits = build(manager)
    for i in range(5):
        net.process_query(bits, query_id=f'Q{i}')
    assert manager.verify_audit_chain()
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE audit_chain SET payload='{}' WHERE seq=2")
    assert not manager.verify_audit_chain()
    with pytest.raises(ValueError):
        AccessGrantManager(secret_key=key, audit_db_path=str(db))


def test_audit_chain_reopens_with_stable_key(tmp_path):
    db = tmp_path/'audit.sqlite'
    key = b'stable-test-key'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db))
    net, bits = build(manager)
    net.process_query(bits)
    assert AccessGrantManager(secret_key=key, audit_db_path=str(db)).verify_audit_chain()
    with pytest.raises(ValueError):
        AccessGrantManager(secret_key=b'wrong-key', audit_db_path=str(db))


def test_concurrent_persistent_audits(tmp_path):
    manager = AccessGrantManager(secret_key=b'parallel-key', audit_db_path=str(tmp_path/'log.db'))
    net, bits = build(manager)
    def run(i):
        for j in range(40):
            result = net.process_query(bits, query_id=f'Q{i}-{j}')
            assert manager.verify_audit(result['audit_session'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert manager.verify_audit_chain()
    assert manager._audit_db.execute('SELECT count(*) FROM audit_chain').fetchone()[0] == 320
    assert len(manager._issued) == 0


def test_evaluator_rejects_forged_signed_audit():
    from src.evaluation_engine import evaluate_access_audit_logs
    manager = AccessGrantManager()
    net, bits = build(manager)
    audit = net.process_query(bits)['audit_session']
    assert evaluate_access_audit_logs([audit], grant_manager=manager)['status'] == 'EVALUATED'
    audit['selected_block_ids'].clear()
    assert evaluate_access_audit_logs([audit], grant_manager=manager)['status'] == 'FAIL'


def test_duplicate_block_id_is_rejected():
    manager = AccessGrantManager()
    net, bits = build(manager)
    with pytest.raises(ValueError):
        net.add_block(KnowledgeBlock('B1', 'C', bits, AddressBlock('B1')))


def test_failed_query_is_signed_and_persisted(tmp_path):
    manager = AccessGrantManager(secret_key=b'failure-audit-key', audit_db_path=str(tmp_path/'fail.db'))
    net, bits = build(manager)
    def fail(*args, **kwargs):
        raise RuntimeError('injected read failure')
    net.blocks[0].read_address_block = fail
    with pytest.raises(RuntimeError, match='injected read failure'):
        net.process_query(bits, query_id='Q-FAILED')
    assert net.last_query_audit['status'] == 'FAILED'
    assert net.last_query_audit['query_id'] == 'Q-FAILED'
    assert manager.verify_audit(net.last_query_audit)
    assert manager.verify_audit_chain()
    row = manager._audit_db.execute('SELECT payload FROM audit_chain').fetchone()
    import json
    assert json.loads(row[0])['status'] == 'FAILED'
    assert not manager._issued


def test_persistent_audit_rejects_ephemeral_key(tmp_path):
    with pytest.raises(ValueError, match='stable secret_key'):
        AccessGrantManager(audit_db_path=str(tmp_path/'audit.db'))
