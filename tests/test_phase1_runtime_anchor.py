import sqlite3

from src.blocks import AccessGrantManager


def _setup(tmp_path):
    db = tmp_path / 'audit.sqlite'
    anchor = tmp_path / 'anchor.json'
    manager = AccessGrantManager(secret_key=b'local-test-key', audit_db_path=str(db), audit_anchor_path=str(anchor))
    audit = {'query_id': 'one'}
    audit['audit_signature'] = manager.sign_audit(audit)
    manager.append_audit(audit)
    assert manager.verify_audit_chain()
    return manager, db, anchor


def test_runtime_anchor_deleted(tmp_path):
    manager, _, anchor = _setup(tmp_path)
    anchor.unlink()
    assert manager.verify_audit_chain() is False


def test_runtime_anchor_modified(tmp_path):
    manager, _, anchor = _setup(tmp_path)
    anchor.write_text('{"tip":[0,"GENESIS"],"signature":"bad"}')
    assert manager.verify_audit_chain() is False


def test_runtime_tail_deleted(tmp_path):
    manager, db, _ = _setup(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute('DELETE FROM audit_chain WHERE seq=1')
    assert manager.verify_audit_chain() is False
