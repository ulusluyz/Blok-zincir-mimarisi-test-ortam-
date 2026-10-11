import sqlite3
import pytest
from src.blocks import AccessGrantManager


def test_tail_deletion_detected(tmp_path):
    db, anchor = tmp_path / 'audit.db', tmp_path / 'checkpoint.json'
    key = b'test-secret-key'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    for index in range(3):
        record = {'query_id': str(index)}
        record['audit_signature'] = manager.sign_audit(record)
        manager.append_audit(record)
    assert manager.verify_audit_chain()
    manager._audit_db.close()
    with sqlite3.connect(db) as connection:
        connection.execute('DELETE FROM audit_chain WHERE seq=3')
    with pytest.raises(ValueError, match='anchor'):
        AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))


def test_anchor_modified_detected(tmp_path):
    db, anchor = tmp_path / 'audit.db', tmp_path / 'checkpoint.json'
    key = b'test-secret-key'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    manager._audit_db.close()
    anchor.write_text('{"tip":[0,"FORGED"],"signature":"FORGED"}')
    with pytest.raises(ValueError, match='anchor'):
        AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))


def test_missing_anchor_detected(tmp_path):
    db, anchor = tmp_path / 'audit.db', tmp_path / 'checkpoint.json'
    key = b'test-secret-key'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    record = {'query_id': 'test'}
    record['audit_signature'] = manager.sign_audit(record)
    manager.append_audit(record)
    manager._audit_db.close()
    anchor.unlink()
    with pytest.raises(ValueError, match='missing external anchor'):
        AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
