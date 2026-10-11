"""Failure-path and key-recovery regression checks, not physical power-loss certification."""
import os
import sqlite3
import pytest
from src.blocks import AccessGrantManager
from src.audit_key_store import provision_audit_key, load_audit_key


def append(manager, index):
    audit = {'query_id': str(index), 'status': 'SUCCESS'}
    audit['audit_signature'] = manager.sign_audit(audit)
    manager.append_audit(audit)


def test_wrong_key_fails_closed_on_restart(tmp_path):
    key_path = tmp_path / 'key'
    key = provision_audit_key(key_path)
    db, anchor = tmp_path / 'audit.db', tmp_path / 'anchor.json'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    append(manager, 1)
    manager._audit_db.close()
    with pytest.raises(ValueError, match='integrity'):
        AccessGrantManager(secret_key=os.urandom(32), audit_db_path=str(db), audit_anchor_path=str(anchor))
    assert AccessGrantManager(secret_key=load_audit_key(key_path), audit_db_path=str(db), audit_anchor_path=str(anchor)).verify_audit_chain()


def test_anchor_write_failure_is_not_silent(tmp_path, monkeypatch):
    key = os.urandom(32)
    db, anchor = tmp_path / 'audit.db', tmp_path / 'anchor.json'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    append(manager, 1)
    original_anchor = anchor.read_bytes()
    def fail():
        raise OSError('simulated filesystem failure')
    monkeypatch.setattr(manager, '_write_anchor', fail)
    with pytest.raises(OSError, match='filesystem failure'):
        append(manager, 2)
    assert anchor.read_bytes() == original_anchor
    assert not manager.verify_audit_chain()
    with pytest.raises(ValueError, match='anchor'):
        AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))


def test_sqlite_append_failure_preserves_anchor(tmp_path):
    key = os.urandom(32)
    db, anchor = tmp_path / 'audit.db', tmp_path / 'anchor.json'
    manager = AccessGrantManager(secret_key=key, audit_db_path=str(db), audit_anchor_path=str(anchor))
    append(manager, 1)
    old_anchor = anchor.read_bytes()
    manager._audit_db.execute('PRAGMA query_only=ON')
    with pytest.raises(sqlite3.OperationalError):
        append(manager, 2)
    assert anchor.read_bytes() == old_anchor
    assert manager.verify_audit_chain()
