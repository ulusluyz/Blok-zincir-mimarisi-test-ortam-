import json
import sqlite3
import pytest
from src.blocks import AccessGrantManager

KEY = b"independent-recovery-test-key"

def record(manager, number):
    item = {"query_id": str(number)}
    item["audit_signature"] = manager.sign_audit(item)
    manager.append_audit(item)

def setup(tmp_path):
    db, anchor = tmp_path / "audit.sqlite", tmp_path / "anchor.json"
    manager = AccessGrantManager(secret_key=KEY, audit_db_path=str(db), audit_anchor_path=str(anchor))
    record(manager, 1)
    old_anchor = anchor.read_bytes()
    record(manager, 2)
    manager._audit_db.close()
    return db, anchor, old_anchor

def test_crash_gap_requires_explicit_recovery(tmp_path):
    db, anchor, old_anchor = setup(tmp_path)
    anchor.write_bytes(old_anchor)  # simulate power loss before checkpoint replacement
    with pytest.raises(ValueError, match="anchor"):
        AccessGrantManager(secret_key=KEY, audit_db_path=str(db), audit_anchor_path=str(anchor))
    recovery = AccessGrantManager(secret_key=KEY, audit_db_path=str(db))
    assert recovery.recover_interrupted_anchor(str(anchor))
    assert recovery.verify_audit_chain()
    recovery._audit_db.close()
    assert AccessGrantManager(secret_key=KEY, audit_db_path=str(db), audit_anchor_path=str(anchor)).verify_audit_chain()

def test_recovery_refuses_invalid_signature(tmp_path):
    db, anchor, old_anchor = setup(tmp_path)
    stale = json.loads(old_anchor)
    stale["signature"] = "forged"
    anchor.write_text(json.dumps(stale))
    recovery = AccessGrantManager(secret_key=KEY, audit_db_path=str(db))
    with pytest.raises(ValueError, match="signature"):
        recovery.recover_interrupted_anchor(str(anchor))

def test_recovery_refuses_multiple_unanchored_rows(tmp_path):
    db, anchor, old_anchor = setup(tmp_path)
    manager = AccessGrantManager(secret_key=KEY, audit_db_path=str(db))
    record(manager, 3)
    anchor.write_bytes(old_anchor)
    with pytest.raises(ValueError, match="exactly one"):
        manager.recover_interrupted_anchor(str(anchor))

def test_recovery_refuses_tampered_ledger(tmp_path):
    db, anchor, old_anchor = setup(tmp_path)
    anchor.write_bytes(old_anchor)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE audit_chain SET payload='{}' WHERE seq=2")
    with pytest.raises(ValueError, match="integrity"):
        AccessGrantManager(secret_key=KEY, audit_db_path=str(db))
