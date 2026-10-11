import sqlite3
import pytest
from src.blocks import AccessGrantManager


def _signed(manager, q):
    record = {"query_id": q}
    record["audit_signature"] = manager.sign_audit(record)
    return record


def test_deleted_tail_cannot_be_hidden_by_next_append(tmp_path):
    db, anchor = tmp_path / "audit.sqlite", tmp_path / "anchor.json"
    manager = AccessGrantManager(secret_key=b"test-secret", audit_db_path=str(db), audit_anchor_path=str(anchor))
    manager.append_audit(_signed(manager, "one"))
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM audit_chain")
    with pytest.raises(ValueError, match="integrity"):
        manager.append_audit(_signed(manager, "two"))
    assert not manager.verify_audit_chain()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0] == 0


def test_modified_anchor_cannot_be_overwritten_by_next_append(tmp_path):
    db, anchor = tmp_path / "audit.sqlite", tmp_path / "anchor.json"
    manager = AccessGrantManager(secret_key=b"test-secret", audit_db_path=str(db), audit_anchor_path=str(anchor))
    manager.append_audit(_signed(manager, "one"))
    anchor.write_text('{"tip":[0,"GENESIS"],"signature":"fake"}')
    with pytest.raises(ValueError, match="integrity"):
        manager.append_audit(_signed(manager, "two"))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0] == 1
