import os
import pytest
from src.audit_key_store import provision_audit_key, load_audit_key

def test_key_provision_and_reload(tmp_path):
    path = tmp_path / 'secret.key'
    key = provision_audit_key(path)
    assert len(key) == 32
    assert load_audit_key(path) == key
    assert (path.stat().st_mode & 0o777) == 0o600

def test_key_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'secret.key'
    original = provision_audit_key(path)
    with pytest.raises(FileExistsError):
        provision_audit_key(path)
    assert load_audit_key(path) == original

def test_key_rejects_world_readable_file(tmp_path):
    path = tmp_path / 'secret.key'
    provision_audit_key(path)
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        load_audit_key(path)

def test_key_rejects_symlink(tmp_path):
    path = tmp_path / 'secret.key'
    provision_audit_key(path)
    link = tmp_path / 'link.key'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        load_audit_key(link)

def test_key_rejects_short_file(tmp_path):
    path = tmp_path / 'secret.key'
    path.write_bytes(b'weak')
    os.chmod(path, 0o600)
    with pytest.raises(ValueError):
        load_audit_key(path)
