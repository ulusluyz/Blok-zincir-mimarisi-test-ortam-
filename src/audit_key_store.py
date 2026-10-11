"""Local audit HMAC key provisioning. Keep generated keys outside repositories."""
from __future__ import annotations
import os
import secrets
import stat
from pathlib import Path

KEY_BYTES = 32

def load_audit_key(path: str | os.PathLike) -> bytes:
    target = Path(path)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("Audit key must be a regular, non-symlink file")
    if info.st_mode & 0o077:
        raise PermissionError("Audit key must be private (mode 0600 or stricter)")
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise PermissionError("Audit key owner differs from process user")
    data = target.read_bytes()
    if len(data) != KEY_BYTES:
        raise ValueError("Audit key must contain exactly 32 random bytes")
    return data

def provision_audit_key(path: str | os.PathLike) -> bytes:
    """Create once, never overwrite. Caller must back up the key securely."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = secrets.token_bytes(KEY_BYTES)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return load_audit_key(target)
