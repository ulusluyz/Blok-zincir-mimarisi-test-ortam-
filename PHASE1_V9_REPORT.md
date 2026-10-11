# Phase 1 V9 – Independent local verification

- Baseline: local V8 archive; no GitHub modifications.
- Added `src/audit_key_store.py`: 32-byte CSPRNG HMAC key provisioning (`O_EXCL`, `O_NOFOLLOW` where available, mode 0600, fsync file and directory), strict load validation (regular file, owner, mode, key length).
- Added `tests/test_audit_key_store.py` with five negative/positive tests.
- Full suite: **104 passed in 32.37s** (`python -m pytest -q -x --disable-warnings`).
- Security caveat: file-based keys are not an HSM, keyring, or TPM. Provision outside repository, protect backups, and plan key rotation/recovery; possession of key allows forging audits. `load_audit_key` performs a path-based check then read, so adversarial local filesystem races remain a risk. Never package the secret key.
- Remaining: actual disk failure/power-cut validation; independent full model benchmark and metrics audit; real deployment key lifecycle/rotation.
