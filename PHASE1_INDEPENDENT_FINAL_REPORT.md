# Zincir Blok — Phase 1 independent local verification (2026-10-11)

## Scope
Software-reference A / A+ access isolation, not physical memory protection or a production security certification. Original GitHub repository was not modified.

## Implemented
- Token registry lifecycle cleanup (`AccessGrantManager.prune_grants`), retaining fail-closed rejection of unknown/replayed token IDs; 128-bit UUID token identifiers.
- Optional SQLite-backed audit ledger (`audit_db_path`), with per-entry HMAC and a previous-entry MAC chain. Ledger verified on reopening with the same supplied secret key.
- `verify_audit_chain()` detects content modification and internal gaps; it is **not** a WORM or externally anchored ledger. Tail deletion, whole-file replacement, privileged process manipulation and secret compromise cannot be ruled out by this design alone.
- `evaluate_access_audit_logs(..., grant_manager=manager)` optionally validates audit HMAC before assessing evidence. Without manager argument, cryptographic validation is not performed for compatibility.
- `BlockNetwork.add_block` uses the query lock and rejects duplicate block IDs.
- Existing `BlockNetwork` query lock serializes requests; it is not parallel execution.

## Independently executed checks
- Full pytest suite: **85 passed in 29.37 s**.
- Stress: **8 threads × 150 queries = 1,200 queries**, 16 blocks, top_k=3, 1.73 s; no exceptions; 1,200 signed persistent ledger entries verified; 0 live token records after cleanup.
- Additional negative tests: replay after pruning, audit record tampering, reopening with stable/wrong secret, concurrent persistent audits, evaluator signature rejection, duplicate block rejection.

## Unresolved / honest boundaries
- Python public object/state mutation is not a process-level security boundary; hostile in-process code can bypass it.
- Audit chain has no external immutable head anchor and is not append-only at the storage layer.
- Secret-key storage/rotation and backup recovery are not implemented; a random default key cannot validate an existing ledger across process restarts. Provide a persistent securely managed secret when using `audit_db_path`.
- Failed queries do not always produce a signed failure audit because audit creation follows the read loop.
- Full model quality benchmarks (OOD, multi-hop, generalization) have **not** been rerun and are not proven by these access tests.
- No claim of 100% security or production certification is warranted.
