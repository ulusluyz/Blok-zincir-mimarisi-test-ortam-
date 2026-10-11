# Phase 1 local hardening v2 — independently executed

Scope: software reference access isolation only, not physical DRAM isolation.

- HMAC tokens tracked in manager registry, with TTL, atomic consumption, revocation, replay protection.
- HMAC-signed query audit snapshots; verify via `network.grant_manager.verify_audit(audit)`; mutation and deletion detected on verification.
- Query isolation: one RLock serializes calls on a BlockNetwork instance; **not** parallel query execution.
- Test suite: 79/79 passed (pytest, 2026-10-11).
- Stress: 8 threads x 150 queries, 16 blocks, top_k=3; audit signatures verified.

Limits: Audit signatures are in-memory, not a durable append-only/WORM store. An attacker with access to the manager secret or Python process can forge records. `verify_audit` must be invoked by the consuming verifier; the evaluation engine does not automatically authenticate all input audits. Direct public state mutation remains possible in Python. Token registry grows until lifecycle pruning is implemented. Full model-quality benchmarks not rerun. This is NOT a production security certification.
