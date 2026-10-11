# A_PLUS_SECURITY_AND_STATE_MACHINE.md: A+ Security Model & 5-State Machine Specification

## 1. Overview
The $A+$ Address Layer represents a decoupled, variable-length relational pointer table attached to an $A$ Knowledge Block kernel. To ensure that unselected $A+$ address layers are never accessed in memory during suitability evaluation, `AddressBlock` enforces a query-scoped **5-State Machine** combined with cryptographically generated `AccessGrantToken` objects.

---

## 2. Address Block 5-State Machine

```
   [NOT_SELECTED] ──(Router Selection)──> [SELECTED]
                                              │
                                   (Token Authorization)
                                              ▼
    [CLOSED] <──(Query Completion)───── [AUTHORIZED]
       ▲                                      │
       │                              (read_layer Call)
       │                                      ▼
       └─────────────────────────────────── [READ]
```

### State Definitions:
1. `NOT_SELECTED`: Initial default state for all $A+$ blocks. `read_layer()` calls in this state raise `AddressAccessViolation`.
2. `SELECTED`: Block identified as Top-k candidate by `BlockNetwork` based on $A$ suitability $S(A, Q)$.
3. `AUTHORIZED`: `BlockNetwork` issues a query-scoped `AccessGrantToken` for the block.
4. `READ`: `read_layer(grant_token)` executes successfully, returning connections and payload.
5. `CLOSED`: Query processing completes; `grant_token` is revoked and block state is reset.

---

## 3. Query-Scoped Access Grant Tokens (`AccessGrantToken`)
- **Structure**:
  ```python
  @dataclass
  class AccessGrantToken:
      query_id: str
      target_block_id: str
      token_id: str
      created_at: float
      is_closed: bool = False
  ```
- **Security Invariants**:
  1. **Token Type Trap**: Direct calls passing strings, legacy tokens, or `None` are rejected with `AddressAccessViolation`.
  2. **Query Scope Trap**: A token issued for query `Q_1` cannot be reused for query `Q_2`. Token revocation upon query completion (`token.revoke()`) prevents replay attacks.
  3. **Target Block Binding Trap**: A token issued for block `B_1` cannot be used to read block `B_2`.

---

## 4. Software Reference Isolation Guarantee vs Hardware Physical Boundaries
> **Technical Scope Clarification**:
> The access state machine and `AccessGrantToken` security traps provide **Software Reference Isolation Guarantees (OS/User Space Software Logic)** within the Python runtime environment.
> They verify that application logic never invokes or accesses unselected address layers. They do not constitute physical DRAM memory hardware bus or kernel MMU page-table hardware isolation.
