"""
Block Architecture Implementation with Robust Memory Access Traps:
A  — Bilgi Bloku (Knowledge Block): Fixed-size N-bit mathematical core. Evaluates suitability mathematically without activating connection layer.
A+ — Adres Bloku (Address Block): Independent, variable-size, extensible connection layer. Contains target block IDs & weights. Read ONLY when A is activated.

Includes software-level access state machine and query-scoped authorization tokens.
Note: Explicitly bounded as a Software Reference Isolation Guarantee (OS/User Space Software Logic), not a DRAM hardware-bus physical isolation.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
import inspect
import uuid
import hmac
import hashlib
import threading
import copy
import json
import sqlite3
import os
import tempfile
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine


class AddressState(Enum):
    NOT_SELECTED = "NOT_SELECTED"
    SELECTED = "SELECTED"
    AUTHORIZED = "AUTHORIZED"
    READ = "READ"
    CLOSED = "CLOSED"


# Allowed state machine transitions
VALID_STATE_TRANSITIONS: Dict[AddressState, Set[AddressState]] = {
    AddressState.NOT_SELECTED: {AddressState.SELECTED, AddressState.CLOSED},
    AddressState.SELECTED: {AddressState.AUTHORIZED, AddressState.NOT_SELECTED, AddressState.CLOSED},
    AddressState.AUTHORIZED: {AddressState.READ, AddressState.CLOSED},
    AddressState.READ: {AddressState.CLOSED, AddressState.NOT_SELECTED},
    AddressState.CLOSED: {AddressState.NOT_SELECTED, AddressState.SELECTED}
}


@dataclass
class AccessGrantToken:
    query_id: str
    target_block_id: str
    token_id: str
    created_at: float
    signature: str
    is_used: bool = False
    is_closed: bool = False

    def revoke(self):
        self.is_closed = True


class AccessGrantManager:
    """
    Session Token Factory: Issues and verifies HMAC-signed query-scoped AccessGrantToken objects.
    Prevents forging, spoofing, cross-block, or cross-query token transfer.
    """

    def __init__(self, secret_key: Optional[bytes] = None, token_ttl_seconds: float = 60.0, audit_db_path: Optional[str] = None, audit_anchor_path: Optional[str] = None):
        if token_ttl_seconds <= 0:
            raise ValueError("token_ttl_seconds must be positive")
        self._secret_key = secret_key if secret_key is not None else uuid.uuid4().bytes
        self._ttl = token_ttl_seconds
        self._issued = {}
        self._lock = threading.RLock()
        self._audit_db = None
        self._audit_anchor_path = audit_anchor_path
        if audit_anchor_path is not None and audit_db_path is None:
            raise ValueError("Audit anchor requires persistent audit_db_path")
        if audit_db_path is not None:
            if secret_key is None:
                raise ValueError("Persistent audit requires an explicitly supplied stable secret_key")
            self._audit_db = sqlite3.connect(audit_db_path, check_same_thread=False)
            self._audit_db.execute("CREATE TABLE IF NOT EXISTS audit_chain (seq INTEGER PRIMARY KEY AUTOINCREMENT, previous TEXT NOT NULL, payload TEXT NOT NULL, mac TEXT NOT NULL)")
            self._audit_db.commit()
            # First verify the ledger independently; the external anchor may
            # need initial creation for an empty ledger.
            anchor_path = self._audit_anchor_path
            self._audit_anchor_path = None
            try:
                valid_ledger = self.verify_audit_chain()
            finally:
                self._audit_anchor_path = anchor_path
            if not valid_ledger:
                raise ValueError("Existing audit ledger failed integrity verification")
            if self._audit_anchor_path is not None:
                self._verify_anchor_or_initialize()

    def prune_grants(self) -> int:
        """Discard consumed, revoked and expired grants. Unknown IDs always fail closed."""
        with self._lock:
            now = time.perf_counter()
            stale = [key for key, entry in self._issued.items()
                     if entry[4] or entry[5] or now - entry[2] > self._ttl]
            for key in stale:
                del self._issued[key]
            return len(stale)

    def append_audit(self, audit: Dict[str, Any]) -> None:
        """Persist a signed, chained audit. Requires audit_db_path at construction."""
        if self._audit_db is None:
            return
        if not self.verify_audit(audit):
            raise ValueError("Cannot persist an invalid audit")
        payload = json.dumps(audit, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self._lock:
            # Fail closed: do not append to a tampered or truncated ledger.
            # Without this check, a valid append could overwrite the external
            # checkpoint and conceal a deleted tail.
            if not self.verify_audit_chain():
                raise ValueError("Audit ledger integrity check failed before append")
            row = self._audit_db.execute("SELECT mac FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
            previous = row[0] if row else "GENESIS"
            mac = hmac.new(self._secret_key, (previous + "\n" + payload).encode(), hashlib.sha256).hexdigest()
            self._audit_db.execute("INSERT INTO audit_chain (previous,payload,mac) VALUES (?,?,?)", (previous,payload,mac))
            self._audit_db.commit()
            if self._audit_anchor_path is not None:
                self._write_anchor()

    def _ledger_tip(self):
        row = self._audit_db.execute("SELECT seq,mac FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
        return [int(row[0]), row[1]] if row else [0, "GENESIS"]

    def _write_anchor(self):
        """Write an atomic signed checkpoint to a separately protected filesystem location."""
        tip = self._ledger_tip()
        content = json.dumps(tip, separators=(",", ":"))
        sig = hmac.new(self._secret_key, content.encode(), hashlib.sha256).hexdigest()
        target = os.path.abspath(self._audit_anchor_path)
        parent = os.path.dirname(target)
        fd, tmp = tempfile.mkstemp(prefix=".audit-anchor-", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                os.fchmod(out.fileno(), 0o600)
                json.dump({"tip": tip, "signature": sig}, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, target)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _verify_anchor_or_initialize(self):
        target = self._audit_anchor_path
        if not os.path.exists(target):
            if self._ledger_tip()[0] != 0:
                raise ValueError("Nonempty audit ledger missing external anchor")
            self._write_anchor()
            return
        with open(target, "r", encoding="utf-8") as source:
            anchor = json.load(source)
        tip = anchor.get("tip")
        content = json.dumps(tip, separators=(",", ":"))
        expected = hmac.new(self._secret_key, content.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, anchor.get("signature", "")) or tip != self._ledger_tip():
            raise ValueError("Audit ledger anchor mismatch or tampering detected")

    def recover_interrupted_anchor(self, anchor_path: str) -> bool:
        """Explicit operator recovery for a crash after ONE committed audit row.

        Does not recover missing, forged, or arbitrarily stale anchors. The
        caller must preserve evidence and independently confirm the outage.
        """
        with self._lock:
            if self._audit_db is None or self._audit_anchor_path is not None:
                raise ValueError("Recovery requires a database-only verification session")
            if not self.verify_audit_chain():
                raise ValueError("Audit chain is invalid; recovery refused")
            with open(anchor_path, "r", encoding="utf-8") as source:
                anchor = json.load(source)
            tip = anchor.get("tip")
            signature = anchor.get("signature")
            if (not isinstance(tip, list) or len(tip) != 2
                    or not isinstance(tip[0], int) or not isinstance(tip[1], str)
                    or not isinstance(signature, str)):
                raise ValueError("Malformed recovery anchor")
            content = json.dumps(tip, separators=(",", ":"))
            expected = hmac.new(self._secret_key, content.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise ValueError("Recovery anchor signature invalid")
            latest = self._ledger_tip()
            if latest[0] != tip[0] + 1:
                raise ValueError("Recovery requires exactly one additional committed row")
            prior = ("GENESIS" if tip[0] == 0 else
                     self._audit_db.execute(
                         "SELECT mac FROM audit_chain WHERE seq=?", (tip[0],)
                     ).fetchone()[0])
            if prior != tip[1]:
                raise ValueError("Recovery anchor is not the previous ledger tip")
            self._audit_anchor_path = anchor_path
            try:
                self._write_anchor()
                if not self.verify_audit_chain():
                    raise ValueError("Post-recovery verification failed")
            except Exception:
                self._audit_anchor_path = None
                raise
            return True

    def verify_audit_chain(self) -> bool:
        if self._audit_db is None:
            return False
        with self._lock:
            previous = "GENESIS"
            rows = self._audit_db.execute("SELECT seq,previous,payload,mac FROM audit_chain ORDER BY seq").fetchall()
            for expected_seq, (seq, prev, payload, mac) in enumerate(rows, 1):
                if seq != expected_seq or prev != previous:
                    return False
                expected = hmac.new(self._secret_key, (prev + "\n" + payload).encode(), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected, mac):
                    return False
                try:
                    if not self.verify_audit(json.loads(payload)):
                        return False
                except (ValueError, TypeError):
                    return False
                previous = mac
            # Check the external checkpoint on every explicit verification,
            # not only when opening the database. A missing or altered anchor
            # must never be reported as a valid audit chain.
            if self._audit_anchor_path is not None:
                try:
                    with open(self._audit_anchor_path, "r", encoding="utf-8") as source:
                        anchor = json.load(source)
                    tip = anchor.get("tip")
                    if not isinstance(tip, list) or len(tip) != 2:
                        return False
                    content = json.dumps(tip, separators=(",", ":"))
                    expected = hmac.new(self._secret_key, content.encode(), hashlib.sha256).hexdigest()
                    if not isinstance(anchor.get("signature"), str):
                        return False
                    if not hmac.compare_digest(expected, anchor["signature"]):
                        return False
                    if tip != self._ledger_tip():
                        return False
                except (OSError, ValueError, TypeError, AttributeError):
                    return False
            return True

    def issue_grant(self, query_id: str, target_block_id: str) -> AccessGrantToken:
        token_id = f"T_{uuid.uuid4().hex}"
        created_at = time.perf_counter()
        payload_bytes = f"{query_id}:{target_block_id}:{token_id}:{created_at:.6f}".encode('utf-8')
        signature = hmac.new(self._secret_key, payload_bytes, hashlib.sha256).hexdigest()

        token = AccessGrantToken(query_id, target_block_id, token_id, created_at, signature)
        with self._lock:
            self._issued[token_id] = (query_id, target_block_id, created_at, signature, False, False)
        return token

    def verify_grant(self, token: AccessGrantToken) -> bool:
        if not isinstance(token, AccessGrantToken):
            return False
        payload_bytes = f"{token.query_id}:{token.target_block_id}:{token.token_id}:{token.created_at:.6f}".encode('utf-8')
        expected_signature = hmac.new(self._secret_key, payload_bytes, hashlib.sha256).hexdigest()
        with self._lock:
            entry = self._issued.get(token.token_id)
            if entry is None or entry[4] or entry[5]:
                return False
            if time.perf_counter() - entry[2] > self._ttl:
                return False
            return (entry[:4] == (token.query_id, token.target_block_id, token.created_at, token.signature)
                    and hmac.compare_digest(expected_signature, token.signature))

    def sign_audit(self, audit: Dict[str, Any]) -> str:
        payload = {k: v for k, v in audit.items() if k != "audit_signature"}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        return hmac.new(self._secret_key, encoded, hashlib.sha256).hexdigest()

    def verify_audit(self, audit: Dict[str, Any]) -> bool:
        if not isinstance(audit, dict) or not isinstance(audit.get("audit_signature"), str):
            return False
        try:
            return hmac.compare_digest(audit["audit_signature"], self.sign_audit(audit))
        except (TypeError, ValueError):
            return False

    def consume_grant(self, token: AccessGrantToken) -> bool:
        with self._lock:
            if not self.verify_grant(token):
                return False
            entry = self._issued[token.token_id]
            self._issued[token.token_id] = (*entry[:4], True, entry[5])
            token.is_used = True
            return True

    def revoke_grant(self, token: AccessGrantToken):
        with self._lock:
            entry = self._issued.get(token.token_id)
            if entry is not None:
                self._issued[token.token_id] = (*entry[:5], True)
            token.revoke()


class AddressAccessViolation(Exception):
    """Raised if an unactivated or unauthorized A+ Address Block's memory payload is accessed illegally."""
    pass


class AddressBlock:
    """
    A+ Block: Independent, variable-size connection layer.
    Contains target block IDs, learned connection weights, and payload metadata.
    Enforces access control via a 5-state machine and query-scoped authorization tokens.
    """

    def __init__(self, block_id: str, connections: Optional[Dict[str, float]] = None, payload: Optional[Dict[str, Any]] = None):
        self.block_id = block_id
        self._connections: Dict[str, float] = connections if connections is not None else {}
        self._payload: Dict[str, Any] = payload if payload is not None else {}
        self.read_count: int = 0                  # Legitimate memory read tracker
        self.unauthorized_access_count: int = 0  # Security violation tracker
        self.current_state: AddressState = AddressState.NOT_SELECTED
        self.access_logs: List[Dict[str, Any]] = []

    @property
    def is_activated(self) -> bool:
        return self.current_state == AddressState.READ

    def add_connection(self, target_id: str, weight: float = 1.0):
        self._connections[target_id] = weight

    def set_state(self, new_state: AddressState):
        valid_next = VALID_STATE_TRANSITIONS.get(self.current_state, set())
        if new_state not in valid_next and new_state != self.current_state:
            raise AddressAccessViolation(
                f"Invalid state transition on block '{self.block_id}': Cannot transition from {self.current_state.value} to {new_state.value}!"
            )
        self.current_state = new_state

    def reset_state(self):
        self.current_state = AddressState.NOT_SELECTED

    def read_layer(self, grant_token: Optional[AccessGrantToken] = None, grant_manager: Optional[AccessGrantManager] = None) -> Dict[str, Any]:
        """
        Accesses connection layer and returns payload & connections.
        Requires a valid, signed AccessGrantToken matching self.block_id and grant_manager verification.
        Raises AddressAccessViolation on any state machine or token violation.
        """
        caller_frame = inspect.stack()[1]

        # 1. Token Object Verification
        if not isinstance(grant_token, AccessGrantToken):
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": getattr(grant_token, "query_id", "UNKNOWN"),
                "status": "VIOLATION_INVALID_TOKEN_TYPE",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Invalid or forged token provided!"
            )

        # 3. Token Single-Use / Double-Spend Check
        if grant_token.is_used:
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id,
                "status": "VIOLATION_TOKEN_DOUBLE_SPEND",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Grant token already consumed (double spend attempt)!"
            )

        # 2. Cryptographic Signature Verification
        if grant_manager is None or not grant_manager.verify_grant(grant_token):
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id,
                "status": "VIOLATION_FORGED_SIGNATURE",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Token HMAC signature verification failed!"
            )

        # 4. Token Revocation Check
        if grant_token.is_closed:
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id,
                "status": "VIOLATION_TOKEN_CLOSED",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': AccessGrantToken for query '{grant_token.query_id}' is closed!"
            )

        # 5. Target Block Binding Check
        if grant_token.target_block_id != self.block_id:
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id,
                "status": "VIOLATION_BLOCK_MISMATCH",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Grant token was issued for block '{grant_token.target_block_id}'!"
            )

        # 6. State Machine Enforcement Check (Disallow SELECTED directly; requires AUTHORIZED)
        if self.current_state != AddressState.AUTHORIZED:
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id,
                "status": f"VIOLATION_INVALID_STATE_{self.current_state.value}",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Block state is '{self.current_state.value}', expected AUTHORIZED!"
            )

        # Authorized Access Successful; manager enforces atomic one-time consumption.
        if not grant_manager.consume_grant(grant_token):
            self.unauthorized_access_count += 1
            raise AddressAccessViolation("Token no longer valid or already consumed")
        self.current_state = AddressState.READ
        self.read_count += 1

        self.access_logs.append({
            "timestamp": time.perf_counter(),
            "query_id": grant_token.query_id,
            "token_id": grant_token.token_id,
            "status": "AUTHORIZED_READ",
            "caller_function": caller_frame.function,
            "caller_filename": caller_frame.filename,
            "read_count": self.read_count
        })

        return {
            "block_id": self.block_id,
            "connections": dict(self._connections),
            "payload": dict(self._payload),
            "read_count": self.read_count
        }


class KnowledgeBlock:
    """
    A Block: Fixed-size N-bit knowledge kernel.
    Contains learned N-bit suitability representation.
    Decoupled from A+ address layer.
    """

    def __init__(self, block_id: str, category: str, representation: BinaryVector, address_block: AddressBlock):
        self.block_id = block_id
        self.category = category
        self.representation = representation  # Fixed N-bit vector
        self.address_block = address_block    # Decoupled A+ block
        self.evaluation_count: int = 0         # Empirical evaluation method invocation tracker

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Callable[[BinaryVector, BinaryVector], float]) -> float:
        """
        Pure low-cost mathematical suitability evaluation on A kernel.
        DOES NOT access or read A+ address block.
        """
        self.evaluation_count += 1
        return sim_func(self.representation, Q)

    def read_address_block(self, grant_token: Optional[AccessGrantToken] = None, grant_manager: Optional[AccessGrantManager] = None) -> Dict[str, Any]:
        """Triggers access to A+ connection layer upon activation using grant_token and grant_manager."""
        return self.address_block.read_layer(grant_token=grant_token, grant_manager=grant_manager)


SIMILARITY_FUNCTIONS = {
    "xor": similarity_xor_popcount,
    "binarized_dot": similarity_binarized_dot,
    "sign_cosine": similarity_sign_cosine,
}


class BlockNetwork:
    """
    Orchestrates evaluation and routing across A Knowledge Blocks and selective reading of A+ Address Blocks.
    Enforces cryptographically verified grant token issuance and state machine cleanup via try...finally.
    """

    def __init__(self, blocks: Optional[List[KnowledgeBlock]] = None, grant_manager: Optional[AccessGrantManager] = None):
        self.blocks: List[KnowledgeBlock] = blocks if blocks is not None else []
        self.grant_manager = grant_manager if grant_manager is not None else AccessGrantManager()
        self.last_query_audit: Dict[str, Any] = {}
        # The A+ state machine is shared mutable state. Serialize complete queries.
        self._query_lock = threading.RLock()

    def add_block(self, block: KnowledgeBlock):
        with self._query_lock:
            if any(existing.block_id == block.block_id for existing in self.blocks):
                raise ValueError(f"Duplicate block ID: {block.block_id}")
            self.blocks.append(block)

    def process_query(
        self,
        Q: BinaryVector,
        method: str = "xor",
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._query_lock:
            effective_query_id = query_id if query_id is not None else f"Q_{uuid.uuid4().hex[:8]}"
            try:
                return self._process_query_locked(Q, method, threshold, top_k, effective_query_id)
            except Exception as exc:
                # Persist a signed failure even when the read loop aborts early.
                # The audit is not allowed to mask the original exception.
                failed = {
                    "query_id": effective_query_id,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "selected_block_ids": [],
                    "authorized_block_ids": [],
                    "read_block_ids": [],
                    "access_events": [],
                    "unauthorized_attempts": 0,
                    "unauthorized_successful_reads": 0,
                    "a_plus_read_count": 0,
                }
                failed["audit_signature"] = self.grant_manager.sign_audit(failed)
                self.last_query_audit = failed
                try:
                    self.grant_manager.append_audit(failed)
                except Exception as audit_exc:
                    exc.add_note(f"Failure audit persistence failed: {audit_exc}")
                finally:
                    self.grant_manager.prune_grants()
                raise

    def _process_query_locked(self, Q, method, threshold, top_k, query_id):
        if query_id is None:
            query_id = f"Q_{uuid.uuid4().hex[:8]}"

        # Capture the access-log boundary for THIS query, excluding earlier queries.
        audit_offsets = {id(b.address_block): len(b.address_block.access_logs) for b in self.blocks}
        for b in self.blocks:
            b.address_block.reset_state()

        sim_func = SIMILARITY_FUNCTIONS.get(method, similarity_xor_popcount)

        # Step 1: Mathematical evaluation across A kernels
        start_eval_time = time.perf_counter()
        evaluated_scores: List[Tuple[KnowledgeBlock, float]] = []
        for block in self.blocks:
            score = block.evaluate_suitability(Q, sim_func)
            evaluated_scores.append((block, score))
        eval_latency = time.perf_counter() - start_eval_time

        # Step 2: Candidate Ranking and Selection
        start_select_time = time.perf_counter()
        evaluated_scores.sort(key=lambda x: x[1], reverse=True)

        active_blocks: List[Tuple[KnowledgeBlock, float]] = []
        if top_k is not None:
            candidate_list = evaluated_scores[:top_k]
            if threshold is not None:
                active_blocks = [item for item in candidate_list if item[1] >= threshold]
            else:
                active_blocks = candidate_list
        elif threshold is not None:
            active_blocks = [item for item in evaluated_scores if item[1] >= threshold]
        else:
            active_blocks = evaluated_scores[:1]
        select_latency = time.perf_counter() - start_select_time

        # Step 3: Issue Tokens & Read A+ Address Layers for SELECTED blocks ONLY
        start_read_time = time.perf_counter()
        accessed_a_plus_results = []
        selected_block_ids = []
        authorized_block_ids = []
        read_block_ids = []
        issued_tokens = []

        try:
            for block, score in active_blocks:
                selected_block_ids.append(block.block_id)
                block.address_block.set_state(AddressState.SELECTED)

                # Issue HMAC-signed AccessGrantToken from AccessGrantManager
                token = self.grant_manager.issue_grant(query_id=query_id, target_block_id=block.address_block.block_id)
                issued_tokens.append(token)

                block.address_block.set_state(AddressState.AUTHORIZED)
                authorized_block_ids.append(block.block_id)

                a_plus_data = block.read_address_block(grant_token=token, grant_manager=self.grant_manager)
                read_block_ids.append(block.block_id)

                accessed_a_plus_results.append({
                    "block_id": block.block_id,
                    "category": block.category,
                    "score": score,
                    "a_plus_data": a_plus_data
                })
        finally:
            # Guaranteed session cleanup in try...finally block!
            for token in issued_tokens:
                self.grant_manager.revoke_grant(token)

            for b in self.blocks:
                if b.block_id in selected_block_ids:
                    if b.address_block.current_state in [AddressState.READ, AddressState.AUTHORIZED, AddressState.SELECTED]:
                        b.address_block.set_state(AddressState.CLOSED)
                else:
                    b.address_block.reset_state()

        read_latency = time.perf_counter() - start_read_time

        # Derive audit evidence from actual AddressBlock access logs, not routing claims.
        access_events = []
        for b in self.blocks:
            for event in b.address_block.access_logs[audit_offsets[id(b.address_block)]:]:
                access_events.append({**event, "block_id": b.block_id})
        successful_reads = [e for e in access_events if e["status"] == "AUTHORIZED_READ"]
        unauthorized_attempts = sum(e["status"].startswith("VIOLATION_") for e in access_events)
        authorized_set = set(authorized_block_ids)
        selected_set = set(selected_block_ids)
        unauthorized_successful_reads = sum(
            e["block_id"] not in selected_set or e["block_id"] not in authorized_set or e.get("query_id") != query_id
            for e in successful_reads
        )
        self.last_query_audit = {
            "query_id": query_id,
            "selected_block_ids": selected_block_ids,
            "authorized_block_ids": authorized_block_ids,
            "read_block_ids": read_block_ids,
            "access_events": access_events,
            "unauthorized_attempts": unauthorized_attempts,
            "unauthorized_successful_reads": unauthorized_successful_reads,
            "a_plus_read_count": len(successful_reads),
            "a_plus_read_latency_ms": read_latency * 1000.0
        }

        self.last_query_audit["audit_signature"] = self.grant_manager.sign_audit(self.last_query_audit)
        self.grant_manager.append_audit(self.last_query_audit)
        self.grant_manager.prune_grants()

        return {
            "query_id": query_id,
            "total_a_evaluated": len(self.blocks),
            "total_a_plus_read": len(accessed_a_plus_results),
            "unaccessed_a_count": 0,
            "unaccessed_a_plus_count": len(self.blocks) - len(accessed_a_plus_results),
            "eval_latency_sec": eval_latency,
            "select_latency_sec": select_latency,
            "read_latency_sec": read_latency,
            "total_latency_sec": eval_latency + select_latency + read_latency,
            "ranked_all_scores": [(b.block_id, b.category, s) for b, s in evaluated_scores],
            "activated_results": accessed_a_plus_results,
            "audit_session": copy.deepcopy(self.last_query_audit)
        }
