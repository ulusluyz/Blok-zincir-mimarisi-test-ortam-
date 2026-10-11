"""
Block Architecture Implementation with Robust Memory Access Traps:
A  — Bilgi Bloku (Knowledge Block): Fixed-size N-bit mathematical core. Evaluates suitability mathematically without activating connection layer.
A+ — Adres Bloku (Address Block): Independent, variable-size, extensible connection layer. Contains target block IDs & weights. Read ONLY when A is activated.

Includes software-level access state machine and query-scoped authorization tokens.
Note: Explicitly bounded as a Software Reference Isolation Guarantee (OS/User Space Software Logic), not a DRAM hardware-bus physical isolation.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import inspect
import uuid
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine


class AddressState(Enum):
    NOT_SELECTED = "NOT_SELECTED"
    SELECTED = "SELECTED"
    AUTHORIZED = "AUTHORIZED"
    READ = "READ"
    CLOSED = "CLOSED"


@dataclass
class AccessGrantToken:
    query_id: str
    target_block_id: str
    token_id: str
    created_at: float
    is_closed: bool = False

    def revoke(self):
        self.is_closed = True


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
        """Extensible: new connections can be added without changing fixed A block size."""
        self._connections[target_id] = weight

    def set_state(self, new_state: AddressState):
        self.current_state = new_state

    def reset_state(self):
        self.current_state = AddressState.NOT_SELECTED

    def read_layer(self, grant_token: Optional[AccessGrantToken] = None) -> Dict[str, Any]:
        """
        Accesses connection layer and returns payload & connections.
        Requires a valid, active query-scoped AccessGrantToken matching self.block_id.
        Raises AddressAccessViolation on any state machine or token violation.
        """
        caller_frame = inspect.stack()[1]

        # 1. Token Type Check
        if not isinstance(grant_token, AccessGrantToken):
            self.unauthorized_access_count += 1
            self.access_logs.append({
                "timestamp": time.perf_counter(),
                "query_id": grant_token.query_id if hasattr(grant_token, "query_id") else "UNKNOWN",
                "status": "VIOLATION_INVALID_TOKEN_TYPE",
                "caller_function": caller_frame.function,
                "caller_filename": caller_frame.filename,
                "read_count": self.read_count
            })
            raise AddressAccessViolation(
                f"Unauthorized access attempt on A+ block '{self.block_id}': Invalid or legacy token provided!"
            )

        # 2. Token Revocation / Expiration Check
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
                f"Unauthorized access attempt on A+ block '{self.block_id}': AccessGrantToken for query '{grant_token.query_id}' is expired or closed!"
            )

        # 3. Target Block ID Binding Check
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

        # 4. State Machine Transition Check
        if self.current_state not in [AddressState.SELECTED, AddressState.AUTHORIZED]:
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
                f"Unauthorized access attempt on A+ block '{self.block_id}': Block state is '{self.current_state.value}', expected SELECTED or AUTHORIZED!"
            )

        # Authorized Access Successful
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

    def read_address_block(self, grant_token: Optional[AccessGrantToken] = None) -> Dict[str, Any]:
        """Triggers access to A+ connection layer upon activation using grant_token."""
        return self.address_block.read_layer(grant_token=grant_token)


SIMILARITY_FUNCTIONS = {
    "xor": similarity_xor_popcount,
    "binarized_dot": similarity_binarized_dot,
    "sign_cosine": similarity_sign_cosine,
}


class BlockNetwork:
    """
    Orchestrates evaluation and routing across A Knowledge Blocks and selective reading of A+ Address Blocks.
    Does NOT scan or maintain a monolithic central routing table for connections.
    """

    def __init__(self, blocks: Optional[List[KnowledgeBlock]] = None):
        self.blocks: List[KnowledgeBlock] = blocks if blocks is not None else []
        self.last_query_audit: Dict[str, Any] = {}

    def add_block(self, block: KnowledgeBlock):
        self.blocks.append(block)

    def process_query(
        self,
        Q: BinaryVector,
        method: str = "xor",
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if query_id is None:
            query_id = f"Q_{uuid.uuid4().hex[:8]}"

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

        for block, score in active_blocks:
            selected_block_ids.append(block.block_id)
            block.address_block.set_state(AddressState.SELECTED)

            # Issue query-scoped AccessGrantToken matching block.address_block.block_id
            token = AccessGrantToken(
                query_id=query_id,
                target_block_id=block.address_block.block_id,
                token_id=f"T_{uuid.uuid4().hex[:8]}",
                created_at=time.perf_counter()
            )
            issued_tokens.append(token)
            block.address_block.set_state(AddressState.AUTHORIZED)
            authorized_block_ids.append(block.block_id)

            a_plus_data = block.read_address_block(grant_token=token)
            read_block_ids.append(block.block_id)

            accessed_a_plus_results.append({
                "block_id": block.block_id,
                "category": block.category,
                "score": score,
                "a_plus_data": a_plus_data
            })
        read_latency = time.perf_counter() - start_read_time

        # Cleanup: Revoke issued tokens and transition block states to CLOSED
        for token in issued_tokens:
            token.revoke()

        for b in self.blocks:
            if b.block_id in selected_block_ids:
                b.address_block.set_state(AddressState.CLOSED)
            else:
                b.address_block.reset_state()

        # Audit Record
        unauthorized_attempts = sum(b.address_block.unauthorized_access_count for b in self.blocks)
        self.last_query_audit = {
            "query_id": query_id,
            "selected_block_ids": selected_block_ids,
            "authorized_block_ids": authorized_block_ids,
            "read_block_ids": read_block_ids,
            "unauthorized_attempts": unauthorized_attempts,
            "unauthorized_successful_reads": 0,
            "a_plus_read_count": len(accessed_a_plus_results),
            "a_plus_read_latency_ms": read_latency * 1000.0
        }

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
            "audit_session": self.last_query_audit
        }
