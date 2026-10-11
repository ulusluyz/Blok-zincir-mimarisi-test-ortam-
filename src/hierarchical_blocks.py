"""
Enhanced Hierarchical Routing and Multi-Topic Activation Network.
Supports BaseGRepresentative implementations (Static, Learnable Single, Learnable Multi, Dynamic Multi).
"""

from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import time
import uuid
import numpy as np
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine
from src.blocks import AddressBlock, AddressState, AccessGrantToken, AccessGrantManager, KnowledgeBlock, SIMILARITY_FUNCTIONS
from src.learnable_g import BaseGRepresentative, StaticGRepresentative, LearnableSingleGRepresentative, LearnableMultiPrototypeGRepresentative, DynamicMultiPrototypeGRepresentative

# Backward compatibility alias
GroupRepresentative = StaticGRepresentative


class HierarchicalBlockNetwork:
    """
    Orchestrates Sparse Hierarchical Routing across learnable G representatives and A knowledge blocks.
    Guarantees sparse block activation and query-scoped A+ access authorization via AccessGrantManager.
    """

    def __init__(self, groups: Optional[List[BaseGRepresentative]] = None, grant_manager: Optional[AccessGrantManager] = None):
        self.groups: List[BaseGRepresentative] = groups if groups is not None else []
        self.flat_a_blocks: List[KnowledgeBlock] = []
        self.grant_manager = grant_manager if grant_manager is not None else AccessGrantManager()

    def add_group(self, group: BaseGRepresentative):
        self.groups.append(group)
        for a_block in group.member_a_blocks:
            if a_block not in self.flat_a_blocks:
                self.flat_a_blocks.append(a_block)

    def register_flat_blocks(self, blocks: List[KnowledgeBlock]):
        self.flat_a_blocks = list(blocks)

    def process_query_hierarchical_2level(
        self,
        Q: BinaryVector,
        method: str = "xor",
        group_routing_strategy: str = "dynamic_multi_label",
        group_top_k: int = 2,
        group_threshold: float = 0.50,
        confidence_margin: float = 0.08,
        min_group_confidence: float = 0.40,
        max_active_groups: int = 3,
        block_top_k: int = 1,
        sim_func: Optional[Callable[[BinaryVector, BinaryVector], float]] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if query_id is None:
            query_id = f"Q_H2_{uuid.uuid4().hex[:8]}"

        for b in self.flat_a_blocks:
            b.address_block.reset_state()

        if sim_func is None:
            sim_func = SIMILARITY_FUNCTIONS.get(method, similarity_xor_popcount)

        start_time = time.perf_counter()

        # Step 1: Evaluate G Representatives
        start_g_time = time.perf_counter()
        g_scores: List[Tuple[BaseGRepresentative, float]] = []
        for g in self.groups:
            score = g.evaluate_suitability(Q, sim_func)
            g_scores.append((g, score))
        g_eval_latency = time.perf_counter() - start_g_time

        # Step 2: Select Active G Groups
        start_g_select_time = time.perf_counter()
        g_scores.sort(key=lambda x: x[1], reverse=True)

        active_groups: List[Tuple[BaseGRepresentative, float]] = []
        fallback_triggered = False
        fallback_reason = "None"

        if group_routing_strategy == "top_k":
            active_groups = g_scores[:group_top_k]

        elif group_routing_strategy == "threshold":
            active_groups = [item for item in g_scores if item[1] >= group_threshold]
            if not active_groups:
                active_groups = g_scores[:1]
                fallback_triggered = True
                fallback_reason = "No group met threshold"

        elif group_routing_strategy == "dynamic_multi_label":
            if not g_scores:
                active_groups = []
            else:
                top1_score = g_scores[0][1]
                cutoff = max(top1_score - confidence_margin, group_threshold)
                candidates = [item for item in g_scores if item[1] >= cutoff]
                active_groups = candidates[:max_active_groups]
                if not active_groups:
                    active_groups = g_scores[:1]

        elif group_routing_strategy == "confidence_margin":
            if not g_scores:
                active_groups = []
            else:
                top1_g, top1_score = g_scores[0]
                top2_score = g_scores[1][1] if len(g_scores) > 1 else 0.0

                if top1_score < min_group_confidence:
                    active_groups = g_scores
                    fallback_triggered = True
                    fallback_reason = f"Low G top1 score ({top1_score:.3f} < {min_group_confidence:.3f})"
                elif (top1_score - top2_score) < confidence_margin:
                    active_groups = [item for item in g_scores if (top1_score - item[1]) <= confidence_margin]
                    fallback_triggered = True
                    fallback_reason = f"Close margin ({top1_score - top2_score:.3f} < {confidence_margin:.3f})"
                else:
                    active_groups = [g_scores[0]]

        else:
            active_groups = g_scores[:group_top_k]

        g_select_latency = time.perf_counter() - start_g_select_time

        # Step 3: Evaluate A Kernels ONLY inside active G groups
        start_a_eval_time = time.perf_counter()
        candidate_a_blocks: List[KnowledgeBlock] = []
        evaluated_a_set = set()

        for g_rep, _ in active_groups:
            for a_blk in g_rep.member_a_blocks:
                if a_blk not in evaluated_a_set:
                    candidate_a_blocks.append(a_blk)
                    evaluated_a_set.add(a_blk)

        evaluated_a_scores: List[Tuple[KnowledgeBlock, float]] = []
        for a_blk in candidate_a_blocks:
            score = a_blk.evaluate_suitability(Q, sim_func)
            evaluated_a_scores.append((a_blk, score))

        a_eval_latency = time.perf_counter() - start_a_eval_time

        # Step 4: Sort A candidates and select Top-K_A
        start_a_select_time = time.perf_counter()
        evaluated_a_scores.sort(key=lambda x: x[1], reverse=True)
        winning_a_blocks = evaluated_a_scores[:block_top_k]
        a_select_latency = time.perf_counter() - start_a_select_time

        # Step 5: Read A+ Address Blocks ONLY for winning A blocks with valid tokens
        start_a_plus_time = time.perf_counter()
        accessed_a_plus_results = []
        issued_tokens = []

        try:
            for a_blk, score in winning_a_blocks:
                a_blk.address_block.set_state(AddressState.SELECTED)

                token = self.grant_manager.issue_grant(query_id=query_id, target_block_id=a_blk.address_block.block_id)
                issued_tokens.append(token)
                a_blk.address_block.set_state(AddressState.AUTHORIZED)

                a_plus_data = a_blk.read_address_block(grant_token=token, grant_manager=self.grant_manager)
                accessed_a_plus_results.append({
                    "block_id": a_blk.block_id,
                    "category": a_blk.category,
                    "score": score,
                    "a_plus_data": a_plus_data
                })
        finally:
            for t in issued_tokens:
                t.revoke()
            for b in self.flat_a_blocks:
                if b.address_block.current_state != AddressState.NOT_SELECTED:
                    b.address_block.set_state(AddressState.CLOSED)

        a_plus_latency = time.perf_counter() - start_a_plus_time

        total_latency = time.perf_counter() - start_time
        total_a_in_network = len(self.flat_a_blocks) if self.flat_a_blocks else sum(len(g.member_a_blocks) for g in self.groups)

        return {
            "total_g_evaluated": len(self.groups),
            "active_g_count": len(active_groups),
            "total_a_evaluated": len(evaluated_a_scores),
            "total_a_in_network": total_a_in_network,
            "unaccessed_a_count": total_a_in_network - len(evaluated_a_scores),
            "total_a_plus_read": len(accessed_a_plus_results),
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "g_eval_latency_sec": g_eval_latency,
            "g_select_latency_sec": g_select_latency,
            "a_eval_latency_sec": a_eval_latency,
            "a_select_latency_sec": a_select_latency,
            "a_plus_latency_sec": a_plus_latency,
            "total_latency_sec": total_latency,
            "active_groups": [(g.group_id, g.name, s) for g, s in active_groups],
            "ranked_candidate_a_scores": [(b.block_id, b.category, s) for b, s in evaluated_a_scores],
            "activated_results": accessed_a_plus_results
        }

    def process_query_hierarchical_multilevel(
        self,
        Q: BinaryVector,
        method: str = "xor",
        top_k_per_level: int = 1,
        block_top_k: int = 1,
        sim_func: Optional[Callable[[BinaryVector, BinaryVector], float]] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if query_id is None:
            query_id = f"Q_HM_{uuid.uuid4().hex[:8]}"

        for b in self.flat_a_blocks:
            b.address_block.reset_state()

        if sim_func is None:
            sim_func = SIMILARITY_FUNCTIONS.get(method, similarity_xor_popcount)

        start_time = time.perf_counter()

        # Level 1: Evaluate Root Groups
        t0 = time.perf_counter()
        l1_scores = []
        for g in self.groups:
            score = g.evaluate_suitability(Q, sim_func)
            l1_scores.append((g, score))
        l1_eval_time = time.perf_counter() - t0

        l1_scores.sort(key=lambda x: x[1], reverse=True)
        active_l1_groups = l1_scores[:top_k_per_level]

        # Level 2: Evaluate Subgroups inside active Level 1 groups
        t1 = time.perf_counter()
        l2_scores = []
        for g_l1, _ in active_l1_groups:
            for sub_g in g_l1.subgroups:
                s_score = sub_g.evaluate_suitability(Q, sim_func)
                l2_scores.append((sub_g, s_score))
        l2_eval_time = time.perf_counter() - t1

        l2_scores.sort(key=lambda x: x[1], reverse=True)
        active_l2_groups = l2_scores[:top_k_per_level] if l2_scores else active_l1_groups

        # Level 3: Evaluate Leaf A Blocks inside active Level 2 subgroups
        t2 = time.perf_counter()
        candidate_a_blocks = []
        evaluated_a_set = set()
        for g_target, _ in active_l2_groups:
            for a_blk in g_target.member_a_blocks:
                if a_blk not in evaluated_a_set:
                    candidate_a_blocks.append(a_blk)
                    evaluated_a_set.add(a_blk)

        evaluated_a_scores = []
        for a_blk in candidate_a_blocks:
            score = a_blk.evaluate_suitability(Q, sim_func)
            evaluated_a_scores.append((a_blk, score))
        l3_a_eval_time = time.perf_counter() - t2

        evaluated_a_scores.sort(key=lambda x: x[1], reverse=True)
        winning_a_blocks = evaluated_a_scores[:block_top_k]

        # Read A+ for winners
        t3 = time.perf_counter()
        accessed_a_plus_results = []
        issued_tokens = []
        try:
            for a_blk, score in winning_a_blocks:
                a_blk.address_block.set_state(AddressState.SELECTED)

                token = self.grant_manager.issue_grant(query_id=query_id, target_block_id=a_blk.address_block.block_id)
                issued_tokens.append(token)
                a_blk.address_block.set_state(AddressState.AUTHORIZED)

                a_plus_data = a_blk.read_address_block(grant_token=token, grant_manager=self.grant_manager)
                accessed_a_plus_results.append({
                    "block_id": a_blk.block_id,
                    "category": a_blk.category,
                    "score": score,
                    "a_plus_data": a_plus_data
                })
        finally:
            for t in issued_tokens:
                t.revoke()
            for b in self.flat_a_blocks:
                if b.address_block.current_state != AddressState.NOT_SELECTED:
                    b.address_block.set_state(AddressState.CLOSED)

        a_plus_read_time = time.perf_counter() - t3

        total_latency = time.perf_counter() - start_time

        total_g_evaluated = len(l1_scores) + len(l2_scores)
        total_a_in_network = len(self.flat_a_blocks) if self.flat_a_blocks else sum(len(g.member_a_blocks) for g in self.groups)

        return {
            "total_g_evaluated": total_g_evaluated,
            "total_a_evaluated": len(evaluated_a_scores),
            "total_a_in_network": total_a_in_network,
            "unaccessed_a_count": total_a_in_network - len(evaluated_a_scores),
            "total_a_plus_read": len(accessed_a_plus_results),
            "l1_eval_time_sec": l1_eval_time,
            "l2_eval_time_sec": l2_eval_time,
            "l3_a_eval_time_sec": l3_a_eval_time,
            "a_plus_read_time_sec": a_plus_read_time,
            "total_latency_sec": total_latency,
            "ranked_candidate_a_scores": [(b.block_id, b.category, s) for b, s in evaluated_a_scores],
            "activated_results": accessed_a_plus_results
        }
