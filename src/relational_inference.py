"""
Learnable Relational A+ Layer & Multi-Hop Inference Engine.

This module extends the standard A+ address layer with formal relational semantics:
- Transitivity (e.g., IS_A, TRANSITIVE_GENERIC vs non-transitive KNOWS)
- Symmetry and Directionality
- Autonomous concept/entity extraction from natural language query text
- CPU-native graph traversal for multi-hop transitive deductions (2 to 5 hops)
- Rejection of invalid non-transitive deductions (e.g. Ali -> Can -> Efe does NOT imply Ali -> Efe).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Set, Tuple, Optional
import math
import re
from src.features import normalize_turkish


class RelationType(Enum):
    IS_A = "IS_A"                        # Subclass / Superclass (Transitive, Directed)
    PART_OF = "PART_OF"                  # Meronymy (Transitive, Directed)
    CAUSE_OF = "CAUSE_OF"                # Causality (Transitive, Directed)
    PREREQUISITE_OF = "PREREQUISITE_OF"  # Dependency (Transitive, Directed)
    OPPOSITE_OF = "OPPOSITE_OF"          # Contradiction (Non-transitive, Symmetric)
    DEPENDS_ON = "DEPENDS_ON"            # Logic Dependency (Transitive, Directed)
    TRANSITIVE_GENERIC = "TRANSITIVE_GENERIC"  # Generic transitive (A > B, B > C => A > C)
    KNOWS = "KNOWS"                      # Non-transitive acquaintance (A knows B, B knows C !=> A knows C)


@dataclass
class RelationSemantics:
    is_transitive: bool
    is_symmetric: bool
    is_directed: bool


RELATION_SEMANTICS_MAP: Dict[RelationType, RelationSemantics] = {
    RelationType.IS_A: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.PART_OF: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.CAUSE_OF: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.PREREQUISITE_OF: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.OPPOSITE_OF: RelationSemantics(is_transitive=False, is_symmetric=True, is_directed=False),
    RelationType.DEPENDS_ON: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.TRANSITIVE_GENERIC: RelationSemantics(is_transitive=True, is_symmetric=False, is_directed=True),
    RelationType.KNOWS: RelationSemantics(is_transitive=False, is_symmetric=True, is_directed=False),
}


@dataclass
class RelationalLink:
    target_block_id: str
    relation_type: RelationType
    weight: float = 1.0  # Confidence / Association strength in [0, 1]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RelationalAddressLayer:
    source_block_id: str
    links: List[RelationalLink] = field(default_factory=list)

    def add_link(self, target_block_id: str, relation_type: RelationType, weight: float = 1.0, metadata: Dict = None):
        if metadata is None:
            metadata = {}
        for l in self.links:
            if l.target_block_id == target_block_id and l.relation_type == relation_type:
                l.weight = weight
                l.metadata.update(metadata)
                return
        self.links.append(RelationalLink(
            target_block_id=target_block_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata
        ))

    def get_links_by_type(self, relation_type: Optional[RelationType] = None) -> List[RelationalLink]:
        if relation_type is None:
            return list(self.links)
        return [l for l in self.links if l.relation_type == relation_type]


@dataclass
class DeductionHop:
    from_block_id: str
    to_block_id: str
    relation_type: RelationType
    edge_weight: float
    accumulated_confidence: float


@dataclass
class MultiHopDeductionResult:
    start_block_id: str
    target_block_id: str
    path: List[DeductionHop]
    total_confidence: float
    hop_count: int
    is_valid_inference: bool
    explanation: str


class RelationalInferenceEngine:
    """
    Multi-hop relational deduction engine traversing A+ address layers.
    Enforces formal transitivity rules to reject invalid deductions.
    """
    def __init__(self, decay_factor: float = 0.85, max_hops: int = 5):
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.a_plus_layers: Dict[str, RelationalAddressLayer] = {}
        self.known_concepts: Set[str] = set()

    def register_a_plus_layer(self, layer: RelationalAddressLayer):
        self.a_plus_layers[layer.source_block_id] = layer
        self.known_concepts.add(layer.source_block_id)
        for l in layer.links:
            self.known_concepts.add(l.target_block_id)

    def add_relation(self, source_id: str, target_id: str, relation_type: RelationType, weight: float = 1.0, metadata: Dict = None):
        if source_id not in self.a_plus_layers:
            self.a_plus_layers[source_id] = RelationalAddressLayer(source_block_id=source_id)
        self.a_plus_layers[source_id].add_link(target_id, relation_type, weight, metadata)
        self.known_concepts.add(source_id)
        self.known_concepts.add(target_id)

    def extract_concepts_from_query(self, query_text: str) -> List[str]:
        """
        Autonomously extracts known relational concepts present in query text without receiving ground truth annotations.
        """
        norm = normalize_turkish(query_text)
        found = []
        # Check known concepts against query text
        for concept in sorted(self.known_concepts, key=len, reverse=True):
            concept_norm = normalize_turkish(concept)
            if concept_norm in norm:
                found.append(concept)

        return found

    def find_all_deductions(
        self,
        start_block_id: str,
        target_block_id: Optional[str] = None,
        allowed_relations: Optional[Set[RelationType]] = None,
        min_confidence: float = 0.01,
        max_depth: Optional[int] = None
    ) -> List[MultiHopDeductionResult]:
        if max_depth is None:
            max_depth = self.max_hops

        results: List[MultiHopDeductionResult] = []

        if start_block_id not in self.a_plus_layers:
            return results

        stack: List[Tuple[str, List[DeductionHop], float, Set[str]]] = [
            (start_block_id, [], 1.0, {start_block_id})
        ]

        while stack:
            curr_id, path, curr_conf, visited = stack.pop()

            if target_block_id and curr_id == target_block_id and len(path) > 0:
                is_valid = True
                invalid_reason = ""
                if len(path) > 1:
                    for h in path:
                        sem = RELATION_SEMANTICS_MAP.get(h.relation_type, RelationSemantics(is_transitive=False, is_symmetric=False, is_directed=True))
                        if not sem.is_transitive:
                            is_valid = False
                            invalid_reason = f"Relational link '{h.relation_type.value}' is non-transitive. Multi-hop deduction rejected."
                            break

                path_str = " -> ".join([f"{h.from_block_id} --[{h.relation_type.value}]--> {h.to_block_id}" for h in path])
                explanation = path_str if is_valid else f"INVALID DEDUCTION: {invalid_reason} ({path_str})"

                results.append(MultiHopDeductionResult(
                    start_block_id=start_block_id,
                    target_block_id=curr_id,
                    path=path,
                    total_confidence=curr_conf if is_valid else 0.0,
                    hop_count=len(path),
                    is_valid_inference=is_valid,
                    explanation=explanation
                ))
                if len(results) >= 50:
                    break
                continue

            if len(path) >= max_depth:
                continue

            layer = self.a_plus_layers.get(curr_id)
            if not layer:
                continue

            for link in layer.links:
                if allowed_relations and link.relation_type not in allowed_relations:
                    continue

                if link.target_block_id in visited:
                    continue

                next_conf = curr_conf * link.weight * self.decay_factor
                if next_conf < min_confidence:
                    continue

                new_hop = DeductionHop(
                    from_block_id=curr_id,
                    to_block_id=link.target_block_id,
                    relation_type=link.relation_type,
                    edge_weight=link.weight,
                    accumulated_confidence=next_conf
                )

                new_path = path + [new_hop]
                new_visited = visited | {link.target_block_id}

                if target_block_id is None and len(new_path) > 0:
                    is_valid = True
                    invalid_reason = ""
                    if len(new_path) > 1:
                        for h in new_path:
                            sem = RELATION_SEMANTICS_MAP.get(h.relation_type, RelationSemantics(is_transitive=False, is_symmetric=False, is_directed=True))
                            if not sem.is_transitive:
                                is_valid = False
                                invalid_reason = f"Relational link '{h.relation_type.value}' is non-transitive."
                                break

                    path_str = " -> ".join([f"{h.from_block_id} --[{h.relation_type.value}]--> {h.to_block_id}" for h in new_path])
                    explanation = path_str if is_valid else f"INVALID DEDUCTION: {invalid_reason} ({path_str})"

                    results.append(MultiHopDeductionResult(
                        start_block_id=start_block_id,
                        target_block_id=link.target_block_id,
                        path=new_path,
                        total_confidence=next_conf if is_valid else 0.0,
                        hop_count=len(new_path),
                        is_valid_inference=is_valid,
                        explanation=explanation
                    ))

                stack.append((link.target_block_id, new_path, next_conf, new_visited))

        results.sort(key=lambda r: r.total_confidence, reverse=True)
        return results
