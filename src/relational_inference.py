"""
Learnable Relational A+ Layer & Multi-Hop Inference Engine.

This module extends the standard A+ address layer with relational semantics
(e.g., IS_A, PART_OF, CAUSE_OF, PREREQUISITE_OF, OPPOSITE_OF, TRANSITIVE_GENERIC)
and provides a CPU-native graph traversal engine for multi-hop transitive deductions
(2 to 5 hops) across Knowledge Blocks without Transformer self-attention.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Set, Tuple, Optional
import math


class RelationType(Enum):
    IS_A = "IS_A"                        # Hierarchy / Subclass (e.g., Aslan -> Memeli)
    PART_OF = "PART_OF"                  # Meronymy (e.g., Motor -> Araba)
    CAUSE_OF = "CAUSE_OF"                # Causality (e.g., Yağmur -> Islaklık)
    PREREQUISITE_OF = "PREREQUISITE_OF"  # Dependency (e.g., Un -> Ekmek)
    OPPOSITE_OF = "OPPOSITE_OF"          # Antonym / Contradiction
    DEPENDS_ON = "DEPENDS_ON"            # Logic Dependency
    TRANSITIVE_GENERIC = "TRANSITIVE_GENERIC"  # Generic transitive relation (A > B, B > C => A > C)


@dataclass
class RelationalLink:
    target_block_id: str
    relation_type: RelationType
    weight: float = 1.0  # Confidence / Association strength in [0, 1]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RelationalAddressLayer:
    """
    Relational A+ address layer attached to a specific A Knowledge Block.
    """
    source_block_id: str
    links: List[RelationalLink] = field(default_factory=list)

    def add_link(self, target_block_id: str, relation_type: RelationType, weight: float = 1.0, metadata: Dict = None):
        if metadata is None:
            metadata = {}
        # Avoid duplicate links of same type to same target, update weight if exists
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
    """
    def __init__(self, decay_factor: float = 0.85, max_hops: int = 5):
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.a_plus_layers: Dict[str, RelationalAddressLayer] = {}

    def register_a_plus_layer(self, layer: RelationalAddressLayer):
        self.a_plus_layers[layer.source_block_id] = layer

    def add_relation(self, source_id: str, target_id: str, relation_type: RelationType, weight: float = 1.0, metadata: Dict = None):
        if source_id not in self.a_plus_layers:
            self.a_plus_layers[source_id] = RelationalAddressLayer(source_block_id=source_id)
        self.a_plus_layers[source_id].add_link(target_id, relation_type, weight, metadata)

    def find_all_deductions(
        self,
        start_block_id: str,
        target_block_id: Optional[str] = None,
        allowed_relations: Optional[Set[RelationType]] = None,
        min_confidence: float = 0.1,
        max_depth: Optional[int] = None
    ) -> List[MultiHopDeductionResult]:
        """
        Find deduction paths starting from `start_block_id`.
        Uses Breadth-First / Depth-First search with cycle detection.
        """
        if max_depth is None:
            max_depth = self.max_hops

        results: List[MultiHopDeductionResult] = []

        if start_block_id not in self.a_plus_layers:
            return results

        # Stack entries: (current_block_id, current_path, current_confidence, visited_set)
        stack: List[Tuple[str, List[DeductionHop], float, Set[str]]] = [
            (start_block_id, [], 1.0, {start_block_id})
        ]

        while stack:
            curr_id, path, curr_conf, visited = stack.pop()

            if target_block_id and curr_id == target_block_id and len(path) > 0:
                explanation = " -> ".join(
                    [f"{h.from_block_id} --[{h.relation_type.value}]--> {h.to_block_id}" for h in path]
                )
                results.append(MultiHopDeductionResult(
                    start_block_id=start_block_id,
                    target_block_id=curr_id,
                    path=path,
                    total_confidence=curr_conf,
                    hop_count=len(path),
                    is_valid_inference=True,
                    explanation=explanation
                ))
                # Continue searching for other potential paths if bounded
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
                    continue  # Cycle prevention

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
                    explanation = " -> ".join(
                        [f"{h.from_block_id} --[{h.relation_type.value}]--> {h.to_block_id}" for h in new_path]
                    )
                    results.append(MultiHopDeductionResult(
                        start_block_id=start_block_id,
                        target_block_id=link.target_block_id,
                        path=new_path,
                        total_confidence=next_conf,
                        hop_count=len(new_path),
                        is_valid_inference=True,
                        explanation=explanation
                    ))

                stack.append((link.target_block_id, new_path, next_conf, new_visited))

        # Sort results by confidence descending
        results.sort(key=lambda r: r.total_confidence, reverse=True)
        return results
