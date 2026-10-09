"""
Comparative Auxiliary Search Indexes & Approximate Nearest Neighbor Routers:
1. LSHIndex: Locality-Sensitive Hashing Buckets for candidate pre-filtering.
2. IVFIndex: Inverted File Index (Coarse quantization centroids + postings list).
3. HNSWIndex: Small-World Graph-based ANN Router for binary vector space.

Note: These are auxiliary comparative indexes evaluated alongside our primary Learned Hierarchical G Architecture.
"""

import random
import math
from typing import List, Dict, Set, Tuple, Optional
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.blocks import KnowledgeBlock


class LSHIndex:
    """
    Locality-Sensitive Hashing (LSH) Bucket Pre-filter.
    Hashes N-bit BinaryVectors into hash buckets using bit mask projections.
    """

    def __init__(self, num_tables: int = 4, hash_size: int = 8, bit_length: int = 128, seed: int = 42):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bit_length = bit_length
        rng = random.Random(seed)

        # Random bit index subsets for each table
        self.tables_bit_indices: List[List[int]] = [
            rng.sample(range(bit_length), hash_size)
            for _ in range(num_tables)
        ]
        # Hash tables: table_idx -> dict(hash_key -> list(KnowledgeBlock))
        self.hash_tables: List[Dict[str, List[KnowledgeBlock]]] = [
            {} for _ in range(num_tables)
        ]

    def _hash_vector(self, vec: BinaryVector, table_idx: int) -> str:
        indices = self.tables_bit_indices[table_idx]
        bits = vec.to_bool_list()
        return "".join('1' if bits[idx] else '0' for idx in indices)

    def add_block(self, block: KnowledgeBlock):
        for t_idx in range(self.num_tables):
            key = self._hash_vector(block.representation, t_idx)
            if key not in self.hash_tables[t_idx]:
                self.hash_tables[t_idx][key] = []
            self.hash_tables[t_idx][key].append(block)

    def query_candidates(self, Q: BinaryVector, fallback_all: List[KnowledgeBlock]) -> List[KnowledgeBlock]:
        candidates = set()
        for t_idx in range(self.num_tables):
            key = self._hash_vector(Q, t_idx)
            bucket = self.hash_tables[t_idx].get(key, [])
            for block in bucket:
                candidates.add(block)

        if not candidates:
            # Fallback if LSH buckets are empty
            return list(fallback_all)
        return list(candidates)


class IVFIndex:
    """
    Inverted File (IVF) Index for Binary Vector Space.
    Clustering coarse centroids into inverted lists (posting lists).
    """

    def __init__(self, num_centroids: int = 8, bit_length: int = 128, seed: int = 42):
        self.num_centroids = num_centroids
        self.bit_length = bit_length
        self.seed = seed
        self.centroids: List[BinaryVector] = []
        self.inverted_lists: Dict[int, List[KnowledgeBlock]] = {}

    def fit_and_build(self, blocks: List[KnowledgeBlock]):
        if not blocks:
            return

        rng = random.Random(self.seed)
        sample_blocks = rng.sample(blocks, min(self.num_centroids, len(blocks)))
        self.centroids = [b.representation for b in sample_blocks]
        self.inverted_lists = {i: [] for i in range(len(self.centroids))}

        # Assign each block to nearest centroid
        for block in blocks:
            best_c_idx = 0
            best_sim = -1.0
            for c_idx, c_vec in enumerate(self.centroids):
                sim = similarity_xor_popcount(c_vec, block.representation)
                if sim > best_sim:
                    best_sim = sim
                    best_c_idx = c_idx
            self.inverted_lists[best_c_idx].append(block)

    def query_candidates(self, Q: BinaryVector, nprobe: int = 2) -> List[KnowledgeBlock]:
        if not self.centroids:
            return []

        # Find top nprobe centroids
        centroid_scores = [
            (c_idx, similarity_xor_popcount(c_vec, Q))
            for c_idx, c_vec in enumerate(self.centroids)
        ]
        centroid_scores.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for c_idx, _ in centroid_scores[:nprobe]:
            candidates.extend(self.inverted_lists.get(c_idx, []))
        return candidates


class HNSWIndex:
    """
    Small-World Graph-based Approximate Nearest Neighbor (ANN) Index.
    Connects KnowledgeBlock nodes to M nearest neighbors in Hamming space.
    """

    def __init__(self, M: int = 8, seed: int = 42):
        self.M = M
        self.seed = seed
        self.graph: Dict[KnowledgeBlock, List[KnowledgeBlock]] = {}
        self.entry_point: Optional[KnowledgeBlock] = None

    def build_graph(self, blocks: List[KnowledgeBlock]):
        if not blocks:
            return

        self.graph = {b: [] for b in blocks}
        self.entry_point = blocks[0]

        # Connect each node to top-M closest blocks in dataset
        for b in blocks:
            neighbors = [
                (other, similarity_xor_popcount(b.representation, other.representation))
                for other in blocks if other != b
            ]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            self.graph[b] = [n[0] for n in neighbors[:self.M]]

    def query_candidates(self, Q: BinaryVector, ef_search: int = 16) -> List[KnowledgeBlock]:
        if not self.entry_point:
            return []

        visited = {self.entry_point}
        candidates = [(self.entry_point, similarity_xor_popcount(self.entry_point.representation, Q))]
        result_set = list(candidates)

        while candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            curr_node, curr_sim = candidates.pop(0)

            for neighbor in self.graph.get(curr_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    sim = similarity_xor_popcount(neighbor.representation, Q)
                    candidates.append((neighbor, sim))
                    result_set.append((neighbor, sim))

                    if len(result_set) >= ef_search * 2:
                        break

        result_set.sort(key=lambda x: x[1], reverse=True)
        return [b for b, _ in result_set[:ef_search]]
