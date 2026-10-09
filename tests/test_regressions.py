"""
Regression Test Suite to prevent historical bugs from re-occurring:
1. Prototype Collision Bug (Modulo-7 generator collision across 10 categories).
2. Candidate Truncation Bug (top_k=1 candidate list truncation in process_query).
"""

import pytest
from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock, BlockNetwork


def test_no_prototype_collisions_across_all_categories():
    """Verify that no two category target prototypes have 1.0 similarity (100% collision)."""
    enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
    prototypes = enc.category_prototypes

    num_cats = len(CATEGORIES)
    collisions = []
    for i in range(num_cats):
        for j in range(i + 1, num_cats):
            sim = similarity_xor_popcount(prototypes[i], prototypes[j])
            if sim >= 0.95:  # Near collision or exact collision threshold
                collisions.append((i, CATEGORIES[i], j, CATEGORIES[j], sim))

    assert len(collisions) == 0, f"Found category prototype collisions: {collisions}"


def test_top_k_candidate_list_truncation():
    """Verify that process_query(top_k=3) returns up to 3 candidates, not truncated to 1."""
    blocks = []
    for i in range(5):
        vec = BinaryVector(i * 100, bit_length=64)
        ap = AddressBlock(f"ap_{i}")
        kb = KnowledgeBlock(f"kb_{i}", f"Cat_{i}", vec, ap)
        blocks.append(kb)

    net = BlockNetwork(blocks)
    Q = BinaryVector(0, bit_length=64)

    res_top1 = net.process_query(Q, method="xor", top_k=1)
    assert len(res_top1["activated_results"]) == 1

    res_top3 = net.process_query(Q, method="xor", top_k=3)
    assert len(res_top3["activated_results"]) == 3
