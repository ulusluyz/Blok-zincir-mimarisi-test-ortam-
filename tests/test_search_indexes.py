"""
Unit tests for comparative auxiliary indexes (LSH, IVF, HNSW).
"""

import pytest
from src.bitwise import BinaryVector
from src.blocks import AddressBlock, KnowledgeBlock
from src.search_indexes import LSHIndex, IVFIndex, HNSWIndex


def create_sample_blocks():
    blocks = []
    # 8 blocks with distinct patterns
    for i in range(8):
        v = BinaryVector((i * 31) & 0xFFFFFFFF, bit_length=64)
        ap = AddressBlock(f"ap_{i}")
        kb = KnowledgeBlock(f"kb_{i}", f"Cat_{i % 3}", v, ap)
        blocks.append(kb)
    return blocks


def test_lsh_index():
    blocks = create_sample_blocks()
    lsh = LSHIndex(num_tables=3, hash_size=4, bit_length=64)
    for b in blocks:
        lsh.add_block(b)

    Q = BinaryVector(0b1111, bit_length=64)
    candidates = lsh.query_candidates(Q, fallback_all=blocks)
    assert len(candidates) > 0


def test_ivf_index():
    blocks = create_sample_blocks()
    ivf = IVFIndex(num_centroids=3, bit_length=64)
    ivf.fit_and_build(blocks)

    Q = BinaryVector(0b1111, bit_length=64)
    candidates = ivf.query_candidates(Q, nprobe=2)
    assert len(candidates) > 0


def test_hnsw_index():
    blocks = create_sample_blocks()
    hnsw = HNSWIndex(M=4)
    hnsw.build_graph(blocks)

    Q = BinaryVector(0b1111, bit_length=64)
    candidates = hnsw.query_candidates(Q, ef_search=4)
    assert len(candidates) > 0
