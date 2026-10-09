"""
Unit tests for Multi-Group Activation, Dynamic Thresholds, and Confidence Fallback Mechanisms.
"""

import pytest
from src.bitwise import BinaryVector
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def build_sample_network():
    # G1: Elektronik (0b11110000)
    g1_vec = BinaryVector(0b11110000, bit_length=64)
    g1 = GroupRepresentative("g1", "Elektronik", g1_vec)
    ap1 = AddressBlock("ap1", payload={"cat": "Elektronik"})
    a1 = KnowledgeBlock("a1", "Elektronik", g1_vec, ap1)
    g1.add_member_a_block(a1)

    # G2: Fizik (0b11100000) -> Close to Elektronik (1-bit diff out of 64 -> 0.984 similarity)
    g2_vec = BinaryVector(0b11100000, bit_length=64)
    g2 = GroupRepresentative("g2", "Fizik", g2_vec)
    ap2 = AddressBlock("ap2", payload={"cat": "Fizik"})
    a2 = KnowledgeBlock("a2", "Fizik", g2_vec, ap2)
    g2.add_member_a_block(a2)

    # G3: Selamlaşma (0b00001111) -> Distant
    g3_vec = BinaryVector(0b00001111, bit_length=64)
    g3 = GroupRepresentative("g3", "Selamlaşma", g3_vec)
    ap3 = AddressBlock("ap3", payload={"cat": "Selamlaşma"})
    a3 = KnowledgeBlock("a3", "Selamlaşma", g3_vec, ap3)
    g3.add_member_a_block(a3)

    return HierarchicalBlockNetwork([g1, g2, g3])


def test_top_k_multi_group_routing():
    h_net = build_sample_network()
    # Query matching Elektronik / Fizik (0b11110000)
    Q = BinaryVector(0b11110000, bit_length=64)

    # Select Top-2 Groups
    res = h_net.process_query_hierarchical_2level(
        Q, method="xor", group_routing_strategy="top_k", group_top_k=2, block_top_k=1
    )

    assert res["active_g_count"] == 2
    assert res["total_a_evaluated"] == 2
    active_g_names = [item[1] for item in res["active_groups"]]
    assert "Elektronik" in active_g_names
    assert "Fizik" in active_g_names


def test_dynamic_threshold_group_routing():
    h_net = build_sample_network()
    Q = BinaryVector(0b11110000, bit_length=64)

    # Threshold = 0.99 (Only Elektronik score 1.0 >= 0.99 meets threshold; Fizik score ~0.984 does not)
    res = h_net.process_query_hierarchical_2level(
        Q, method="xor", group_routing_strategy="threshold", group_threshold=0.99, block_top_k=1
    )

    assert res["active_g_count"] == 1
    assert res["active_groups"][0][1] == "Elektronik"


def test_confidence_margin_fallback_on_ambiguous_query():
    h_net = build_sample_network()
    # Query with close margin between Elektronik and Fizik
    Q = BinaryVector(0b11111000, bit_length=64)

    # If confidence_margin = 0.10, margin (top1_score - top2_score) < 0.10 triggers Fallback to open multiple groups
    res = h_net.process_query_hierarchical_2level(
        Q, method="xor", group_routing_strategy="confidence_margin", confidence_margin=0.10, block_top_k=1
    )

    assert res["fallback_triggered"] is True
    assert "Close margin" in res["fallback_reason"] or "Low G" in res["fallback_reason"]
    assert res["active_g_count"] >= 2
