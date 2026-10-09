"""
Unit tests for Group Representative (G) and Hierarchical Block Network.
Verifies sparse activation, access traps, 2-level and multi-level routing flows.
"""

import pytest
from src.bitwise import BinaryVector
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def test_group_representative_and_sparse_activation():
    # Group 1: Selamlaşma
    g1_vec = BinaryVector(0b11110000, bit_length=64)
    g1 = GroupRepresentative("g1", "Selamlaşma Group", g1_vec)

    ap1 = AddressBlock("ap1", payload={"cat": "Selamlaşma"})
    a1 = KnowledgeBlock("a1", "Selamlaşma", g1_vec, ap1)
    g1.add_member_a_block(a1)

    # Group 2: Gezegenler
    g2_vec = BinaryVector(0b00001111, bit_length=64)
    g2 = GroupRepresentative("g2", "Gezegenler Group", g2_vec)

    ap2 = AddressBlock("ap2", payload={"cat": "Gezegenler"})
    a2 = KnowledgeBlock("a2", "Gezegenler", g2_vec, ap2)
    g2.add_member_a_block(a2)

    h_net = HierarchicalBlockNetwork([g1, g2])

    # Query matching Group 1 (0b11110000)
    Q = BinaryVector(0b11110000, bit_length=64)

    # Single active group top_k=1
    res = h_net.process_query_hierarchical_2level(
        Q, method="xor", group_routing_strategy="top_k", group_top_k=1, block_top_k=1
    )

    # G evaluations = 2, A evaluations = 1 (A2 in Group 2 was NEVER evaluated!)
    assert res["total_g_evaluated"] == 2
    assert res["total_a_evaluated"] == 1
    assert res["total_a_plus_read"] == 1
    assert res["unaccessed_a_count"] == 1

    # Verify A+ read counts and access isolation
    assert ap1.read_count == 1
    assert ap1.is_activated is True
    assert ap2.read_count == 0, "Unselected group's A+ block MUST NOT be read!"
    assert ap2.is_activated is False


def test_multilevel_hierarchical_routing():
    # Root Group 1
    root_vec = BinaryVector(0b11000000, bit_length=64)
    root_g = GroupRepresentative("root_1", "Science Group", root_vec)

    # Subgroup 1: Physics
    sub1_vec = BinaryVector(0b11100000, bit_length=64)
    sub1_g = GroupRepresentative("sub_1", "Physics Group", sub1_vec)

    ap_phys = AddressBlock("ap_phys", payload={"topic": "Physics"})
    a_phys = KnowledgeBlock("a_phys", "Physics", sub1_vec, ap_phys)
    sub1_g.add_member_a_block(a_phys)

    root_g.add_subgroup(sub1_g)

    h_net = HierarchicalBlockNetwork([root_g])

    Q = BinaryVector(0b11100000, bit_length=64)
    res = h_net.process_query_hierarchical_multilevel(Q, method="xor", top_k_per_level=1, block_top_k=1)

    assert res["total_g_evaluated"] == 2  # 1 root + 1 subgroup
    assert res["total_a_evaluated"] == 1
    assert res["total_a_plus_read"] == 1
    assert ap_phys.read_count == 1
