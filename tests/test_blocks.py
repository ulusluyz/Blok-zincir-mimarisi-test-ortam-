"""
Unit tests for Block Architecture (A and A+), decoupled memory tracking, access traps, and Top-K / Threshold activation logic.
"""

import pytest
from src.bitwise import BinaryVector
from src.blocks import AddressBlock, AddressAccessViolation, KnowledgeBlock, BlockNetwork


def test_address_block_decoupling_and_read_count():
    a_plus_1 = AddressBlock("block_1", payload={"info": "Selamlaşma bloku icerigi"})
    a_plus_2 = AddressBlock("block_2", payload={"info": "Gezegenler bloku icerigi"})

    v1 = BinaryVector(0b11110000, bit_length=64)
    v2 = BinaryVector(0b00001111, bit_length=64)

    b1 = KnowledgeBlock("b1", "Selamlaşma", v1, a_plus_1)
    b2 = KnowledgeBlock("b2", "Gezegenler", v2, a_plus_2)

    network = BlockNetwork([b1, b2])

    # Initial read count must be 0
    assert a_plus_1.read_count == 0
    assert a_plus_2.read_count == 0
    assert not a_plus_1.is_activated
    assert not a_plus_2.is_activated

    # Query matching b1 (0b11110000)
    Q = BinaryVector(0b11110000, bit_length=64)
    result = network.process_query(Q, method="xor", top_k=1)

    assert result["total_a_evaluated"] == 2
    assert result["total_a_plus_read"] == 1
    assert result["unaccessed_a_plus_count"] == 1

    # Check actual read_count and activation properties on AddressBlocks
    assert a_plus_1.read_count == 1
    assert a_plus_1.is_activated is True
    assert len(a_plus_1.access_logs) == 1

    assert a_plus_2.read_count == 0, "Irrelevant A+ block MUST NOT be read!"
    assert a_plus_2.is_activated is False
    assert len(a_plus_2.access_logs) == 0


def test_extensibility_of_address_block():
    a_plus = AddressBlock("block_exp")
    assert len(a_plus._connections) == 0

    # Add dynamic connections without changing fixed A kernel
    a_plus.add_connection("target_block_99", weight=0.85)
    assert a_plus._connections["target_block_99"] == 0.85
