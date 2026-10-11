"""
Unit tests for Learnable Relational A+ Layer & Multi-Hop Inference Engine.
"""

import pytest
from src.relational_inference import (
    RelationType, RelationalAddressLayer, RelationalInferenceEngine
)


def test_relational_address_layer():
    layer = RelationalAddressLayer(source_block_id="A")
    layer.add_link("B", RelationType.IS_A, weight=0.9)
    layer.add_link("C", RelationType.PART_OF, weight=0.8)

    assert len(layer.links) == 2
    is_a_links = layer.get_links_by_type(RelationType.IS_A)
    assert len(is_a_links) == 1
    assert is_a_links[0].target_block_id == "B"


def test_multi_hop_deduction_3_hops():
    # Setup chain: Aslan -> Memeli -> Hayvan -> Canlı
    engine = RelationalInferenceEngine(decay_factor=0.9, max_hops=5)
    engine.add_relation("Aslan", "Memeli", RelationType.IS_A, weight=1.0)
    engine.add_relation("Memeli", "Hayvan", RelationType.IS_A, weight=1.0)
    engine.add_relation("Hayvan", "Canlı", RelationType.IS_A, weight=1.0)

    deductions = engine.find_all_deductions("Aslan", "Canlı")
    assert len(deductions) > 0
    best = deductions[0]
    assert best.hop_count == 3
    assert best.start_block_id == "Aslan"
    assert best.target_block_id == "Canlı"
    assert abs(best.total_confidence - (0.9 ** 3)) < 1e-4


def test_transitive_generic_deduction():
    # Setup chain: A > B, B > C, C > D, D > E (4 hops)
    engine = RelationalInferenceEngine(decay_factor=0.8, max_hops=5)
    engine.add_relation("A", "B", RelationType.TRANSITIVE_GENERIC, weight=1.0)
    engine.add_relation("B", "C", RelationType.TRANSITIVE_GENERIC, weight=1.0)
    engine.add_relation("C", "D", RelationType.TRANSITIVE_GENERIC, weight=1.0)
    engine.add_relation("D", "E", RelationType.TRANSITIVE_GENERIC, weight=1.0)

    deductions = engine.find_all_deductions("A", "E")
    assert len(deductions) == 1
    d = deductions[0]
    assert d.hop_count == 4
    assert d.is_valid_inference is True
