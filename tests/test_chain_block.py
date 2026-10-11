"""
Comprehensive Unit Test Suite for Chain-Block Architecture.
"""

import pytest
from src.dataset import get_full_split_datasets, CATEGORIES
from src.dataset_chain_block import get_chain_block_benchmark_dataset
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.blocks import KnowledgeBlock, AddressBlock, BlockNetwork
from src.meaning_encoder import MeaningFeatureExtractor, MeaningBinaryEncoder, detect_turkish_intent_and_polarity
from src.relational_inference import RelationalInferenceEngine, RelationType
from src.cognitive_control import CognitiveUncertaintyAuditor, DecisionType


def test_chain_block_dataset_structure():
    cases = get_chain_block_benchmark_dataset()
    assert len(cases) > 20
    negated_cases = [c for c in cases if c.is_negated]
    assert len(negated_cases) >= 3
    multihop_cases = [c for c in cases if c.relation_chain]
    assert len(multihop_cases) >= 4


def test_intent_polarity_and_meaning_encoder():
    p1 = detect_turkish_intent_and_polarity("Gel")
    assert p1["is_negated"] is False

    p2 = detect_turkish_intent_and_polarity("Gelme")
    assert p2["is_negated"] is True

    p3 = detect_turkish_intent_and_polarity("5 ile 5 i çarpma")
    assert p3["is_negated"] is True


def test_relational_inference_engine_multihop():
    engine = RelationalInferenceEngine(decay_factor=0.9, max_hops=5)
    engine.add_relation("A", "B", RelationType.IS_A, weight=1.0)
    engine.add_relation("B", "C", RelationType.IS_A, weight=1.0)
    engine.add_relation("C", "D", RelationType.IS_A, weight=1.0)
    engine.add_relation("D", "E", RelationType.IS_A, weight=1.0)

    deductions = engine.find_all_deductions("A", "E")
    assert len(deductions) == 1
    d = deductions[0]
    assert d.hop_count == 4
    assert d.is_valid_inference is True
    assert abs(d.total_confidence - (0.9 ** 4)) < 1e-4


def test_cognitive_auditor_decisions():
    auditor = CognitiveUncertaintyAuditor(accept_threshold=0.55, margin_threshold=0.05)

    # 1. High score & clear margin -> ACCEPT
    d1 = auditor.audit([("B1", 0.90), ("B2", 0.40)])
    assert d1.decision_type == DecisionType.ACCEPT

    # 2. Low top score -> BELİRSİZ_VEYA_BİLİNMİYOR
    d2 = auditor.audit([("B1", 0.45), ("B2", 0.40)])
    assert d2.decision_type == DecisionType.BELIRSIZ_VEYA_BILINMIYOR

    # 3. High score & narrow margin -> ÇELİŞKİLİ_KANIT
    d3 = auditor.audit([("B1", 0.85), ("B2", 0.83)])
    assert d3.decision_type == DecisionType.CELISKILI_KANIT


def test_a_plus_address_read_isolation():
    b1 = KnowledgeBlock("B1", "Selamlaşma", BinaryVector.from_bool_list([True]*128), AddressBlock("B1"))
    b2 = KnowledgeBlock("B2", "Matematik", BinaryVector.from_bool_list([False]*128), AddressBlock("B2"))

    network = BlockNetwork([b1, b2])

    q_vec = BinaryVector.from_bool_list([True]*128)
    res = network.process_query(q_vec, method="xor", top_k=1)

    assert b1.address_block.read_count == 1
    assert b2.address_block.read_count == 0  # Unselected block A+ layer NEVER read
