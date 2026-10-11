"""
Unit tests for Meaning, Intent, and Negation Encoder Module.
"""

import pytest
from src.meaning_encoder import detect_turkish_intent_and_polarity, MeaningFeatureExtractor, MeaningBinaryEncoder
from src.bitwise import similarity_xor_popcount


def test_turkish_intent_and_polarity_detection():
    p1 = detect_turkish_intent_and_polarity("Gel")
    assert p1["is_negated"] is False
    assert p1["intent_type"] in ["COMMAND", "STATEMENT"]

    p2 = detect_turkish_intent_and_polarity("Gelme")
    assert p2["is_negated"] is True
    assert p2["intent_type"] == "PROHIBITION"

    p3 = detect_turkish_intent_and_polarity("5 ile 5 i çarpma")
    assert p3["is_negated"] is True

    p4 = detect_turkish_intent_and_polarity("Çay getirir misin?")
    assert p4["intent_type"] == "REQUEST"


def test_negation_disambiguation_separability():
    negation_pairs = [
        ("Gel", "Gelme"),
        ("Konuş", "Konuşma"),
        ("5 ile 5 i çarp", "5 ile 5 i çarpma"),
        ("Can motoruna bin", "Can motoruna binme"),
        ("Ağacı kes", "Ağacı kesme"),
        ("Işığı aç", "Işığı açma")
    ]

    all_texts = []
    for p_t, n_t in negation_pairs:
        all_texts.extend([p_t, n_t])

    mfe = MeaningFeatureExtractor(max_features=256)
    mfe.fit(all_texts)

    encoder = MeaningBinaryEncoder(input_dim=256, bit_length=128, seed=42)
    encoder.fit_meaning_pairs(mfe, negation_pairs, epochs=25, lr=0.08)

    sims = []
    for pos_t, neg_t in negation_pairs:
        x_pos = mfe.transform_dense(pos_t)
        x_neg = mfe.transform_dense(neg_t)
        v_pos = encoder.encode(x_pos)
        v_neg = encoder.encode(x_neg)
        sim = similarity_xor_popcount(v_pos, v_neg)
        sims.append(sim)

    avg_sim = sum(sims) / len(sims)
    # Average Hamming similarity between positive command and prohibition MUST drop below 0.50
    assert avg_sim < 0.50, f"Average negation pair similarity should be < 0.50, got {avg_sim:.4f}"
