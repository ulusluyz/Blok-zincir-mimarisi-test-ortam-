"""
Unit tests for features, bitwise math, encoders, and baselines.
"""

import pytest
from src.features import normalize_turkish, extract_ngrams, FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine
from src.encoder import RandomLSHEncoder, TrainableBinaryEncoder
from src.baselines import KeywordBaselineMatcher, TFIDFBaselineMatcher


def test_turkish_normalization():
    raw = "İSTANBUL, MUSA'NIN DÜNYASI! Çılgınlık 123..."
    norm = normalize_turkish(raw)
    assert "istanbul" in norm
    assert "çılgınlık" in norm
    assert "!" not in norm


def test_bitwise_operations():
    # 64-bit vector
    v1 = BinaryVector(0b1100, bit_length=64)
    v2 = BinaryVector(0b1010, bit_length=64)

    # Hamming distance for 0b1100 XOR 0b1010 = 0b0110 is 2
    # S_XOR = 1 - 2/64 = 62/64 = 0.96875
    s_xor = similarity_xor_popcount(v1, v2)
    assert abs(s_xor - (1.0 - 2.0 / 64.0)) < 1e-6

    # Sign Cosine: (64 - 2*2)/64 = 60/64 = 0.9375
    s_sign = similarity_sign_cosine(v1, v2)
    assert abs(s_sign - (60.0 / 64.0)) < 1e-6


def test_trainable_encoder():
    fe = FeatureExtractor(max_features=64)
    texts = ["merhaba selam", "dünya güneş gezegen", "123 matematik"]
    fe.fit(texts)

    X = [fe.transform_dense(t) for t in texts]
    y = [0, 1, 2]

    encoder = TrainableBinaryEncoder(input_dim=64, bit_length=128, num_categories=3)
    encoder.fit(X, y, epochs=5)

    v0 = encoder.encode(X[0])
    v1 = encoder.encode(X[1])
    assert v0.bit_length == 128
    assert v1.bit_length == 128


def test_baselines():
    kw_matcher = KeywordBaselineMatcher({
        "Selamlaşma": ["merhaba", "selam"],
        "Gezegenler": ["dünya", "mars", "günes"]
    })
    scores = kw_matcher.rank_blocks("merhaba arkadaşım")
    assert scores[0][0] == "Selamlaşma"
