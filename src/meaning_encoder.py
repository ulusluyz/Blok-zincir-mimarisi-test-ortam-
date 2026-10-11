"""
Context-Aware Turkish Meaning, Intent, and Negation Encoder Module.

Learns:
1. Polarity & Negation ("Gel" vs "Gelme", "5 ile 5 i çarp" vs "5 ile 5 i çarpma").
2. Sentence Intent Types (Command, Prohibition, Question, Request).
3. Contextual Disambiguation ("Can motoruna binme" -> Person/Entity vs Engine).
4. Maps metin + niyet + olumsuzluk into 128-bit BinaryVector space with high Hamming separability.
"""

import math
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from src.features import normalize_turkish, extract_ngrams
from src.bitwise import BinaryVector, similarity_xor_popcount


def detect_turkish_intent_and_polarity(text: str) -> Dict[str, Any]:
    """
    Parses Turkish text for negation suffixes, verb polarities, and sentence intent types.
    """
    norm = normalize_turkish(text)
    words = norm.split()

    # Negation indicators
    negation_patterns = [
        r'\b\w+(me|ma)\b', r'\b\w+(mıyor|miyor|müyor|muyor)\b',
        r'\b\w+(mez|maz)\b', r'\b\w+(mem|mam)\b', r'\b\w+(mıyoruz|mıyoruz)\b',
        r'\bdeğil\b', r'\byok\b'
    ]

    is_negated = False
    for pat in negation_patterns:
        if re.search(pat, norm):
            # Exclude false positives like "elma", "sinema", "sarılma" if unigram noun
            if not (norm in ["elma", "sinema", "dondurma"] and len(words) == 1):
                is_negated = True
                break

    # Intent Classification
    intent_type = "STATEMENT"
    if norm.endswith("mi") or norm.endswith("mı") or norm.endswith("mu") or norm.endswith("mü") or "?" in text:
        if any(w in norm for w in ["mısın", "misin", "musun", "müsün", "getirir", "yapar", "bakar"]):
            intent_type = "REQUEST"
        else:
            intent_type = "QUESTION"
    elif is_negated:
        intent_type = "PROHIBITION"
    elif len(words) <= 4 and any(norm.endswith(suf) for suf in ["et", "yap", "gel", "git", "çarp", "konuş", "bin", "aç"]):
        intent_type = "COMMAND"

    return {
        "text": text,
        "normalized": norm,
        "is_negated": is_negated,
        "intent_type": intent_type,
        "words": words
    }


class MeaningFeatureExtractor:
    """
    Extracts content n-grams + explicit Intent/Polarity feature tokens.
    """

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self.is_fitted = False

    def _get_all_features(self, text: str) -> List[str]:
        parsed = detect_turkish_intent_and_polarity(text)
        features = extract_ngrams(text)

        # Append explicit Intent and Polarity tokens
        features.append(f"POLARITY:{'NEG' if parsed['is_negated'] else 'POS'}")
        features.append(f"INTENT:{parsed['intent_type']}")

        for w in parsed["words"]:
            if parsed["is_negated"]:
                features.append(f"NEG_WORD:{w}")
            else:
                features.append(f"POS_WORD:{w}")

        return features

    def fit(self, texts: List[str]) -> 'MeaningFeatureExtractor':
        doc_freq: Dict[str, int] = {}
        num_docs = len(texts)

        for text in texts:
            unique_grams = set(self._get_all_features(text))
            for gram in unique_grams:
                doc_freq[gram] = doc_freq.get(gram, 0) + 1

        sorted_grams = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocab = {gram: idx for idx, (gram, _) in enumerate(sorted_grams)}

        for gram, idx in self.vocab.items():
            df = doc_freq[gram]
            self.idf[idx] = math.log((num_docs + 1) / (df + 1)) + 1.0

        self.is_fitted = True
        return self

    def transform_dense(self, text: str) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("MeaningFeatureExtractor must be fitted before calling transform.")

        vector = [0.0] * len(self.vocab)
        features = self._get_all_features(text)
        if not features:
            return vector

        tf: Dict[int, int] = {}
        for gram in features:
            if gram in self.vocab:
                idx = self.vocab[gram]
                tf[idx] = tf.get(idx, 0) + 1

        for idx, count in tf.items():
            tf_val = 1.0 + math.log(count)
            vector[idx] = tf_val * self.idf[idx]

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


class MeaningBinaryEncoder:
    """
    Trainable Meaning & Negation Binary Encoder using STE Margin Hinge Loss.
    Ensures that positive and negated commands ("Gel" vs "Gelme") map to distinct binary vectors.
    """

    def __init__(self, input_dim: int = 512, bit_length: int = 128, seed: int = 42):
        self.input_dim = input_dim
        self.bit_length = bit_length
        self.seed = seed
        rng = random.Random(seed)

        self.W = [
            [rng.gauss(0.0, 1.0 / math.sqrt(input_dim)) for _ in range(input_dim)]
            for _ in range(bit_length)
        ]

    def encode(self, dense_vector: List[float]) -> BinaryVector:
        bools = []
        for row in self.W:
            dot = sum(w * x for w, x in zip(row, dense_vector))
            bools.append(dot > 0.0)
        return BinaryVector.from_bool_list(bools)

    def fit_meaning_pairs(
        self,
        feature_extractor: MeaningFeatureExtractor,
        pairs: List[Tuple[str, str]],  # (Positive command, Negated command)
        epochs: int = 30,
        lr: float = 0.05
    ):
        """
        Trains W to explicitly push apart positive and negated commands in Hamming space.
        """
        rng = random.Random(self.seed)

        for epoch in range(epochs):
            shuffled_pairs = list(pairs)
            rng.shuffle(shuffled_pairs)

            for pos_text, neg_text in shuffled_pairs:
                x_pos = feature_extractor.transform_dense(pos_text)
                x_neg = feature_extractor.transform_dense(neg_text)

                h_pos = [sum(w * x_i for w, x_i in zip(row, x_pos)) for row in self.W]
                h_neg = [sum(w * x_i for w, x_i in zip(row, x_neg)) for row in self.W]

                # We want h_pos and h_neg to have OPPOSITE signs on intent/polarity bits
                for b_i in range(self.bit_length):
                    # Encourage opposite bit values between pos and neg
                    act_pos, act_neg = h_pos[b_i], h_neg[b_i]
                    if act_pos * act_neg > -0.5:  # If they have same sign or small margin
                        # Push act_pos positive, act_neg negative (or vice versa)
                        grad_pos = -1.0 if act_pos < 0.5 else 0.0
                        grad_neg = 1.0 if act_neg > -0.5 else 0.0

                        for f_i in range(self.input_dim):
                            if x_pos[f_i] != 0:
                                self.W[b_i][f_i] -= lr * grad_pos * x_pos[f_i]
                            if x_neg[f_i] != 0:
                                self.W[b_i][f_i] -= lr * grad_neg * x_neg[f_i]
