"""
Text Preprocessor and Feature Extraction for Turkish Text.
Extracts char n-grams (3, 4, 5) and word n-grams (1, 2) to build a vocabulary and sparse feature vectors.
Uses zero external transformer dependencies.
"""

import math
import re
from typing import List, Dict, Set, Tuple


def normalize_turkish(text: str) -> str:
    """Normalize Turkish characters to lowercase and strip punctuation."""
    # First map Turkish specific uppercase characters to lowercase
    turkish_upper_map = {
        'İ': 'i', 'I': 'ı', 'Ş': 'ş', 'Ğ': 'ğ', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç'
    }
    for k, v in turkish_upper_map.items():
        text = text.replace(k, v)

    text = text.lower()

    # Clean non-alphanumeric except spaces
    text = re.sub(r'[^a-z0-9çğıöşü\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_ngrams(text: str) -> List[str]:
    """Extract word n-grams (1-2) and character n-grams (3-5) from normalized text."""
    norm_text = normalize_turkish(text)
    words = norm_text.split()
    features = []

    # Word unigrams and bigrams
    for w in words:
        features.append(f"w1:{w}")
    for i in range(len(words) - 1):
        features.append(f"w2:{words[i]}_{words[i+1]}")

    # Char n-grams (3, 4, 5)
    padded_text = f" {norm_text} "
    for n in (3, 4, 5):
        for i in range(len(padded_text) - n + 1):
            gram = padded_text[i:i+n]
            features.append(f"c{n}:{gram}")

    return features


class FeatureExtractor:
    """
    TF-IDF / Frequency based feature extractor for Turkish text.
    Builds vocabulary from training corpus and maps text to fixed-size dense feature vectors.
    """

    def __init__(self, max_features: int = 1024):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        doc_freq: Dict[str, int] = {}
        num_docs = len(texts)

        for text in texts:
            unique_grams = set(extract_ngrams(text))
            for gram in unique_grams:
                doc_freq[gram] = doc_freq.get(gram, 0) + 1

        # Select top max_features by document frequency
        sorted_grams = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocab = {gram: idx for idx, (gram, _) in enumerate(sorted_grams)}

        # Compute IDF
        for gram, idx in self.vocab.items():
            df = doc_freq[gram]
            self.idf[idx] = math.log((num_docs + 1) / (df + 1)) + 1.0

        self.is_fitted = True
        return self

    def transform_dense(self, text: str) -> List[float]:
        """Transform text to a dense float vector of dimension `max_features`."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before calling transform.")

        vector = [0.0] * len(self.vocab)
        grams = extract_ngrams(text)
        if not grams:
            return vector

        # Term frequencies
        tf: Dict[int, int] = {}
        for gram in grams:
            if gram in self.vocab:
                idx = self.vocab[gram]
                tf[idx] = tf.get(idx, 0) + 1

        total_terms = len(grams)
        # TF-IDF calculation
        for idx, count in tf.items():
            tf_val = count / total_terms
            vector[idx] = tf_val * self.idf[idx]

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector
