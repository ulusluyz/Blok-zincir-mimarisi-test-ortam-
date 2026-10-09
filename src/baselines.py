"""
Baseline Models for Comparison:
1. Exact/Partial Keyword Matcher
2. Standard TF-IDF + Cosine Similarity Matcher
"""

from typing import List, Dict, Tuple
from src.features import FeatureExtractor, normalize_turkish
import math


class KeywordBaselineMatcher:
    """
    Keyword Overlap Baseline Matcher.
    Calculates keyword overlap ratio between query and block keyphrases.
    """

    def __init__(self, block_keywords: Dict[str, List[str]]):
        # block_keywords: category_name -> list of keywords/phrases
        self.block_keywords = {
            cat: [normalize_turkish(kw) for kw in kws]
            for cat, kws in block_keywords.items()
        }

    def predict_score(self, query: str, category: str) -> float:
        norm_query = set(normalize_turkish(query).split())
        if not norm_query or category not in self.block_keywords:
            return 0.0

        cat_words = set()
        for kw in self.block_keywords[category]:
            cat_words.update(kw.split())

        if not cat_words:
            return 0.0

        overlap = len(norm_query.intersection(cat_words))
        return overlap / len(norm_query)

    def rank_blocks(self, query: str) -> List[Tuple[str, float]]:
        scores = [(cat, self.predict_score(query, cat)) for cat in self.block_keywords]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class TFIDFBaselineMatcher:
    """
    TF-IDF + Cosine Similarity Baseline Matcher.
    Computes cosine similarity between query TF-IDF vector and block prototype TF-IDF vectors.
    """

    def __init__(self, feature_extractor: FeatureExtractor, block_texts: Dict[str, List[str]]):
        self.fe = feature_extractor
        self.block_centroids: Dict[str, List[float]] = {}

        for cat, texts in block_texts.items():
            if not texts:
                continue
            vectors = [self.fe.transform_dense(t) for t in texts]
            # Compute centroid vector
            dim = len(vectors[0])
            centroid = [0.0] * dim
            for vec in vectors:
                for i in range(dim):
                    centroid[i] += vec[i]
            centroid = [c / len(vectors) for c in centroid]
            # Normalize centroid
            norm = math.sqrt(sum(c * c for c in centroid))
            if norm > 0:
                centroid = [c / norm for c in centroid]
            self.block_centroids[cat] = centroid

    def predict_score(self, query: str, category: str) -> float:
        if category not in self.block_centroids:
            return 0.0
        q_vec = self.fe.transform_dense(query)
        c_vec = self.block_centroids[category]
        # Cosine dot product (since both are L2 normalized)
        return sum(q * c for q, c in zip(q_vec, c_vec))

    def rank_blocks(self, query: str) -> List[Tuple[str, float]]:
        scores = [(cat, self.predict_score(query, cat)) for cat in self.block_centroids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
