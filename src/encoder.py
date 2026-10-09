"""
Trainable Binary Encoders & Learnable Binary Representations.
Provides:
1. Static LSH Encoder (Random Hash projection baseline)
2. Trainable Supervised Perceptron STE (Straight-Through Estimator) Encoder
3. Supervised Contrastive STE Encoder with Hard Negative Mining (Inter-class separability)
"""

import math
import random
import numpy as np
from typing import List, Dict, Tuple
from src.features import FeatureExtractor
from src.bitwise import BinaryVector, similarity_xor_popcount


class RandomLSHEncoder:
    """
    Baseline: Random Locality-Sensitive Hashing (LSH) projection.
    Maps input vector x -> sign(W * x) using static random weights.
    No training on suitability labels.
    """

    def __init__(self, input_dim: int, bit_length: int = 128, seed: int = 42):
        self.input_dim = input_dim
        self.bit_length = bit_length
        rng = random.Random(seed)
        self.W = [
            [rng.gauss(0.0, 1.0) for _ in range(input_dim)]
            for _ in range(bit_length)
        ]

    def encode(self, dense_vector: List[float]) -> BinaryVector:
        bools = []
        for row in self.W:
            dot = sum(w * x for w, x in zip(row, dense_vector))
            bools.append(dot > 0.0)
        return BinaryVector.from_bool_list(bools)


class TrainableBinaryEncoder:
    """
    Trainable Binary Encoder using Straight-Through Estimator (STE) Perceptron and Contrastive Hard Negative Mining.
    Learns projection weights W (shape bit_length x input_dim) and category target binary prototypes.
    """

    def __init__(self, input_dim: int, bit_length: int = 128, num_categories: int = 10, seed: int = 42):
        self.input_dim = input_dim
        self.bit_length = bit_length
        self.num_categories = num_categories
        self.seed = seed
        rng = random.Random(seed)

        # Learnable Projection Matrix W
        self.W = [
            [rng.gauss(0.0, 1.0 / math.sqrt(input_dim)) for _ in range(input_dim)]
            for _ in range(bit_length)
        ]

        # Initialize distinct, orthogonally dispersed target prototypes for each category class in N-bit space
        self.category_prototypes: Dict[int, BinaryVector] = {}
        for cat_idx in range(num_categories):
            cat_rng = random.Random(seed + cat_idx * 1000 + 7)
            proto_bits = [cat_rng.choice([True, False]) for _ in range(bit_length)]
            self.category_prototypes[cat_idx] = BinaryVector.from_bool_list(proto_bits)

    def encode(self, dense_vector: List[float]) -> BinaryVector:
        bools = []
        for row in self.W:
            dot = sum(w * x for w, x in zip(row, dense_vector))
            bools.append(dot > 0.0)
        return BinaryVector.from_bool_list(bools)

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 30, lr: float = 0.05, contrastive_weight: float = 0.5):
        """
        Train W using Supervised Contrastive Margin Hinge Loss + Hard Negative Mining + STE gradients.
        Pushes representation towards positive target prototype while actively pushing away from hardest negative prototype.
        """
        num_samples = len(X)
        if num_samples == 0:
            return

        rng = random.Random(self.seed)

        for epoch in range(epochs):
            indices = list(range(num_samples))
            rng.shuffle(indices)

            for idx in indices:
                x = X[idx]
                target_cat = y[idx]
                pos_proto = self.category_prototypes[target_cat].to_bool_list()

                # Compute continuous projection h = W * x
                h = [sum(w * x_i for w, x_i in zip(row, x)) for row in self.W]
                q_vec = BinaryVector.from_bool_list([act > 0.0 for act in h])

                # Hard Negative Mining: Find non-target prototype with highest similarity to q_vec
                hard_neg_cat = None
                max_neg_sim = -1.0
                for cat_i, proto in self.category_prototypes.items():
                    if cat_i != target_cat:
                        sim = similarity_xor_popcount(q_vec, proto)
                        if sim > max_neg_sim:
                            max_neg_sim = sim
                            hard_neg_cat = cat_i

                neg_proto = self.category_prototypes[hard_neg_cat].to_bool_list() if hard_neg_cat is not None else pos_proto

                # Update weights for each bit
                for b_idx in range(self.bit_length):
                    pos_desired = 1.0 if pos_proto[b_idx] else -1.0
                    neg_desired = 1.0 if neg_proto[b_idx] else -1.0
                    act = h[b_idx]

                    # 1. Positive Margin Loss: Encourage act to match pos_desired
                    pos_margin = pos_desired * act
                    pos_grad = 0.0
                    if pos_margin < 1.0:
                        pos_grad = -pos_desired

                    # 2. Hard Negative Margin Loss: Encourage act to DIFFER from neg_desired
                    neg_margin = neg_desired * act
                    neg_grad = 0.0
                    if neg_margin > -0.2:  # If too close to negative prototype bit
                        neg_grad = neg_desired

                    total_grad = pos_grad + contrastive_weight * neg_grad

                    if abs(total_grad) > 1e-6:
                        row = self.W[b_idx]
                        for f_idx in range(self.input_dim):
                            row[f_idx] -= lr * total_grad * x[f_idx]
