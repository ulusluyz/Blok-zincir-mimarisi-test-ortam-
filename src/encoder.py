"""
Trainable Binary Encoders & Learnable Binary Representations.
Provides:
1. Static LSH Encoder (Random Hash projection baseline)
2. Trainable Supervised Perceptron STE (Straight-Through Estimator) Encoder
3. Trainable Metric Learning Binary Encoder
"""

import math
import random
from typing import List, Dict, Tuple
from src.features import FeatureExtractor
from src.bitwise import BinaryVector


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
        # Random projection matrix W of shape (bit_length, input_dim)
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
    Trainable Binary Encoder using Straight-Through Estimator (STE) Perceptron learning.
    Learns projection weights W (shape bit_length x input_dim) and category target binary prototypes
    such that queries from category C map to binary representations close to category C's target prototype in Hamming space.
    """

    def __init__(self, input_dim: int, bit_length: int = 128, num_categories: int = 10, seed: int = 42):
        self.input_dim = input_dim
        self.bit_length = bit_length
        self.num_categories = num_categories
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

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 30, lr: float = 0.05):
        """
        Train W using supervised contrastive / margin hinge loss + Straight-Through Estimator (STE).
        Given query x belonging to class c, we want:
        Forward: h = W * x, b = sign(h)
        Loss encourages b to match category_prototypes[c] and stay away from category_prototypes[other_c].
        Gradient w.r.t h using STE: dL/dh ≈ dL/db (tanh surrogate gradient).
        """
        num_samples = len(X)
        if num_samples == 0:
            return

        for epoch in range(epochs):
            # Shuffle indices
            indices = list(range(num_samples))
            random.shuffle(indices)

            for idx in indices:
                x = X[idx]
                target_cat = y[idx]
                target_proto = self.category_prototypes[target_cat].to_bool_list()

                # Compute continuous projection h = W * x
                h = [sum(w * x_i for w, x_i in zip(row, x)) for row in self.W]

                # Update weights for each bit
                for b_idx in range(self.bit_length):
                    desired_bit = 1.0 if target_proto[b_idx] else -1.0
                    act = h[b_idx]

                    # Margin loss: if desired_bit * act < 1.0, update
                    margin = desired_bit * act
                    if margin < 1.0:
                        grad = -desired_bit  # Loss gradient w.r.t act
                        # Update row b_idx of W
                        row = self.W[b_idx]
                        for f_idx in range(self.input_dim):
                            row[f_idx] -= lr * grad * x[f_idx]
