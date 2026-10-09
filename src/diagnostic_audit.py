"""
Diagnostic Audit Script: Prototype Collision & Feature Separability Analysis.
"""

import math
import random
from src.bitwise import BinaryVector, similarity_xor_popcount
from src.dataset import CATEGORIES, get_full_10cat_dataset
from src.encoder import TrainableBinaryEncoder

def audit_prototype_collisions():
    print("=== 1. AUDITING CATEGORY TARGET PROTOTYPE SEPARATION ===")
    enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)

    prototypes = enc.category_prototypes
    num_cats = len(CATEGORIES)

    collision_matrix = []
    identical_pairs = []

    for i in range(num_cats):
        row = []
        for j in range(num_cats):
            sim = similarity_xor_popcount(prototypes[i], prototypes[j])
            row.append(sim)
            if i < j and sim == 1.0:
                identical_pairs.append((i, CATEGORIES[i], j, CATEGORIES[j]))
        collision_matrix.append(row)

    print(f"Total Categories: {num_cats}")
    print(f"Identical Prototype Collisions Found: {len(identical_pairs)}")
    for c1_idx, c1_name, c2_idx, c2_name in identical_pairs:
        print(f"  CRITICAL COLLISION: Category {c1_idx} ({c1_name}) and Category {c2_idx} ({c2_name}) have 100.0% IDENTICAL target vectors!")

    print("\nPairwise Prototype Similarity Matrix:")
    header = "          " + " ".join(f"{CATEGORIES[i][:4]:>6}" for i in range(num_cats))
    print(header)
    for i in range(num_cats):
        row_str = f"{CATEGORIES[i][:8]:<10}" + " ".join(f"{collision_matrix[i][j]:6.2f}" for j in range(num_cats))
        print(row_str)

if __name__ == "__main__":
    audit_prototype_collisions()
