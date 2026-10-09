"""
Learnable Group Representatives (G) & Joint A/G Co-Optimization Module.

Architectures Implemented:
1. StaticGRepresentative: Control baseline with fixed category prototypes.
2. LearnableSingleGRepresentative: Single trainable continuous/binarized prototype per group.
3. LearnableMultiPrototypeGRepresentative: Fixed M learnable sub-prototypes per group with max/soft score aggregation.
4. DynamicMultiPrototypeGRepresentative: Dynamic M sub-prototypes that split, merge, or prune based on intra-group variance and routing error.

5 Training Strategies Implemented:
1. A-Only Training
2. G-Only Training
3. Simultaneous Joint A/G Training
4. Staged Training (A then G)
5. Contrastive Margin Joint Training (Unified Loss L_total = lambda_A * L_A + lambda_G * L_G + lambda_sep * L_sep)
"""

import math
import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from src.bitwise import BinaryVector, similarity_xor_popcount, similarity_binarized_dot, similarity_sign_cosine
from src.blocks import KnowledgeBlock, AddressBlock


class BaseGRepresentative:
    """Abstract Base Class for Group Representatives (G)."""
    def __init__(self, group_id: str, name: str, bit_length: int = 128):
        self.group_id = group_id
        self.name = name
        self.bit_length = bit_length
        self.member_a_blocks: List[KnowledgeBlock] = []
        self.subgroups: List['BaseGRepresentative'] = []
        self.evaluation_count: int = 0

    def add_member_a_block(self, block: KnowledgeBlock):
        self.member_a_blocks.append(block)

    def add_subgroup(self, subgroup: 'BaseGRepresentative'):
        self.subgroups.append(subgroup)

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Optional[Callable] = None) -> float:
        raise NotImplementedError

    def update_weights(self, gradient: float, lr: float):
        pass


class StaticGRepresentative(BaseGRepresentative):
    """Control Baseline: Fixed, non-trainable category prototype vector."""
    def __init__(self, group_id: str, name: str, representation: BinaryVector):
        super().__init__(group_id, name, bit_length=representation.bit_length)
        self.representation = representation

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Optional[Callable] = None) -> float:
        self.evaluation_count += 1
        if sim_func is None:
            sim_func = similarity_xor_popcount
        return sim_func(self.representation, Q)


class LearnableSingleGRepresentative(BaseGRepresentative):
    """Architecture A: Single learnable prototype per group."""
    def __init__(self, group_id: str, name: str, bit_length: int = 128, seed: int = 42):
        super().__init__(group_id, name, bit_length=bit_length)
        rng = random.Random(seed)
        self.P = [rng.gauss(0.0, 0.1) for _ in range(bit_length)]
        self.activation_count = 0

    def get_binary_representation(self) -> BinaryVector:
        bools = [p > 0.0 for p in self.P]
        return BinaryVector.from_bool_list(bools)

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Optional[Callable] = None) -> float:
        self.evaluation_count += 1
        proto_bin = BinaryVector.from_bool_list([p > 0.0 for p in self.P])
        if sim_func is None:
            sim_func = similarity_xor_popcount
        return sim_func(proto_bin, Q)

    def update_gradients(self, q_bools: List[bool], is_target: bool, lr: float = 0.05, margin: float = 0.1):
        target_val = 1.0 if is_target else -1.0
        for i in range(self.bit_length):
            q_val = 1.0 if q_bools[i] else -1.0
            act = self.P[i] * q_val
            if target_val * act < margin:
                grad = -target_val * q_val
                self.P[i] -= lr * grad


class LearnableMultiPrototypeGRepresentative(BaseGRepresentative):
    """Architecture B: Fixed M learnable sub-prototypes per group."""
    def __init__(self, group_id: str, name: str, num_sub_prototypes: int = 3, bit_length: int = 128, seed: int = 42):
        super().__init__(group_id, name, bit_length=bit_length)
        self.num_sub_prototypes = num_sub_prototypes
        rng = random.Random(seed)
        self.sub_prototypes = [
            [rng.gauss(0.0, 0.1) for _ in range(bit_length)]
            for _ in range(num_sub_prototypes)
        ]
        self.sub_activation_counts = [0] * num_sub_prototypes

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Optional[Callable] = None, aggregation: str = "max") -> float:
        self.evaluation_count += 1
        if sim_func is None:
            sim_func = similarity_xor_popcount

        scores = []
        for p_vec in self.sub_prototypes:
            proto_bin = BinaryVector.from_bool_list([p > 0.0 for p in p_vec])
            scores.append(sim_func(proto_bin, Q))

        if aggregation == "max":
            best_idx = int(np.argmax(scores))
            self.sub_activation_counts[best_idx] += 1
            return float(scores[best_idx])
        elif aggregation == "soft_mean":
            return float(np.mean(scores))
        else:
            return float(scores[0])

    def update_gradients(self, q_bools: List[bool], is_target: bool, lr: float = 0.05):
        q_bin = BinaryVector.from_bool_list(q_bools)
        scores = [
            similarity_xor_popcount(BinaryVector.from_bool_list([p > 0.0 for p in p_vec]), q_bin)
            for p_vec in self.sub_prototypes
        ]
        best_sub_idx = int(np.argmax(scores)) if is_target else int(np.argmin(scores))

        p_target = self.sub_prototypes[best_sub_idx]
        target_val = 1.0 if is_target else -1.0
        for i in range(self.bit_length):
            q_val = 1.0 if q_bools[i] else -1.0
            act = p_target[i] * q_val
            if target_val * act < 0.1:
                grad = -target_val * q_val
                p_target[i] -= lr * grad


class DynamicMultiPrototypeGRepresentative(BaseGRepresentative):
    """Architecture C: Dynamic M sub-prototypes that split, merge, or prune during training."""
    def __init__(self, group_id: str, name: str, bit_length: int = 128, max_sub_prototypes: int = 8, seed: int = 42):
        super().__init__(group_id, name, bit_length=bit_length)
        self.max_sub_prototypes = max_sub_prototypes
        self.rng = random.Random(seed)

        self.sub_prototypes = [[self.rng.gauss(0.0, 0.1) for _ in range(bit_length)]]
        self.sub_activation_counts = [0]
        self.sub_errors = [0.0]

    def evaluate_suitability(self, Q: BinaryVector, sim_func: Optional[Callable] = None) -> float:
        self.evaluation_count += 1
        if sim_func is None:
            sim_func = similarity_xor_popcount

        scores = []
        for p_vec in self.sub_prototypes:
            proto_bin = BinaryVector.from_bool_list([p > 0.0 for p in p_vec])
            scores.append(sim_func(proto_bin, Q))

        best_idx = int(np.argmax(scores))
        self.sub_activation_counts[best_idx] += 1
        return float(scores[best_idx])

    def update_gradients_and_adapt(self, q_bools: List[bool], is_target: bool, lr: float = 0.05):
        q_bin = BinaryVector.from_bool_list(q_bools)
        scores = [
            similarity_xor_popcount(BinaryVector.from_bool_list([p > 0.0 for p in p_vec]), q_bin)
            for p_vec in self.sub_prototypes
        ]
        best_sub_idx = int(np.argmax(scores))

        p_target = self.sub_prototypes[best_sub_idx]
        target_val = 1.0 if is_target else -1.0

        if is_target and scores[best_sub_idx] < 0.6:
            self.sub_errors[best_sub_idx] += 1.0

        for i in range(self.bit_length):
            q_val = 1.0 if q_bools[i] else -1.0
            act = p_target[i] * q_val
            if target_val * act < 0.1:
                grad = -target_val * q_val
                p_target[i] -= lr * grad

    def adapt_topology(self, split_error_threshold: float = 3.0, merge_similarity_threshold: float = 0.95):
        """Dynamic splitting, merging, and pruning pass."""
        if len(self.sub_prototypes) < self.max_sub_prototypes:
            for idx in range(len(self.sub_prototypes)):
                if self.sub_errors[idx] >= split_error_threshold:
                    parent = self.sub_prototypes[idx]
                    child1 = [w + self.rng.gauss(0.0, 0.05) for w in parent]
                    child2 = [w - self.rng.gauss(0.0, 0.05) for w in parent]
                    self.sub_prototypes[idx] = child1
                    self.sub_prototypes.append(child2)
                    self.sub_activation_counts[idx] = 0
                    self.sub_activation_counts.append(0)
                    self.sub_errors[idx] = 0.0
                    self.sub_errors.append(0.0)
                    break

        if len(self.sub_prototypes) > 1:
            for i in range(len(self.sub_prototypes)):
                for j in range(i + 1, len(self.sub_prototypes)):
                    v_i = BinaryVector.from_bool_list([w > 0.0 for w in self.sub_prototypes[i]])
                    v_j = BinaryVector.from_bool_list([w > 0.0 for w in self.sub_prototypes[j]])
                    sim = similarity_xor_popcount(v_i, v_j)
                    if sim >= merge_similarity_threshold:
                        merged = [0.5 * (a + b) for a, b in zip(self.sub_prototypes[i], self.sub_prototypes[j])]
                        self.sub_prototypes[i] = merged
                        self.sub_prototypes.pop(j)
                        self.sub_activation_counts.pop(j)
                        self.sub_errors.pop(j)
                        return


class JointAGTrainer:
    """
    Joint Co-Optimization Trainer supporting 5 Training Strategies.
    """

    def __init__(self, encoder, g_representatives: Dict[str, BaseGRepresentative]):
        self.encoder = encoder
        self.g_representatives = g_representatives

    def enforce_inter_class_separation(self, separation_margin: float = 0.60, lr_sep: float = 0.02):
        g_list = list(self.g_representatives.values())
        num_g = len(g_list)

        for i in range(num_g):
            for j in range(i + 1, num_g):
                g_i, g_j = g_list[i], g_list[j]

                p_i_vecs = []
                if isinstance(g_i, LearnableSingleGRepresentative):
                    p_i_vecs = [g_i.P]
                elif isinstance(g_i, (LearnableMultiPrototypeGRepresentative, DynamicMultiPrototypeGRepresentative)):
                    p_i_vecs = g_i.sub_prototypes

                p_j_vecs = []
                if isinstance(g_j, LearnableSingleGRepresentative):
                    p_j_vecs = [g_j.P]
                elif isinstance(g_j, (LearnableMultiPrototypeGRepresentative, DynamicMultiPrototypeGRepresentative)):
                    p_j_vecs = g_j.sub_prototypes

                for p_i in p_i_vecs:
                    for p_j in p_j_vecs:
                        bin_i = BinaryVector.from_bool_list([w > 0.0 for w in p_i])
                        bin_j = BinaryVector.from_bool_list([w > 0.0 for w in p_j])
                        sim = similarity_xor_popcount(bin_i, bin_j)

                        if sim > separation_margin:
                            for b_k in range(min(len(p_i), len(p_j))):
                                dir_j = 1.0 if p_j[b_k] > 0.0 else -1.0
                                dir_i = 1.0 if p_i[b_k] > 0.0 else -1.0
                                if dir_i == dir_j:
                                    p_i[b_k] -= lr_sep * dir_j
                                    p_j[b_k] -= lr_sep * dir_i

    def fit_strategy(
        self,
        X_train: List[List[float]],
        y_train_categories: List[List[str]],
        all_categories: List[str],
        strategy: str = "contrastive_joint",  # "a_only", "g_only", "simultaneous_joint", "staged", "contrastive_joint"
        epochs: int = 25,
        lr: float = 0.05
    ) -> Dict[str, Any]:
        W_init = np.array(self.encoder.W, copy=True)
        G_inits = {}
        for g_id, g_rep in self.g_representatives.items():
            if isinstance(g_rep, LearnableSingleGRepresentative):
                G_inits[g_id] = np.array(g_rep.P, copy=True)
            elif isinstance(g_rep, (LearnableMultiPrototypeGRepresentative, DynamicMultiPrototypeGRepresentative)):
                G_inits[g_id] = [np.array(p, copy=True) for p in g_rep.sub_prototypes]

        num_samples = len(X_train)
        loss_history = []

        for epoch in range(epochs):
            indices = list(range(num_samples))
            random.shuffle(indices)
            epoch_loss = 0.0

            # Determine whether to train A, G, or both based on strategy
            train_a = True
            train_g = True

            if strategy == "a_only":
                train_g = False
            elif strategy == "g_only":
                train_a = False
            elif strategy == "staged":
                # Staged: train A for first half, G for second half
                if epoch < epochs // 2:
                    train_g = False
                else:
                    train_a = False

            for idx in indices:
                x = X_train[idx]
                target_cats = y_train_categories[idx]

                # Train Encoder W (A)
                if train_a:
                    primary_cat_name = target_cats[0]
                    primary_cat_idx = all_categories.index(primary_cat_name) if primary_cat_name in all_categories else 0
                    pos_proto = self.encoder.category_prototypes[primary_cat_idx].to_bool_list()

                    h = [sum(w * x_i for w, x_i in zip(row, x)) for row in self.encoder.W]
                    q_vec = BinaryVector.from_bool_list([act > 0.0 for act in h])

                    if strategy == "contrastive_joint":
                        # Contrastive Hard Negative Mining
                        hard_neg_idx = None
                        max_neg_sim = -1.0
                        for cat_i, proto in self.encoder.category_prototypes.items():
                            if cat_i != primary_cat_idx:
                                sim = similarity_xor_popcount(q_vec, proto)
                                if sim > max_neg_sim:
                                    max_neg_sim = sim
                                    hard_neg_idx = cat_i

                        neg_proto = self.encoder.category_prototypes[hard_neg_idx].to_bool_list() if hard_neg_idx is not None else pos_proto

                        for b_i in range(self.encoder.bit_length):
                            pos_desired = 1.0 if pos_proto[b_i] else -1.0
                            neg_desired = 1.0 if neg_proto[b_i] else -1.0
                            act = h[b_i]

                            pos_grad = -pos_desired if (pos_desired * act < 1.0) else 0.0
                            neg_grad = neg_desired if (neg_desired * act > -0.2) else 0.0

                            total_grad = pos_grad + 0.6 * neg_grad
                            if abs(total_grad) > 1e-6:
                                epoch_loss += abs(total_grad)
                                for f_i in range(self.encoder.input_dim):
                                    self.encoder.W[b_i][f_i] -= lr * total_grad * x[f_i]
                    else:
                        # Standard Perceptron STE Loss
                        for b_i in range(self.encoder.bit_length):
                            desired = 1.0 if pos_proto[b_i] else -1.0
                            margin = desired * h[b_i]
                            if margin < 1.0:
                                grad = -desired
                                epoch_loss += abs(1.0 - margin)
                                for f_i in range(self.encoder.input_dim):
                                    self.encoder.W[b_i][f_i] -= lr * grad * x[f_i]

                # Train G Representatives
                if train_g:
                    Q_curr = self.encoder.encode(x)
                    q_bools = Q_curr.to_bool_list()

                    for cat_name, g_rep in self.g_representatives.items():
                        is_target = (cat_name in target_cats)

                        if isinstance(g_rep, LearnableSingleGRepresentative):
                            g_rep.update_gradients(q_bools, is_target, lr=lr)
                        elif isinstance(g_rep, LearnableMultiPrototypeGRepresentative):
                            g_rep.update_gradients(q_bools, is_target, lr=lr)
                        elif isinstance(g_rep, DynamicMultiPrototypeGRepresentative):
                            g_rep.update_gradients_and_adapt(q_bools, is_target, lr=lr)

            # Apply Inter-Class Prototype Separation Regularization in contrastive mode
            if train_g and strategy == "contrastive_joint":
                self.enforce_inter_class_separation(separation_margin=0.60, lr_sep=0.02)

            for g_rep in self.g_representatives.values():
                if isinstance(g_rep, DynamicMultiPrototypeGRepresentative):
                    g_rep.adapt_topology()

            loss_history.append(epoch_loss / max(1, num_samples))

        # Weight Drift
        W_final = np.array(self.encoder.W, copy=True)
        w_drift = float(np.linalg.norm(W_final - W_init))

        g_drifts = {}
        for g_id, g_rep in self.g_representatives.items():
            if isinstance(g_rep, LearnableSingleGRepresentative):
                P_final = np.array(g_rep.P, copy=True)
                g_drifts[g_id] = float(np.linalg.norm(P_final - G_inits[g_id]))
            elif isinstance(g_rep, (LearnableMultiPrototypeGRepresentative, DynamicMultiPrototypeGRepresentative)):
                sub_drifts = []
                for sub_i, p_final in enumerate(g_rep.sub_prototypes):
                    if sub_i < len(G_inits[g_id]):
                        sub_drifts.append(float(np.linalg.norm(np.array(p_final) - G_inits[g_id][sub_i])))
                g_drifts[g_id] = float(np.mean(sub_drifts)) if sub_drifts else 0.0

        return {
            "strategy": strategy,
            "loss_history": loss_history,
            "encoder_w_drift_norm": w_drift,
            "g_prototypes_drift_norm": g_drifts,
            "average_g_drift": float(np.mean(list(g_drifts.values()))) if g_drifts else 0.0
        }

    def fit_joint(self, X_train, y_train_categories, all_categories, epochs=30, lr=0.05, contrastive_weight=0.6):
        return self.fit_strategy(
            X_train, y_train_categories, all_categories, strategy="contrastive_joint", epochs=epochs, lr=lr
        )
