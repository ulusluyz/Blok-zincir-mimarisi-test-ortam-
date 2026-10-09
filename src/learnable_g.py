"""
Learnable Group Representatives (G) & Joint A/G Co-Optimization Module.

Architectures Implemented:
1. StaticGRepresentative: Control baseline with fixed category prototypes.
2. LearnableSingleGRepresentative: Single trainable continuous/binarized prototype per group.
3. LearnableMultiPrototypeGRepresentative: Fixed M learnable sub-prototypes per group with max/soft score aggregation.
4. DynamicMultiPrototypeGRepresentative: Dynamic M sub-prototypes that split, merge, or prune based on intra-group variance and routing error.

Joint Optimization:
- JointAGTrainer: Co-optimizes encoder projection matrix W (A kernels) AND G prototype weights {P_k} using Multi-Label BCE/Margin Loss with STE gradients.
- Tracks parameter drift (||W_new - W_old||, ||P_new - P_old||) to prove true joint learning.
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
        # Continuous weight vector P of shape (bit_length,)
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
        """Update continuous P weights using STE Margin Hinge Loss."""
        target_val = 1.0 if is_target else -1.0
        for i in range(self.bit_length):
            q_val = 1.0 if q_bools[i] else -1.0
            act = self.P[i] * q_val
            if target_val * act < margin:
                # Gradient update
                grad = -target_val * q_val
                self.P[i] -= lr * grad


class LearnableMultiPrototypeGRepresentative(BaseGRepresentative):
    """Architecture B: Fixed M learnable sub-prototypes per group."""
    def __init__(self, group_id: str, name: str, num_sub_prototypes: int = 3, bit_length: int = 128, seed: int = 42):
        super().__init__(group_id, name, bit_length=bit_length)
        self.num_sub_prototypes = num_sub_prototypes
        rng = random.Random(seed)
        # Sub-prototype continuous weight matrices: List of M vectors of length bit_length
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
        """Updates winning sub-prototype."""
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

        # Start with M = 1 initial sub-prototype
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

    def adapt_topology(self, split_error_threshold: float = 4.0, merge_similarity_threshold: float = 0.95):
        """Dynamic splitting, merging, and pruning pass."""
        # 1. Split sub-prototype with high routing error if capacity allows
        if len(self.sub_prototypes) < self.max_sub_prototypes:
            for idx in range(len(self.sub_prototypes)):
                if self.sub_errors[idx] >= split_error_threshold:
                    # Split into 2 perturbed prototypes
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

        # 2. Merge overly similar sub-prototypes
        if len(self.sub_prototypes) > 1:
            for i in range(len(self.sub_prototypes)):
                for j in range(i + 1, len(self.sub_prototypes)):
                    v_i = BinaryVector.from_bool_list([w > 0.0 for w in self.sub_prototypes[i]])
                    v_j = BinaryVector.from_bool_list([w > 0.0 for w in self.sub_prototypes[j]])
                    sim = similarity_xor_popcount(v_i, v_j)
                    if sim >= merge_similarity_threshold:
                        # Merge j into i and remove j
                        merged = [0.5 * (a + b) for a, b in zip(self.sub_prototypes[i], self.sub_prototypes[j])]
                        self.sub_prototypes[i] = merged
                        self.sub_prototypes.pop(j)
                        self.sub_activation_counts.pop(j)
                        self.sub_errors.pop(j)
                        return


class JointAGTrainer:
    """
    Joint Co-Optimization Trainer for Encoder W (A Kernels) and G Representatives.
    """

    def __init__(self, encoder, g_representatives: Dict[str, BaseGRepresentative]):
        self.encoder = encoder
        self.g_representatives = g_representatives

    def fit_joint(
        self,
        X_train: List[List[float]],
        y_train_categories: List[List[str]],
        all_categories: List[str],
        epochs: int = 30,
        lr: float = 0.05,
        g_loss_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Jointly trains Encoder W and G representatives.
        Measures initial and final weight drift to prove co-optimization.
        """
        # Snapshot initial weights
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

            for idx in indices:
                x = X_train[idx]
                target_cats = y_train_categories[idx]

                # Forward 1: Encoder projection h = W * x -> Q_bits
                Q_vector = self.encoder.encode(x)
                q_bools = Q_vector.to_bool_list()

                # Train Encoder W on primary category (first target)
                primary_cat_name = target_cats[0]
                primary_cat_idx = all_categories.index(primary_cat_name) if primary_cat_name in all_categories else 0
                target_proto_bools = self.encoder.category_prototypes[primary_cat_idx].to_bool_list()

                # STE Gradient update on W
                h = [sum(w * x_i for w, x_i in zip(row, x)) for row in self.encoder.W]
                for b_i in range(self.encoder.bit_length):
                    desired = 1.0 if target_proto_bools[b_i] else -1.0
                    margin = desired * h[b_i]
                    if margin < 1.0:
                        grad = -desired
                        epoch_loss += abs(1.0 - margin)
                        for f_i in range(self.encoder.input_dim):
                            self.encoder.W[b_i][f_i] -= lr * grad * x[f_i]

                # Forward & Update 2: Train G Representatives
                for cat_name, g_rep in self.g_representatives.items():
                    is_target = (cat_name in target_cats)

                    if isinstance(g_rep, LearnableSingleGRepresentative):
                        g_rep.update_gradients(q_bools, is_target, lr=lr)
                    elif isinstance(g_rep, LearnableMultiPrototypeGRepresentative):
                        g_rep.update_gradients(q_bools, is_target, lr=lr)
                    elif isinstance(g_rep, DynamicMultiPrototypeGRepresentative):
                        g_rep.update_gradients_and_adapt(q_bools, is_target, lr=lr)

            # Topology adaptation pass for dynamic G models
            for g_rep in self.g_representatives.values():
                if isinstance(g_rep, DynamicMultiPrototypeGRepresentative):
                    g_rep.adapt_topology()

            loss_history.append(epoch_loss / max(1, num_samples))

        # Measure Weight Drift
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
            "loss_history": loss_history,
            "encoder_w_drift_norm": w_drift,
            "g_prototypes_drift_norm": g_drifts,
            "average_g_drift": float(np.mean(list(g_drifts.values()))) if g_drifts else 0.0
        }
