"""Graph Neural Network (GNN) for Player Relationships & Playing Style.

Implements:
1. GCN (Graph Convolutional Network) - normalized Laplacian message passing
2. GraphSAGE - neighborhood mean/max aggregation with residual skip-connections
3. GAT (Graph Attention Network) - self-attention edge weighting with edge features
4. Global Attention Pooling to derive team embeddings
5. Match Graph Representation & 3-way Outcome Classifier (Home/Draw/Away)
"""

from __future__ import annotations

import math
from typing import Literal
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)


class GCNLayer:
    """Graph Convolutional Network Layer (Kipf & Welling)."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass for a team graph (N nodes, in_features)."""
        # A_tilde = A + I
        N = X.shape[0]
        A_tilde = A + np.eye(N)
        # D_tilde^(-1/2)
        deg = np.sum(A_tilde, axis=1)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        # Message passing: A_norm @ X @ W + b
        Z = A_norm @ X @ self.W + self.b
        H = relu(Z)
        return H, A_norm


class GraphSAGELayer:
    """GraphSAGE Layer (Hamilton et al.) with mean aggregation."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (2 * in_features + out_features))
        self.W_self = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.W_neigh = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Neighborhood mean aggregation
        N = X.shape[0]
        deg = np.maximum(np.sum(A, axis=1, keepdims=True), 1.0)
        A_norm = A / deg
        neigh_agg = A_norm @ X

        Z = X @ self.W_self + neigh_agg @ self.W_neigh + self.b
        H = relu(Z)
        return H, A_norm


class GATLayer:
    """Graph Attention Network Layer (Veličković et al.) with Edge-Biased Attention."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_heads = n_heads
        self.out_per_head = out_features // n_heads
        limit = math.sqrt(6.0 / (in_features + self.out_per_head))
        self.W = [rng.uniform(-limit, limit, size=(in_features, self.out_per_head)) for _ in range(n_heads)]
        self.a_src = [rng.uniform(-limit, limit, size=(self.out_per_head, 1)) for _ in range(n_heads)]
        self.a_dst = [rng.uniform(-limit, limit, size=(self.out_per_head, 1)) for _ in range(n_heads)]
        self.b = np.zeros(out_features)

    def forward(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = X.shape[0]
        head_outputs = []
        att_matrices = []

        for h in range(self.n_heads):
            # Projected features
            H_proj = X @ self.W[h]  # (N, out_per_head)
            src_score = H_proj @ self.a_src[h]  # (N, 1)
            dst_score = H_proj @ self.a_dst[h]  # (N, 1)

            # Raw attention logits: score_i + score_j + edge_weight
            e_ij = src_score + dst_score.T + A * 0.5
            e_ij = leaky_relu(e_ij)

            # Mask non-edges (where A == 0 and not diagonal)
            mask = (A > 0) | np.eye(N, dtype=bool)
            e_ij_masked = np.where(mask, e_ij, -1e9)
            alpha = softmax(e_ij_masked, axis=1)

            H_head = alpha @ H_proj
            head_outputs.append(H_head)
            att_matrices.append(alpha)

        H_concat = np.concatenate(head_outputs, axis=1) + self.b
        H_out = relu(H_concat)
        mean_att = np.mean(att_matrices, axis=0)
        return H_out, mean_att


class TeamGNN:
    """Full Team Graph Neural Network with Attention Pooling."""

    def __init__(
        self,
        in_features: int = 26,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        architecture: Literal["gcn", "graphsage", "gat"] = "gat",
        seed: int = 42,
    ):
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed)

        if architecture == "gcn":
            self.l1 = GCNLayer(in_features, hidden_dim, seed=seed)
            self.l2 = GCNLayer(hidden_dim, embed_dim, seed=seed + 1)
        elif architecture == "graphsage":
            self.l1 = GraphSAGELayer(in_features, hidden_dim, seed=seed)
            self.l2 = GraphSAGELayer(hidden_dim, embed_dim, seed=seed + 1)
        elif architecture == "gat":
            self.l1 = GATLayer(in_features, hidden_dim, n_heads=2, seed=seed)
            self.l2 = GATLayer(hidden_dim, embed_dim, n_heads=2, seed=seed + 1)

        # Global Attention Pooling weights
        self.w_pool = rng.uniform(-0.1, 0.1, size=(embed_dim, 1))

    def embed_team(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Produce team-level embedding z and learned relationship/attention matrix."""
        H1, _ = self.l1.forward(X, A)
        H2, A_learned = self.l2.forward(H1, A)

        # Global attention pooling over 11 players
        attn_weights = softmax(H2 @ self.w_pool, axis=0)  # (11, 1)
        z = np.sum(attn_weights * H2, axis=0)  # (embed_dim,)
        return z, A_learned


class MatchGNNClassifier:
    """Predicts Home/Draw/Away match outcome using Team GNN embeddings + Match Context."""

    def __init__(
        self,
        gnn_arch: Literal["gcn", "graphsage", "gat"] = "gat",
        in_features: int = 26,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        context_dim: int = 6,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.gnn = TeamGNN(in_features, hidden_dim, embed_dim, architecture=gnn_arch, seed=seed)
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.lr = lr
        self.seed = seed

        # Match representation: [z_A, z_B, z_A - z_B, |z_A - z_B|, context]
        match_dim = 4 * embed_dim + context_dim
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (match_dim + 3))
        self.W_out = rng.uniform(-limit, limit, size=(match_dim, 3))
        self.b_out = np.zeros(3)

    def forward_match(
        self,
        X_h: np.ndarray,
        A_h: np.ndarray,
        X_a: np.ndarray,
        A_a: np.ndarray,
        context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3-way match probabilities and representations."""
        z_h, A_learned_h = self.gnn.embed_team(X_h, A_h)
        z_a, A_learned_a = self.gnn.embed_team(X_a, A_a)

        diff = z_h - z_a
        abs_diff = np.abs(diff)
        rep = np.concatenate([z_h, z_a, diff, abs_diff, context])

        logits = rep @ self.W_out + self.b_out
        probs = softmax(logits)
        return probs, rep, A_learned_h, A_learned_a

    def predict_proba_dataset(
        self,
        dataset: list[dict],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Predict probabilities for an entire match dataset."""
        probs_all = []
        rep_all = []
        for sample in dataset:
            p, r, _, _ = self.forward_match(
                sample["X_home"],
                sample["A_home"],
                sample["X_away"],
                sample["A_away"],
                sample["context"],
            )
            probs_all.append(p)
            rep_all.append(r)
        return np.vstack(probs_all), rep_all
