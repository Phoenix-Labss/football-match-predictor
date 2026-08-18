"""Neural Sequence Encoders for Temporal Team State Representation.

Implements PyTorch-based sequence models (GRU, LSTM, Temporal Transformer)
that encode chronological match histories into compact learned team representations
(unstructured and structured) with zero lookahead leakage.
"""

from __future__ import annotations

import math
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Temporal Transformer."""

    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SequenceEncoderCore(nn.Module):
    """Core neural encoder supporting GRU, LSTM, and Temporal Transformer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        arch: Literal["gru", "lstm", "transformer"] = "gru",
        nhead: int = 4,
    ):
        super().__init__()
        self.arch = arch
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if arch == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif arch == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif arch == "transformer":
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=50)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Attention pooling over timesteps
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode sequence x (batch_size, seq_len, input_dim) -> (batch_size, hidden_dim)."""
        # Guard against rows with all elements padded (mask == 0 for all timesteps)
        if mask is not None:
            all_pad = (mask.sum(dim=1) == 0)
            if all_pad.any():
                mask = mask.clone()
                mask[all_pad, -1] = 1.0

        # Project input
        h = self.input_proj(x)

        if self.arch == "gru":
            out, _ = self.rnn(h)
        elif self.arch == "lstm":
            out, _ = self.rnn(h)
        elif self.arch == "transformer":
            h = self.pos_encoder(h)
            key_padding_mask = (mask == 0) if mask is not None else None
            out = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Weighted attention pooling over sequence length
        if mask is not None:
            weights = self.attn_pool(out)  # (batch_size, seq_len, 1)
            weights = weights * mask.unsqueeze(-1)
            denom = weights.sum(dim=1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            weights = weights / denom
            pooled = (out * weights).sum(dim=1)
        else:
            weights = self.attn_pool(out)
            pooled = (out * weights).sum(dim=1)

        return torch.nan_to_num(pooled, nan=0.0)


class TemporalMatchupNet(nn.Module):
    """Complete Neural Predictor encoding home & away team temporal states.

    Produces:
      - Unstructured latent states z_home, z_away
      - Structured latent states (attack, defense, form, consistency, tempo)
      - Matchup interaction vector [z_h, z_a, z_h - z_a, |z_h - z_a|]
      - 3-way outcome logits (Away, Draw, Home)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        arch: Literal["gru", "lstm", "transformer"] = "gru",
        structured: bool = False,
    ):
        super().__init__()
        self.encoder = SequenceEncoderCore(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            arch=arch,
        )
        self.structured = structured
        self.hidden_dim = hidden_dim

        if structured:
            sub_dim = max(8, hidden_dim // 5)
            self.attack_head = nn.Linear(hidden_dim, sub_dim)
            self.defense_head = nn.Linear(hidden_dim, sub_dim)
            self.form_head = nn.Linear(hidden_dim, sub_dim)
            self.consistency_head = nn.Linear(hidden_dim, sub_dim)
            self.tempo_head = nn.Linear(hidden_dim, sub_dim)
            self.latent_dim = sub_dim * 5
        else:
            self.latent_dim = hidden_dim

        # Matchup interaction dimension: z_h (D) + z_a (D) + diff (D) + abs_diff (D) = 4 * D
        matchup_dim = self.latent_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(matchup_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def extract_team_state(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Extract team latent vector z_team."""
        pooled = self.encoder(seq, mask)
        if self.structured:
            att = self.attack_head(pooled)
            dfn = self.defense_head(pooled)
            frm = self.form_head(pooled)
            cns = self.consistency_head(pooled)
            tmp = self.tempo_head(pooled)
            return torch.cat([att, dfn, frm, cns, tmp], dim=-1)
        return pooled

    def forward(
        self,
        seq_home: torch.Tensor,
        mask_home: torch.Tensor,
        seq_away: torch.Tensor,
        mask_away: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing logits and latent representations."""
        z_home = self.extract_team_state(seq_home, mask_home)
        z_away = self.extract_team_state(seq_away, mask_away)

        diff = z_home - z_away
        abs_diff = torch.abs(diff)

        matchup_repr = torch.cat([z_home, z_away, diff, abs_diff], dim=-1)
        logits = self.classifier(matchup_repr)
        return logits, z_home, z_away


class TemporalStatePredictor:
    """Scikit-Learn style wrapper for training and predicting with TemporalMatchupNet."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        arch: Literal["gru", "lstm", "transformer"] = "gru",
        structured: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 10,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.arch = arch
        self.structured = structured
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.random_state = random_state
        self.model: TemporalMatchupNet | None = None

    def _init_model(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = TemporalMatchupNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            arch=self.arch,
            structured=self.structured,
        ).to(self.device)

    def fit(
        self,
        seq_home: np.ndarray,
        mask_home: np.ndarray,
        seq_away: np.ndarray,
        mask_away: np.ndarray,
        y: np.ndarray,
        val_data: tuple | None = None,
        verbose: bool = False,
    ):
        self._init_model()
        self.model.train()

        t_sh = torch.tensor(seq_home, dtype=torch.float32)
        t_mh = torch.tensor(mask_home, dtype=torch.float32)
        t_sa = torch.tensor(seq_away, dtype=torch.float32)
        t_ma = torch.tensor(mask_away, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(t_sh, t_mh, t_sa, t_ma, t_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for ep in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for b_sh, b_mh, b_sa, b_ma, b_y in loader:
                b_sh, b_mh = b_sh.to(self.device), b_mh.to(self.device)
                b_sa, b_ma = b_sa.to(self.device), b_ma.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self.model(b_sh, b_mh, b_sa, b_ma)
                loss = criterion(logits, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(b_y)

            if verbose and (ep + 1) % max(1, self.epochs // 5) == 0:
                avg_loss = total_loss / len(dataset)
                print(f"  [Epoch {ep+1}/{self.epochs}] Loss: {avg_loss:.4f}")

        return self

    def predict_proba(
        self,
        seq_home: np.ndarray,
        mask_home: np.ndarray,
        seq_away: np.ndarray,
        mask_away: np.ndarray,
    ) -> np.ndarray:
        self.model.eval()
        t_sh = torch.tensor(seq_home, dtype=torch.float32)
        t_mh = torch.tensor(mask_home, dtype=torch.float32)
        t_sa = torch.tensor(seq_away, dtype=torch.float32)
        t_ma = torch.tensor(mask_away, dtype=torch.float32)

        dataset = TensorDataset(t_sh, t_mh, t_sa, t_ma)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for b_sh, b_mh, b_sa, b_ma in loader:
                b_sh, b_mh = b_sh.to(self.device), b_mh.to(self.device)
                b_sa, b_ma = b_sa.to(self.device), b_ma.to(self.device)
                logits, _, _ = self.model(b_sh, b_mh, b_sa, b_ma)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(probs)

        cat_p = np.vstack(probs_list)
        cat_p = np.nan_to_num(cat_p, nan=1.0 / 3.0)
        cat_p = np.clip(cat_p, 1e-12, 1.0)
        return cat_p / cat_p.sum(axis=1, keepdims=True)

    def extract_features(
        self,
        seq_home: np.ndarray,
        mask_home: np.ndarray,
        seq_away: np.ndarray,
        mask_away: np.ndarray,
    ) -> np.ndarray:
        """Extract the full matchup representation [z_h, z_a, diff, abs_diff]."""
        self.model.eval()
        t_sh = torch.tensor(seq_home, dtype=torch.float32)
        t_mh = torch.tensor(mask_home, dtype=torch.float32)
        t_sa = torch.tensor(seq_away, dtype=torch.float32)
        t_ma = torch.tensor(mask_away, dtype=torch.float32)

        dataset = TensorDataset(t_sh, t_mh, t_sa, t_ma)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        features_list = []
        with torch.no_grad():
            for b_sh, b_mh, b_sa, b_ma in loader:
                b_sh, b_mh = b_sh.to(self.device), b_mh.to(self.device)
                b_sa, b_ma = b_sa.to(self.device), b_ma.to(self.device)
                z_h = self.model.extract_team_state(b_sh, b_mh)
                z_a = self.model.extract_team_state(b_sa, b_ma)
                diff = z_h - z_a
                abs_diff = torch.abs(diff)
                matchup = torch.cat([z_h, z_a, diff, abs_diff], dim=-1).cpu().numpy()
                features_list.append(matchup)

        cat_f = np.vstack(features_list)
        return np.nan_to_num(cat_f, nan=0.0)
