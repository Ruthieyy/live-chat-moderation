"""Bag-of-words encoder used only for training and export.

Serving does not import this module, so the Cloud Run image does not ship
PyTorch. Dropout is applied with a Python branch on ``self.training`` and is
absent from the traced ONNX graph.
"""

from __future__ import annotations

import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        pad_idx: int = 0,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.dropout_p = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).to(dtype=embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if self.training and self.dropout_p > 0:
            pooled = torch.nn.functional.dropout(pooled, p=self.dropout_p, training=True)
        hidden = torch.relu(self.hidden(pooled))
        return self.output(hidden)
