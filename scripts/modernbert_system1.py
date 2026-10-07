"""Vanilla PyTorch ModernBERT System-1 Decision Engine.

Non-autoregressive bidirectional encoder that scores all candidate options
simultaneously using gathered [MASK] marker token representations.
Zero third-party framework dependencies (pure torch + transformers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ModernBertSystem1(nn.Module):
    """Bidirectional ModernBERT encoder with marker-pointer option scoring."""

    def __init__(self, backbone: str = "answerdotai/ModernBERT-base", dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
        d = self.encoder.config.hidden_size  # 768
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
        qtype: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute option logits in a single forward pass.

        Args:
            input_ids: [B, L] token IDs.
            attention_mask: [B, L] boolean/long mask.
            marker_pos: [B, K] token indices of [MASK] markers.
            marker_mask: [B, K] boolean mask for active options (optional).
            qtype: [B] query type index (optional, default 0 for choice).

        Returns:
            logits: [B, K] unconstrained option logits.
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if qtype is not None:
            hidden = hidden + self.type_emb(qtype)[:, None, :]

        # Gather hidden state at each [MASK] marker: [B, K, d]
        index = marker_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        marker_hidden = torch.gather(hidden, 1, index)

        # Project to scalar logits: [B, K]
        logits = self.scorer(marker_hidden).squeeze(-1).float()
        if marker_mask is not None:
            logits = logits.masked_fill(~marker_mask, -1e4)
        return logits


def encode_encounter(
    tok: AutoTokenizer,
    clinical_note: str,
    criteria: dict[str, str],
    code_list: list[str],
    question: str = "Manakah diagnosis ICD-10 utama yang dicatat dokter?",
    max_len: int = 768,
    max_opt_len: int = 16,
) -> tuple[list[int], list[int]]:
    """Build the single-pass marker-pointer prompt."""
    mask_id = tok.mask_token_id
    head_ids = tok(f"choice question: {question}", add_special_tokens=False)["input_ids"]
    input_ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]

    marker_positions = []
    for code in code_list:
        desc = criteria.get(code, "")
        marker_positions.append(len(input_ids))
        opt_ids = tok(f" {code}: {desc}", add_special_tokens=False)["input_ids"][:max_opt_len]
        input_ids.extend([mask_id] + opt_ids)

    input_ids.append(tok.sep_token_id)
    room = max(16, max_len - len(input_ids) - 1)
    state_ids = tok(f" {clinical_note}", add_special_tokens=False)["input_ids"][:room]
    input_ids.extend(state_ids + [tok.sep_token_id])

    return input_ids, marker_positions


def load_modernbert_system1(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    backbone: str = "answerdotai/ModernBERT-base",
) -> tuple[ModernBertSystem1, AutoTokenizer]:
    """Load native ModernBERT System-1 model and extended tokenizer."""
    model_dir = Path(model_dir)
    tok_dir = model_dir / "tokenizer" if (model_dir / "tokenizer").exists() else model_dir

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    model = ModernBertSystem1(backbone=backbone)
    model.encoder.resize_token_embeddings(len(tok))

    weight_path = model_dir / "model.pt" if (model_dir / "model.pt").exists() else model_dir / "best_model.pt"
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model, tok
