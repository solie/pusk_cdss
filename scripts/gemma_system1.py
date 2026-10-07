"""Vanilla HuggingFace Gemma 3 System-1 Decision Engine.

Cross-encoder pairwise candidate scoring using Google Gemma 3 270M and LoRA.
Zero third-party repository dependencies (pure transformers + peft).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Gemma3TextForSequenceClassification


def encode_candidate(tok: AutoTokenizer, state: str, question: str, option: str, max_len: int = 192) -> list[int]:
    """Encode (State, Question, Option) with intact tail."""
    tail_ids = tok("\n\nQuestion:\n" + question + "\n\nOption:\n" + option, add_special_tokens=False)["input_ids"]
    if len(tail_ids) >= max_len:
        return tail_ids[-max_len:]
    head_ids = tok("State:\n" + state, add_special_tokens=False)["input_ids"]
    return head_ids[: max_len - len(tail_ids)] + tail_ids


def pad_sequences(seqs: list[list[int]], pad_id: int, max_len: int = 192) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad sequence batch for pooled scalar scoring."""
    L = min(max(len(s) for s in seqs), max_len)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return ids, attn


def load_gemma_system1(
    adapter_dir: str | Path,
    base_model: str = "google/gemma-3-270m",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load native Gemma 3 270M backbone with trained LoRA adapter."""
    adapter_path = Path(adapter_dir)
    tok_dir = adapter_path.parent / "tokenizer" if (adapter_path.parent / "tokenizer").exists() else adapter_path

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = Gemma3TextForSequenceClassification.from_pretrained(
        base_model, num_labels=1, torch_dtype=dtype
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    # Resize embedding table if expanded vocabulary was added
    if len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def score_encounter_gemma(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    clinical_note: str,
    criteria: dict[str, str],
    code_list: list[str],
    device: torch.device,
    question: str = "Manakah diagnosis ICD-10 utama yang dicatat dokter?",
    temperature: float = 1.65,
    max_len: int = 192,
) -> tuple[list[float], list[str]]:
    """Score all candidate options for a single patient visit in one batched forward pass."""
    seqs = []
    for code in code_list:
        desc = criteria.get(code, "")
        opt_str = f"{code}: {desc}" if desc else code
        seqs.append(encode_candidate(tok, clinical_note, question, opt_str, max_len=max_len))

    ids, attn = pad_sequences(seqs, tok.pad_token_id, max_len=max_len)
    out = model(input_ids=ids.to(device), attention_mask=attn.to(device))
    logits = out.logits.squeeze(-1).float().cpu().numpy()

    # Temperature-scaled softmax
    scaled = logits / temperature
    scaled -= np.max(scaled)
    exp = np.exp(scaled)
    probs = (exp / np.sum(exp)).tolist()

    ranked_indices = np.argsort(-np.array(probs))
    ranked_codes = [code_list[i] for i in ranked_indices]
    return probs, ranked_codes
