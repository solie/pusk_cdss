"""Vanilla PyTorch ELECTRA-Small (13.5M) System-1 Decision Engine.

Ultra-compact non-autoregressive bidirectional encoder with marker-pointer option scoring.
Targeted to microcontrollers and low-resource edge hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poncol_data import get_icd_labels
from system_one_gemma_experiment import brier, ece, fit_temperature, read_jsonl, resolve_device, softmax


class ElectraSystem1(nn.Module):
    """Bidirectional ELECTRA-small encoder with marker-pointer option scoring."""

    def __init__(self, backbone: str = "google/electra-small-discriminator", dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        d = self.encoder.config.hidden_size  # 256
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
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
    max_len: int = 512,
    max_opt_len: int = 6,
) -> tuple[list[int], list[int], int]:
    """Build single-pass marker-pointer prompt within ELECTRA 512-token limit."""
    mask_id = tok.mask_token_id
    head_ids = tok(f"pertanyaan: {question}", add_special_tokens=False)["input_ids"]
    input_ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]

    marker_positions = []
    for code in code_list:
        desc = criteria.get(code, "")
        marker_positions.append(len(input_ids))
        opt_ids = tok(f" {code}: {desc}", add_special_tokens=False)["input_ids"][:max_opt_len]
        input_ids.extend([mask_id] + opt_ids)

    input_ids.append(tok.sep_token_id)
    # Remaining budget for clinical note
    room = max(16, max_len - len(input_ids) - 1)
    state_ids = tok(f" catatan: {clinical_note}", add_special_tokens=False)["input_ids"][:room]
    input_ids.extend(state_ids + [tok.sep_token_id])

    return input_ids, marker_positions, len(input_ids)


def collate_batch(items: list[tuple[list[int], list[int], int]], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    B = len(items)
    max_l = max(it[2] for it in items)
    K = len(items[0][1])

    input_ids = torch.full((B, max_l), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_l), dtype=torch.bool, device=device)
    marker_pos = torch.zeros((B, K), dtype=torch.long, device=device)
    marker_mask = torch.ones((B, K), dtype=torch.bool, device=device)

    for i, (ids, markers, length) in enumerate(items):
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :length] = True
        marker_pos[i, : len(markers)] = torch.tensor(markers, dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
    }


@torch.no_grad()
def evaluate_model(
    model: ElectraSystem1,
    tok: AutoTokenizer,
    rows: list[dict],
    criteria: dict[str, str],
    code_list: list[str],
    device: torch.device,
    batch_size: int = 32,
    temperature: float = 1.0,
    tag: str = "eval",
) -> tuple[dict, list[list[float]], list[int], list[str]]:
    model.eval()
    t0 = time.time()

    encoded_items = []
    answers = []
    labels_all = []

    for r in rows:
        ids, markers, l = encode_encounter(tok, r["state"]["clinical_note"], criteria, code_list)
        encoded_items.append((ids, markers, l))
        true_code = r["label"]
        ans_idx = code_list.index(true_code) if true_code in code_list else 0
        answers.append(ans_idx)
        labels_all.append(true_code)

    prob_sets: list[np.ndarray] = []
    pred_labels: list[str] = []

    for i in range(0, len(encoded_items), batch_size):
        chunk = encoded_items[i : i + batch_size]
        batch = collate_batch(chunk, tok.pad_token_id, device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"])
        probs = F.softmax(logits / max(1e-4, temperature), dim=-1).cpu().numpy()
        for p in probs:
            prob_sets.append(p)
            pred_labels.append(code_list[int(np.argmax(p))])
    total_time = time.time() - t0
    n = len(rows)
    ms_per_enc = (total_time / n) * 1000.0

    top1 = sum(int(np.argmax(p)) == ans for p, ans in zip(prob_sets, answers))
    top2 = sum(ans in np.argsort(p)[-2:] for p, ans in zip(prob_sets, answers))
    top3 = sum(ans in np.argsort(p)[-3:] for p, ans in zip(prob_sets, answers))

    metrics = {
        "tag": tag,
        "n": n,
        "top1_accuracy": top1 / n,
        "top2_accuracy": top2 / n,
        "top3_accuracy": top3 / n,
        "macro_f1": f1_score(labels_all, pred_labels, average="macro", zero_division=0),
        "ece": ece(prob_sets, answers),
        "brier": brier(prob_sets, answers),
        "mean_confidence": float(np.mean([p.max() for p in prob_sets])),
        "temperature": temperature,
        "total_seconds": total_time,
        "ms_per_encounter": ms_per_enc,
    }
    return metrics, prob_sets, answers, labels_all


def train_electra(
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 3e-4,
    label_smoothing: float = 0.10,
    output_dir: str = "runs/electra",
    resume: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"Device: {device}")

    # Load Tokenizer & add clinical tokens
    backbone_id = "google/electra-small-discriminator"
    print(f"Loading tokenizer: {backbone_id}...")
    tok = AutoTokenizer.from_pretrained(backbone_id)
    tokens_file = Path("clinical_tokens.json")
    if tokens_file.exists():
        tokens = json.loads(tokens_file.read_text())
        n_added = tok.add_tokens(tokens)
        print(f"Added {n_added} clinical tokens to vocabulary (total: {len(tok)})")

    # Load criteria & code list
    criteria = get_icd_labels("id")
    code_list = list(criteria.keys())
    print(f"Loaded {len(code_list)} ICD-10 diagnostic categories")

    # Load model
    print(f"Initializing ElectraSystem1 (13.5M params)...")
    model = ElectraSystem1(backbone_id).to(device)
    model.encoder.resize_token_embeddings(len(tok))
    if resume and (out_path / "best_model.pt").exists():
        print(f"Loading existing checkpoint from {out_path / 'best_model.pt'} to continue training...")
        model.load_state_dict(torch.load(out_path / "best_model.pt", map_location=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    # Load datasets
    train_rows = read_jsonl(Path("runs/xxx-holdout-id/data/train.jsonl"))
    dev_rows = read_jsonl(Path("runs/xxx-holdout-id/data/development.jsonl"))
    cal_rows = read_jsonl(Path("runs/xxx-holdout-id/data/calibration.jsonl"))
    test_rows = read_jsonl(Path("runs/xxx-holdout-id/data/test.jsonl"))
    print(f"Datasets: train={len(train_rows)}, dev={len(dev_rows)}, cal={len(cal_rows)}, test={len(test_rows)}")

    # Pre-encode training data for maximum speed
    print("Pre-encoding training encounters...")
    train_encoded = []
    train_targets = []
    for r in train_rows:
        ids, markers, l = encode_encounter(tok, r["state"]["clinical_note"], criteria, code_list)
        train_encoded.append((ids, markers, l))
        true_code = r["label"]
        train_targets.append(code_list.index(true_code) if true_code in code_list else 0)

    # Optimizer & Scheduler
    total_steps = (len(train_encoded) // batch_size) * epochs
    warmup_steps = int(total_steps * 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print(f"\n--- Training {epochs} Epochs ({total_steps} steps, batch_size={batch_size}) ---")
    t_start = time.time()
    history_file = out_path / "training_history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    epoch_offset = len(history)
    if resume:
        dev_init, _, _, _ = evaluate_model(model, tok, dev_rows, criteria, code_list, device, tag="dev_init")
        best_dev_acc = dev_init["top1_accuracy"]
        print(f"Resumed baseline (after {epoch_offset} epochs) Dev Top-1: {best_dev_acc*100:.2f}% | Top-3: {dev_init['top3_accuracy']*100:.2f}%")
    else:
        best_dev_acc = 0.0
        epoch_offset = 0
        history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = np.random.permutation(len(train_encoded))
        t_ep = time.time()

        for step, idx in enumerate(range(0, len(perm), batch_size)):
            batch_idxs = perm[idx : idx + batch_size]
            items = [train_encoded[i] for i in batch_idxs]
            targets = torch.tensor([train_targets[i] for i in batch_idxs], dtype=torch.long, device=device)

            batch = collate_batch(items, tok.pad_token_id, device)
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"])
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if (step + 1) % 150 == 0:
                print(f"  Epoch {epoch}/{epochs} | Step {step+1}/{len(perm)//batch_size} | Loss: {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / (len(perm) // batch_size)
        ep_sec = time.time() - t_ep
        print(f"Epoch {epoch} finished in {ep_sec:.1f}s | Avg Loss: {avg_loss:.4f}")

        # Quick validation
        dev_metrics, _, _, _ = evaluate_model(model, tok, dev_rows, criteria, code_list, device, tag="dev")
        is_best = dev_metrics["top1_accuracy"] > best_dev_acc
        if is_best:
            best_dev_acc = dev_metrics["top1_accuracy"]
            torch.save(model.state_dict(), out_path / "best_model.pt")
            tok.save_pretrained(out_path / "tokenizer")
            print(f"  --> NEW BEST CHECKPOINT SAVED (Dev Top-1: {best_dev_acc*100:.2f}%)")
        else:
            print(f"  --> No improvement (Best Dev Top-1: {best_dev_acc*100:.2f}%)")

        history.append({
            "total_epoch": epoch_offset + epoch,
            "run_epoch": epoch,
            "train_loss": avg_loss,
            "dev_top1": dev_metrics["top1_accuracy"],
            "dev_top3": dev_metrics["top3_accuracy"],
            "dev_ece": dev_metrics["ece"],
            "is_best": is_best,
        })
        history_file.write_text(json.dumps(history, indent=2))
    print(f"\nTraining completed in {(time.time() - t_start)/60:.1f} minutes!")

    # Load best checkpoint
    print("\nLoading best model checkpoint for evaluation...")
    model.load_state_dict(torch.load(out_path / "best_model.pt", map_location=device))

    # Calibrate on calibration set
    print("Calibrating on calibration set (3,381 visits)...")
    _, cal_probs, cal_answers, _ = evaluate_model(model, tok, cal_rows, criteria, code_list, device, tag="cal")
    cal_logits = [np.log(np.maximum(p, 1e-12)) for p in cal_probs]
    T, _ = fit_temperature(cal_logits, cal_answers)
    print(f"Optimal Calibration Temperature: T = {T:.3f}")

    # Evaluate on XXX Holdout
    print("\nEvaluating on Out-of-City XXX Holdout (3,758 visits)...")
    raw_metrics, _, _, _ = evaluate_model(model, tok, test_rows, criteria, code_list, device, temperature=1.0, tag="raw")
    cal_metrics, test_probs, test_answers, test_labels = evaluate_model(
        model, tok, test_rows, criteria, code_list, device, temperature=T, tag="calibrated"
    )

    results = {
        "model": "ELECTRA-Small System-1",
        "parameters": f"{n_params/1e6:.1f}M",
        "raw_test": raw_metrics,
        "calibrated_test": cal_metrics,
        "optimal_temperature": T,
    }

    results_file = out_path / "xxx_holdout_metrics.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\n=== FINAL ELECTRA-SMALL (13.5M) XXX HOLDOUT BENCHMARK ===")
    print(f"  Top-1 Accuracy: {cal_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-2 Accuracy: {cal_metrics['top2_accuracy']*100:.2f}%")
    print(f"  Top-3 Accuracy: {cal_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Macro-F1:       {cal_metrics['macro_f1']:.4f}")
    print(f"  ECE (15 bins):  {cal_metrics['ece']:.4f}")
    print(f"  Brier Score:    {cal_metrics['brier']:.4f}")
    print(f"  Mean Latency:   {cal_metrics['ms_per_encounter']:.1f} ms / encounter")
    print(f"\nSaved metrics to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()
    if args.eval_only:
        out_path = Path("runs/electra")
        device = resolve_device("auto")
        tok = AutoTokenizer.from_pretrained(out_path / "tokenizer")
        criteria = get_icd_labels("id")
        code_list = list(criteria.keys())
        model = ElectraSystem1("google/electra-small-discriminator").to(device)
        model.encoder.resize_token_embeddings(len(tok))
        model.load_state_dict(torch.load(out_path / "best_model.pt", map_location=device))
        n_params = sum(p.numel() for p in model.parameters())
        cal_rows = read_jsonl(Path("runs/xxx-holdout-id/data/calibration.jsonl"))
        test_rows = read_jsonl(Path("runs/xxx-holdout-id/data/test.jsonl"))
        print("Calibrating on calibration set (3,381 visits)...")
        _, cal_probs, cal_answers, _ = evaluate_model(model, tok, cal_rows, criteria, code_list, device, tag="cal")
        cal_logits = [np.log(np.maximum(p, 1e-12)) for p in cal_probs]
        T, _ = fit_temperature(cal_logits, cal_answers)
        print(f"Optimal Calibration Temperature: T = {T:.3f}")
        raw_metrics, _, _, _ = evaluate_model(model, tok, test_rows, criteria, code_list, device, temperature=1.0, tag="raw")
        cal_metrics, test_probs, test_answers, test_labels = evaluate_model(
            model, tok, test_rows, criteria, code_list, device, temperature=T, tag="calibrated"
        )
        results = {
            "model": "ELECTRA-Small System-1",
            "parameters": f"{n_params/1e6:.1f}M",
            "raw_test": raw_metrics,
            "calibrated_test": cal_metrics,
            "optimal_temperature": T,
        }
        results_file = out_path / "xxx_holdout_metrics.json"
        results_file.write_text(json.dumps(results, indent=2))
        print(f"\n=== FINAL ELECTRA-SMALL (13.5M) XXX HOLDOUT BENCHMARK ===")
        print(f"  Top-1 Accuracy: {cal_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-2 Accuracy: {cal_metrics['top2_accuracy']*100:.2f}%")
        print(f"  Top-3 Accuracy: {cal_metrics['top3_accuracy']*100:.2f}%")
        print(f"  Macro-F1:       {cal_metrics['macro_f1']:.4f}")
        print(f"  ECE (15 bins):  {cal_metrics['ece']:.4f}")
        print(f"  Brier Score:    {cal_metrics['brier']:.4f}")
        print(f"  Mean Latency:   {cal_metrics['ms_per_encounter']:.1f} ms / encounter")
        print(f"\nSaved metrics to {results_file}")
    else:
        train_electra(epochs=args.epochs, lr=args.lr, resume=args.resume)
