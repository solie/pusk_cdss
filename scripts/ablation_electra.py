"""Ablation Study: Clinical Feature Contributions in ELECTRA-Small System-1.

Evaluates on the out-of-city holdout (n=3,758) across three input conditions:
1. Full Note (Anamnesis + Patient Vitals + Demographics)
2. Anamnesis Only (Vitals & Demographics Ablated)
3. Vitals & Demographics Only (Anamnesis Ablated)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from electra_system1 import ElectraSystem1, collate_batch, resolve_device
from poncol_data import get_icd_labels
from system_one_gemma_experiment import brier, ece, read_jsonl


def encode_ablation(
    tok: AutoTokenizer,
    row: dict,
    criteria: dict[str, str],
    code_list: list[str],
    condition: str = "full",
    max_len: int = 512,
    max_opt_len: int = 6,
) -> tuple[list[int], list[int], int]:
    """Encode encounter under specific ablation condition."""
    mask_id = tok.mask_token_id
    question = "Manakah diagnosis ICD-10 utama yang dicatat dokter?"
    head_ids = tok(f"pertanyaan: {question}", add_special_tokens=False)["input_ids"]
    input_ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]

    marker_positions = []
    for code in code_list:
        desc = criteria.get(code, "")
        marker_positions.append(len(input_ids))
        opt_ids = tok(f" {code}: {desc}", add_special_tokens=False)["input_ids"][:max_opt_len]
        input_ids.extend([mask_id] + opt_ids)
    input_ids.append(tok.sep_token_id)

    # Build state text based on condition
    full_text = row["state"]["clinical_note"]
    parts = full_text.split("Keluhan utama (anamnesa):")
    header = parts[0].strip() if len(parts) > 1 else ""
    anamnesa = parts[1].strip() if len(parts) > 1 else full_text

    if condition == "full":
        state_text = full_text
    elif condition == "anamnesa_only":
        state_text = f"Keluhan utama (anamnesa): {anamnesa}"
    elif condition == "vitals_only":
        state_text = f"{header}\nKeluhan utama (anamnesa): tidak ada keluhan"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    room = max(16, max_len - len(input_ids) - 1)
    state_ids = tok(f" catatan: {state_text}", add_special_tokens=False)["input_ids"][:room]
    input_ids.extend(state_ids + [tok.sep_token_id])

    return input_ids, marker_positions, len(input_ids)


@torch.no_grad()
def evaluate_condition(
    model: ElectraSystem1,
    tok: AutoTokenizer,
    rows: list[dict],
    criteria: dict[str, str],
    code_list: list[str],
    device: torch.device,
    condition: str = "full",
    temperature: float = 0.90,
    batch_size: int = 64,
) -> dict:
    encoded_items = []
    answers = []
    labels_all = []

    for r in rows:
        ids, markers, l = encode_ablation(tok, r, criteria, code_list, condition=condition)
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

    n = len(rows)
    top1 = sum(int(np.argmax(p)) == ans for p, ans in zip(prob_sets, answers))
    top2 = sum(ans in np.argsort(p)[-2:] for p, ans in zip(prob_sets, answers))
    top3 = sum(ans in np.argsort(p)[-3:] for p, ans in zip(prob_sets, answers))

    # Fraction predicting Z00
    z00_idx = code_list.index("Z00") if "Z00" in code_list else -1
    z00_rate = sum(int(np.argmax(p)) == z00_idx for p in prob_sets) / n if z00_idx != -1 else 0.0

    return {
        "condition": condition,
        "n": n,
        "top1_accuracy": top1 / n,
        "top2_accuracy": top2 / n,
        "top3_accuracy": top3 / n,
        "macro_f1": f1_score(labels_all, pred_labels, average="macro", zero_division=0),
        "ece": ece(prob_sets, answers),
        "brier": brier(prob_sets, answers),
        "z00_predicted_rate": z00_rate,
    }


def main():
    device = resolve_device("auto")
    model_dir = Path("runs/electra")
    print(f"Loading ELECTRA model from {model_dir} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    criteria = get_icd_labels("id")
    code_list = list(criteria.keys())

    model = ElectraSystem1("google/electra-small-discriminator").to(device)
    model.encoder.resize_token_embeddings(len(tok))
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    test_rows = read_jsonl(Path("runs/xxx-holdout-id/data/test.jsonl"))
    print(f"Evaluating {len(test_rows)} encounters across 3 ablation conditions...")

    results = {}
    for cond in ["full", "anamnesa_only", "vitals_only"]:
        print(f"Running condition: {cond}...")
        res = evaluate_condition(model, tok, test_rows, criteria, code_list, device, condition=cond)
        results[cond] = res
        print(
            f"  [{cond:<14}] Top-1: {res['top1_accuracy']*100:.2f}% | "
            f"Top-3: {res['top3_accuracy']*100:.2f}% | "
            f"Macro-F1: {res['macro_f1']:.4f} | "
            f"Z00 Rate: {res['z00_predicted_rate']*100:.1f}%"
        )

    out_file = Path("runs/electra/ablation_results.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved ablation results to {out_file}")


if __name__ == "__main__":
    main()
