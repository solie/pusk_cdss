"""Export native ELECTRA-Small (13.5M) System-1 Decision Engine to ONNX & INT8.

Exports:
  models/electra_system1/model.onnx (FP32)
  models/electra_system1/model_int8.onnx (INT8 dynamic quantization)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from electra_system1 import ElectraSystem1, encode_encounter
from pusk_data import get_icd_labels


def main():
    parser = argparse.ArgumentParser(description="Export ELECTRA-Small System-1 to ONNX")
    parser.add_argument("--model-dir", default="runs/electra")
    parser.add_argument("--out-dir", default="models/electra_system1")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx_path = out_dir / "model.onnx"
    int8_onnx_path = out_dir / "model_int8.onnx"

    print(f"Loading ELECTRA-Small System-1 from {model_dir}...")
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = ElectraSystem1("google/electra-small-discriminator")
    model.encoder.resize_token_embeddings(len(tok))
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    # 1. Prepare dummy inputs (max seq len 512, 29 options)
    B, L, K = 1, 256, 29
    dummy_input_ids = torch.randint(0, 1000, (B, L), dtype=torch.long)
    dummy_attention_mask = torch.ones((B, L), dtype=torch.bool)
    dummy_marker_pos = torch.linspace(20, 200, K, dtype=torch.long).unsqueeze(0)

    print(f"Exporting FP32 ONNX model to {fp32_onnx_path} (opset={args.opset})...")
    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_marker_pos),
        str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask", "marker_pos"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "marker_pos": {0: "batch_size", 1: "num_options"},
            "logits": {0: "batch_size", 1: "num_options"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    fp32_size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX exported in {time.time() - t0:.2f}s! File size: {fp32_size_mb:.1f} MB")

    # 2. Check model integrity
    print("Verifying ONNX model integrity...")
    onnx_model = onnx.load(str(fp32_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model integrity verified successfully!")

    # 3. Dynamic INT8 Quantization
    print("Quantizing to dynamic INT8...")
    t_q0 = time.time()
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)
    print(f"INT8 Quantization completed in {time.time() - t_q0:.2f}s! File size: {int8_size_mb:.1f} MB")

    # 4. Verify Numerical Parity against PyTorch
    print("\nVerifying numerical parity against PyTorch...")
    criteria = get_icd_labels("id")
    code_list = list(criteria.keys())
    sample_note = "Pasien demam tinggi 2 hari, batuk berdahak, pilek, badan nggregesi, tenggorokan sakit."
    ids, markers, length = encode_encounter(tok, sample_note, criteria, code_list)

    input_ids = torch.tensor([ids], dtype=torch.long)
    attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
    marker_pos = torch.tensor([markers], dtype=torch.long)

    with torch.no_grad():
        pt_logits = model(input_ids, attn_mask, marker_pos).numpy()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession(str(fp32_onnx_path), sess_options, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(int8_onnx_path), sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attn_mask.numpy(),
        "marker_pos": marker_pos.numpy(),
    }
    ort_fp32_logits = session_fp32.run(["logits"], ort_inputs)[0]
    ort_int8_logits = session_int8.run(["logits"], ort_inputs)[0]

    diff_fp32 = np.max(np.abs(pt_logits - ort_fp32_logits))
    diff_int8 = np.max(np.abs(pt_logits - ort_int8_logits))
    print(f"Max absolute logit difference |PyTorch - ONNX FP32|: {diff_fp32:.2e}")
    print(f"Max absolute logit difference |PyTorch - ONNX INT8|: {diff_int8:.2e}")

    # 5. Benchmark ONNX INT8 CPU Latency
    print("\nBenchmarking ONNX INT8 CPU Latency (warmup 5, runs 50)...")
    for _ in range(5):
        session_int8.run(["logits"], ort_inputs)

    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        session_int8.run(["logits"], ort_inputs)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_lat = np.mean(latencies)
    p50_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)

    print(f"ONNX INT8 CPU Latency:")
    print(f"  Mean: {mean_lat:.1f} ms / encounter ({1000.0/mean_lat:.1f} enc/s)")
    print(f"  p50:  {p50_lat:.1f} ms")
    print(f"  p95:  {p95_lat:.1f} ms")

    # Save tokenizer files to models/electra_system1/tokenizer
    tok.save_pretrained(out_dir / "tokenizer")
    print(f"\nSaved tokenizer and ONNX models to {out_dir}")


if __name__ == "__main__":
    main()
