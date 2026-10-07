"""Evaluation metrics and runtime utilities for pusk_cdss."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve compute device based on hardware availability."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: Path | str) -> list[dict]:
    """Read JSON lines file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_splits(
    csv_path: Path | str = "data/dummy_pusk.csv",
    splits_dir: Path | str = "data/splits",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Load JSONL splits, automatically generating them from CSV if not present."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "train.jsonl"
    dev_p = out_dir / "development.jsonl"
    cal_p = out_dir / "calibration.jsonl"
    test_p = out_dir / "test.jsonl"

    if train_p.exists() and dev_p.exists() and test_p.exists():
        cal_rows = read_jsonl(cal_p) if cal_p.exists() else read_jsonl(dev_p)
        return read_jsonl(train_p), read_jsonl(dev_p), cal_rows, read_jsonl(test_p)

    import pandas as pd
    import re
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Neither splits in {splits_dir} nor CSV at {csv_path} exist.")

    df = pd.read_csv(csv_p, low_memory=False)
    def clean_label(x):
        if pd.isna(x): return "Z00"
        m = re.findall(r'[A-Z]\d{2}(?:\.\d+)?', str(x))
        return m[0] if m else "Z00"

    df["clean_label"] = df["kode_penyakit"].apply(clean_label)
    # Calculate age if missing
    if "age" not in df.columns:
        def parse_age(row):
            try:
                t1 = pd.to_datetime(row["tanggal"])
                t0 = pd.to_datetime(row["tg_lahir"])
                return max(0.0, (t1 - t0).days / 365.25)
            except Exception:
                return 30.0
        df["age"] = df.apply(parse_age, axis=1)

    rows = []
    for idx, r in df.iterrows():
        jkl = str(r.get("jkl", "Laki-laki"))
        age = float(r.get("age", 30.0))
        bb = float(r.get("bb", 60.0))
        tb = float(r.get("tb", 165.0))
        sistole = int(r.get("sistole", 120))
        diastole = int(r.get("diastole", 80))
        anamnesa = str(r.get("anamnesa", "tidak ada keluhan"))
        label = str(r.get("clean_label", "Z00"))

        note = (
            f"Pasien: jenis kelamin {jkl}, usia {age:.1f} tahun, berat badan {bb:.0f} kg, "
            f"tinggi badan {tb:.0f} cm, tensi / tekanan darah {sistole}/{diastole} mmHg.\n"
            f"Keluhan utama (anamnesa): {anamnesa}"
        )
        rows.append({
            "id": f"row-{idx}",
            "label": label,
            "state": {"clinical_note": note}
        })

    # 70% train, 10% dev, 10% cal, 10% test
    n = len(rows)
    n_train = int(n * 0.70)
    n_dev = int(n * 0.10)
    n_cal = int(n * 0.10)

    train_rows = rows[:n_train]
    dev_rows = rows[n_train : n_train + n_dev]
    cal_rows = rows[n_train + n_dev : n_train + n_dev + n_cal]
    test_rows = rows[n_train + n_dev + n_cal :]

    write_jsonl(train_rows, train_p)
    write_jsonl(dev_rows, dev_p)
    write_jsonl(cal_rows, cal_p)
    write_jsonl(test_rows, test_p)

    return train_rows, dev_rows, cal_rows, test_rows
def write_jsonl(rows: Sequence[dict], path: Path | str) -> None:
    """Write dictionaries as a JSON lines file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def softmax(xs: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a list of floats."""
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / max(1e-12, s) for e in es]


def ece(probs: list[np.ndarray], answers: list[int], n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    conf = np.array([p.max() for p in probs], dtype=float)
    pred = np.array([int(np.argmax(p)) for p in probs])
    correct = pred == np.array(answers)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        out += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def brier(probs: list[np.ndarray], answers: list[int]) -> float:
    """Compute multi-class Brier score."""
    tot = 0.0
    for p, a in zip(probs, answers):
        y = np.zeros(len(p))
        y[int(a)] = 1.0
        tot += float(((p - y) ** 2).sum())
    return float(tot / max(1, len(probs)))


def fit_temperature(logit_sets: list[list[float]], answers: list[int]) -> tuple[float, float]:
    """Fit optimal single-parameter Platt temperature scaling via NLL."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.2, 5.0, 97):
        nll = 0.0
        for L, a in zip(logit_sets, answers):
            p = softmax([x / t for x in L])
            nll -= math.log(max(p[a], 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return float(best_t), float(best_nll / max(1, len(answers)))
