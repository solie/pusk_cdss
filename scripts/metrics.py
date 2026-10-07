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
