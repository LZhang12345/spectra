from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def to_simplex(values: Sequence[float], eps: float = 0.0, fallback: str = "uniform") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, eps, None)
    total = float(arr.sum())
    if total <= 0.0:
        if fallback == "nan":
            return np.full(len(arr), np.nan, dtype=np.float64)
        if fallback != "uniform":
            raise ValueError(f"unknown simplex fallback: {fallback}")
        return np.full(len(arr), 1.0 / len(arr), dtype=np.float64)
    return arr / total


def dict_to_vec(d: Mapping[str, float], vocab: Sequence[str]) -> np.ndarray:
    return to_simplex([float(d.get(g, 0.0)) for g in vocab])


def is_valid_prediction(p: Sequence[float]) -> bool:
    arr = np.asarray(p, dtype=np.float64)
    return bool(arr.size > 0 and np.all(np.isfinite(arr)) and arr.sum() > 0.0)


def ranking_to_vec(ranking: Iterable[str], vocab: Sequence[str]) -> np.ndarray:
    idx = {g: i for i, g in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    seen = set()
    for label in ranking:
        if label in idx and label not in seen:
            counts[idx[label]] += 1.0
            seen.add(label)
    return to_simplex(counts)


def js_divergence_bits(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    if not is_valid_prediction(p):
        return float("nan")
    p_arr = to_simplex(p, eps=eps)
    q_arr = to_simplex(q, eps=eps)
    m = 0.5 * (p_arr + q_arr)
    kl_pm = float(np.sum(p_arr * (np.log(p_arr) - np.log(m))))
    kl_qm = float(np.sum(q_arr * (np.log(q_arr) - np.log(m))))
    return (0.5 * kl_pm + 0.5 * kl_qm) / math.log(2.0)


def ndcg_at_k(pred: Sequence[float], truth: Sequence[float], k: int = 10) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred_arr)):
        return float("nan")
    truth_arr = to_simplex(truth)
    order = np.lexsort((np.arange(len(pred_arr)), -pred_arr))[:k]
    ideal = np.argsort(-truth_arr)[:k]

    def dcg(indices: np.ndarray) -> float:
        return float(sum(truth_arr[j] / math.log2(pos + 2.0) for pos, j in enumerate(indices)))

    denom = dcg(ideal)
    return dcg(order) / denom if denom > 0.0 else 0.0


def entropy_bits(p: Sequence[float], eps: float = 1e-12) -> float:
    if not is_valid_prediction(p):
        return float("nan")
    arr = to_simplex(p, eps=eps)
    return float(-np.sum(arr * np.log2(arr)))


def topk_exposure_entropy(pred_matrix: np.ndarray, k: int) -> float:
    preds = np.atleast_2d(np.asarray(pred_matrix, dtype=np.float64))
    counts = np.zeros(preds.shape[1], dtype=np.float64)
    for row in preds:
        order = np.lexsort((np.arange(len(row)), -row))[:k]
        counts[order] += 1.0
    share = counts / counts.sum()
    nonzero = share[share > 0.0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def global_buckets(truth_matrix: np.ndarray) -> dict[str, list[int]]:
    mass = np.asarray(truth_matrix, dtype=np.float64).mean(axis=0)
    order = np.argsort(-mass)
    n = len(order)
    head_cut = max(1, n // 3)
    tail_cut = max(2, 2 * n // 3)
    head = sorted(order[:head_cut].tolist())
    mid = sorted(order[head_cut:tail_cut].tolist())
    tail = sorted(order[tail_cut:].tolist())
    return {"head": head, "mid": mid, "tail": tail}


def bucket_mass(p: Sequence[float], indices: Sequence[int]) -> float:
    if not is_valid_prediction(p):
        return float("nan")
    arr = to_simplex(p)
    return float(arr[list(indices)].sum()) if indices else 0.0


def summarize_rows(rows: list[dict], value_keys: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    for key in value_keys:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        out[f"{key}_n_finite"] = float(len(vals))
        out[f"{key}_n_nonfinite"] = float(len(rows) - len(vals))
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out
