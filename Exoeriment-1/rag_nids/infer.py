"""Inference, evaluation, and retrieval-explanation printer."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import CICDataset
from .pipeline import RAGNIDS


def _display_order(label_names: list[str]) -> list[int]:
    """Return indices into label_names sorted alphabetically, BENIGN first if present.

    We reorder for display only — the sklearn confusion_matrix is computed over
    integer labels 0..N-1, so we permute rows/cols of the resulting matrix.
    """
    order = sorted(range(len(label_names)), key=lambda i: label_names[i].lower())
    benign_pos = next((p for p, i in enumerate(order) if label_names[i].upper() == "BENIGN"), None)
    if benign_pos is not None and benign_pos != 0:
        order = [order[benign_pos]] + [i for p, i in enumerate(order) if p != benign_pos]
    return order


def confusion_matrix(
    trues: np.ndarray, preds: np.ndarray, label_names: list[str],
    out_dir: Optional[str] = None, top_k_mistakes: int = 3,
) -> dict:
    """Full N×N confusion matrix report.

    Prints raw counts + row-normalized rates (per-class recall) and the top-K
    most common misclassification pairs. Optionally writes two CSVs to out_dir:
    confusion_matrix_counts.csv and confusion_matrix_rates.csv.

    Class ordering: alphabetical with BENIGN first. The same ordering is used
    for both printed output and CSVs so delta matrices (run B minus run A) are
    trivially computable.

    Returns a dict with keys: counts (DataFrame), rates (DataFrame),
    top_misclassifications (list of (true, pred, count, rate_of_true) tuples).
    """
    num_classes = len(label_names)
    # Compute over all class indices 0..N-1 so absent labels still produce zero rows/cols.
    cm = sk_confusion_matrix(trues, preds, labels=range(num_classes))

    order = _display_order(label_names)
    names = [label_names[i] for i in order]
    cm_ord = cm[np.ix_(order, order)]

    counts_df = pd.DataFrame(cm_ord, index=names, columns=names)
    counts_df.index.name = "true"
    counts_df.columns.name = "pred"

    row_sums = cm_ord.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(row_sums > 0, cm_ord / np.maximum(row_sums, 1), 0.0)
    rates_df = pd.DataFrame(rates, index=names, columns=names).round(4)
    rates_df.index.name = "true"
    rates_df.columns.name = "pred"

    # Find top-K off-diagonal cells by absolute count
    off_diag = cm_ord.copy().astype(np.int64)
    np.fill_diagonal(off_diag, 0)
    flat_idx = np.argsort(off_diag, axis=None)[::-1][:top_k_mistakes]
    top_pairs = []
    for flat in flat_idx:
        i, j = np.unravel_index(flat, off_diag.shape)
        count = int(off_diag[i, j])
        if count == 0:
            continue
        true_total = int(row_sums[i, 0])
        rate = count / true_total if true_total > 0 else 0.0
        top_pairs.append((names[i], names[j], count, rate))

    # Print
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", num_classes + 2)
    print("\n[confusion matrix] raw counts (rows=true, cols=pred):")
    print(counts_df.to_string())
    print("\n[confusion matrix] row-normalized rates (per-class recall):")
    print(rates_df.to_string())
    print("\n[confusion matrix] top misclassifications:")
    for true_name, pred_name, count, rate in top_pairs:
        print(f"  {true_name} mistaken for {pred_name}: {count} times "
              f"({rate:.1%} of {true_name}'s test samples)")

    # Optional: persist to disk
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        counts_df.to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"))
        rates_df.to_csv(os.path.join(out_dir, "confusion_matrix_rates.csv"))

    return {"counts": counts_df, "rates": rates_df, "top_misclassifications": top_pairs}


@torch.no_grad()
def evaluate(model: RAGNIDS, X: np.ndarray, y: np.ndarray,
             label_names: list[str], batch_size: int = 512, device: str = "cpu",
             cm_out_dir: Optional[str] = None) -> dict:
    model.eval().to(device)
    loader = DataLoader(CICDataset(X, y), batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, _, _, _, _ = model(xb, exclude_self=False)  # FIX: forward() now returns sims too
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    report = classification_report(trues, preds, target_names=label_names,
                                   digits=4, zero_division=0)
    print(report)
    print(f"macro-F1: {macro_f1:.4f}")

    # Always produce the full N×N confusion matrix report. Keeps plumbing in one
    # place so initial POC runs and improvement reruns produce comparable CSVs.
    cm = confusion_matrix(trues, preds, label_names, out_dir=cm_out_dir)

    return {"macro_f1": macro_f1, "preds": preds, "trues": trues, "confusion": cm}


@torch.no_grad()
def explain(model: RAGNIDS, x: torch.Tensor, label_names: list[str],
            top_n: int = 5, device: str = "cpu") -> None:
    """Print top-k neighbours, their labels, similarities, and the model's prediction."""
    model.eval().to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    preds = model.predict(x.to(device))
    for i, p in enumerate(preds):
        print(f"\n--- query #{i} ---")
        print(f"  predicted: {label_names[p.label]}  (confidence={p.confidence:.3f})")
        print(f"  top-{top_n} neighbours:")
        order = np.argsort(-p.neighbor_sims)[:top_n]
        for rank, j in enumerate(order, 1):
            print(f"    {rank}. id={p.neighbor_ids[j]:>7}  "
                  f"label={label_names[p.neighbor_labels[j]]:<20}  sim={p.neighbor_sims[j]:.3f}")


@torch.no_grad()
def evaluate_two_stage(
    two_stage, X: np.ndarray, y: np.ndarray, label_names: list[str],
    benign_label: int, attack_class_names: list[str],
    batch_size: int = 512, device: str = "cpu",
    cm_out_dir: Optional[str] = None,
) -> dict:
    """Three-tier evaluation for the two-stage pipeline.

    Tier 1 — Stage 1 binary (BENIGN vs attack): precision/recall/F1 of the VAE filter.
    Tier 2 — Stage 2 attack-only classification on truly-flagged attacks: macro-F1
             over the attack label space (uses ground-truth attack mask, ignores Stage 1
             misses; this isolates Stage 2 quality).
    Tier 3 — End-to-end N-class classification: macro-F1 over the original label space.
             A Stage 1 false negative collapses to BENIGN regardless of true class.
    """
    out = two_stage.predict_array(X, batch_size=batch_size, device=device)
    preds = out["preds"]
    scores = out["stage1_scores"]
    flagged = out["stage1_attack"]

    is_attack = (y != benign_label)

    # ----- Tier 1: Stage 1 binary -----
    tp = int((flagged & is_attack).sum())
    fp = int((flagged & ~is_attack).sum())
    fn = int((~flagged & is_attack).sum())
    tn = int((~flagged & ~is_attack).sum())
    s1_prec = tp / max(tp + fp, 1)
    s1_rec = tp / max(tp + fn, 1)
    s1_f1 = 2 * s1_prec * s1_rec / max(s1_prec + s1_rec, 1e-9)
    print(f"\n[stage1] binary: prec={s1_prec:.4f}  rec={s1_rec:.4f}  f1={s1_f1:.4f}  "
          f"(tp={tp} fp={fp} fn={fn} tn={tn})")

    # ----- Tier 2: Stage 2 attack-only on truly-attack rows that Stage 1 forwarded -----
    # We need Stage 2 predictions on the GROUND-TRUTH attack rows, not just flagged ones,
    # to fairly score the classifier. Re-run Stage 2 on every true-attack row.
    s2_macro_f1 = float("nan")
    s2_report_lines = []
    if is_attack.any():
        atk_idx = np.where(is_attack)[0]
        X_atk = X[atk_idx]
        two_stage.stage2.eval().to(device)
        chunks = []
        for i in range(0, len(X_atk), batch_size):
            xb = torch.from_numpy(X_atk[i:i + batch_size]).to(device)
            logits, *_ = two_stage.stage2(xb, exclude_self=False)
            chunks.append(logits.argmax(-1).cpu().numpy())
        s2_preds_attack_space = np.concatenate(chunks)

        # y_atk in attack-only label space
        lut = two_stage._lut.cpu().numpy()
        # Build the inverse: original label -> new label (skip benign)
        orig_to_new = {int(lut[i]): i for i in range(len(lut)) if lut[i] >= 0}
        y_atk_new = np.array([orig_to_new[int(v)] for v in y[atk_idx]], dtype=np.int64)

        s2_macro_f1 = f1_score(y_atk_new, s2_preds_attack_space,
                               average="macro", zero_division=0)
        print(f"\n[stage2] attack-only macro-F1: {s2_macro_f1:.4f}")
        report = classification_report(y_atk_new, s2_preds_attack_space,
                                       target_names=attack_class_names,
                                       digits=4, zero_division=0)
        print(report)
        s2_report_lines = report.splitlines()

    # ----- Tier 3: End-to-end N-class -----
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    print(f"\n[two-stage] end-to-end N-class macro-F1: {macro_f1:.4f}")
    e2e_report = classification_report(y, preds, target_names=label_names,
                                       digits=4, zero_division=0)
    print(e2e_report)

    cm = confusion_matrix(y, preds, label_names, out_dir=cm_out_dir)

    metrics = {
        "stage1_precision": s1_prec, "stage1_recall": s1_rec, "stage1_f1": s1_f1,
        "stage1_tp": tp, "stage1_fp": fp, "stage1_fn": fn, "stage1_tn": tn,
        "stage2_macro_f1": s2_macro_f1,
        "e2e_macro_f1": macro_f1,
    }
    return {
        "macro_f1": macro_f1, "preds": preds, "trues": y,
        "stage1_scores": scores, "stage1_flagged": flagged,
        "metrics": metrics, "confusion": cm,
    }


def run_writeback(model: RAGNIDS, x: torch.Tensor, pred, attack_class_ids: set[int],
                  min_confidence: float = 0.95) -> bool:
    """Add a high-confidence attack prediction to the index."""
    with torch.no_grad():
        z = model.encoder(x.unsqueeze(0)).cpu().numpy()[0]
    return model.index.writeback(
        embedding=z, label=pred.label,
        min_confidence=min_confidence, confidence=pred.confidence,
        is_attack=pred.label in attack_class_ids,
    )
