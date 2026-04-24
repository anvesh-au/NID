"""Inference, evaluation, and retrieval-explanation printer."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from .data import CICDataset
from .pipeline import RAGNIDS


@torch.no_grad()
def evaluate(model: RAGNIDS, X: np.ndarray, y: np.ndarray,
             label_names: list[str], batch_size: int = 512, device: str = "cpu") -> dict:
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
    return {"macro_f1": macro_f1, "preds": preds, "trues": trues}


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
