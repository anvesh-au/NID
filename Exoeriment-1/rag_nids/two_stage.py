"""Two-stage NIDS: VAE BENIGN-filter (Stage 1) → RAG attack-classifier (Stage 2).

Stage 1 (FlowVAE) is trained on BENIGN flows only; samples whose anomaly score
exceeds a calibrated threshold are forwarded to Stage 2. Stage 2 is a RAGNIDS
trained on attack flows only — its label space excludes BENIGN, so the FAISS
index is never polluted by the majority class.

End-to-end prediction: VAE flags → if BENIGN-side, predict BENIGN; otherwise
defer to Stage 2 and re-map the attack-only label back into the original
N-class label space.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .pipeline import RAGNIDS, Prediction
from .stage1_vae import FlowVAE, compute_anomaly_scores


class TwoStageNIDS(nn.Module):
    """Compose FlowVAE + RAGNIDS into a single inference module.

    Args:
        vae:                Stage 1 BENIGN density model.
        threshold:          Anomaly-score cutoff (≥ → forward to Stage 2).
        stage2:             Stage 2 RAGNIDS over attack classes only.
        benign_label:       Original integer label of BENIGN in the full label space.
        new_to_orig:        Mapping from Stage 2 attack-only labels (0..N-2) back
                            to the original integer label space (skipping BENIGN).
        beta:               β coefficient for the VAE anomaly score.
    """

    def __init__(
        self,
        vae: FlowVAE,
        threshold: float,
        stage2: RAGNIDS,
        benign_label: int,
        new_to_orig: dict[int, int],
        beta: float = 1.0,
        stage2_reject_threshold: float = 0.0,
    ):
        super().__init__()
        self.vae = vae
        self.threshold = float(threshold)
        self.stage2 = stage2
        self.benign_label = int(benign_label)
        # Store as numpy lookup for fast mapping at predict-time
        max_new = max(new_to_orig.keys()) if new_to_orig else -1
        lut = np.full(max_new + 1, -1, dtype=np.int64)
        for k, v in new_to_orig.items():
            lut[k] = v
        self.register_buffer("_lut", torch.from_numpy(lut), persistent=False)
        self.beta = float(beta)
        # If stage 2 is uncertain, fall back to BENIGN instead of forcing an attack label.
        self.stage2_reject_threshold = float(stage2_reject_threshold)

    @torch.no_grad()
    def stage1_scores(self, X: np.ndarray, batch_size: int = 1024,
                      device: str = "cpu") -> np.ndarray:
        return compute_anomaly_scores(self.vae, X, beta=self.beta,
                                      batch_size=batch_size, device=device)

    @torch.no_grad()
    def stage2_confidences(self, X: np.ndarray, batch_size: int = 512,
                           device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """Return Stage-2 predicted attack labels and confidence for each row."""
        if len(X) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        self.stage2.eval().to(device)
        preds, confs = [], []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            logits, *_ = self.stage2(xb, exclude_self=False)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
        return np.concatenate(preds), np.concatenate(confs)

    @torch.no_grad()
    def predict_array(self, X: np.ndarray, batch_size: int = 512,
                      device: str = "cpu") -> dict:
        """Run end-to-end inference. Returns dict with:
        - preds:           (N,) int labels in the ORIGINAL label space
        - stage1_scores:   (N,) anomaly scores
        - stage1_attack:   (N,) bool — flagged as attack by Stage 1
        - stage2_preds:    (M,) Stage-2 predictions (attack label-space) for M flagged samples
        - stage2_idx:      (M,) row indices in X that were forwarded to Stage 2
        - stage2_conf:     (M,) Stage-2 confidences for M flagged samples
        - stage2_rejected:  number of Stage-2 candidates rejected back to BENIGN
        """
        scores = self.stage1_scores(X, batch_size=batch_size, device=device)
        flagged = scores >= self.threshold

        preds = np.full(len(X), self.benign_label, dtype=np.int64)
        stage2_preds = np.empty(0, dtype=np.int64)
        stage2_conf = np.empty(0, dtype=np.float32)
        stage2_rejected = 0
        flagged_idx = np.where(flagged)[0]

        if flagged_idx.size > 0:
            X_flag = X[flagged_idx]
            stage2_preds, stage2_conf = self.stage2_confidences(
                X_flag, batch_size=batch_size, device=device,
            )
            accepted = stage2_conf >= self.stage2_reject_threshold
            stage2_rejected = int((~accepted).sum())
            # Remap attack-only labels back to the original label space.
            lut = self._lut.cpu().numpy()
            if accepted.any():
                preds[flagged_idx[accepted]] = lut[stage2_preds[accepted]]

        return {
            "preds": preds,
            "stage1_scores": scores,
            "stage1_attack": flagged,
            "stage2_preds": stage2_preds,
            "stage2_conf": stage2_conf,
            "stage2_rejected": stage2_rejected,
            "stage2_idx": flagged_idx,
        }


@torch.no_grad()
def calibrate_stage2_reject_threshold(
    vae: FlowVAE,
    stage2: RAGNIDS,
    X_val: np.ndarray,
    y_val: np.ndarray,
    benign_label: int,
    new_to_orig: dict[int, int],
    stage1_threshold: float,
    beta: float = 1.0,
    device: str = "cpu",
    n_grid: int = 200,
) -> tuple[float, dict]:
    """Choose a Stage-2 confidence floor that can reject BENIGN false positives.

    The threshold is swept over quantiles of the forwarded Stage-2 confidence
    distribution and selected by end-to-end macro-F1 on the calibration slice.
    """
    scores = compute_anomaly_scores(vae, X_val, beta=beta, device=device)
    stage1_attack = scores >= stage1_threshold
    flagged_idx = np.where(stage1_attack)[0]

    if flagged_idx.size == 0:
        return 0.0, {
            "stage2_reject_threshold": 0.0,
            "stage2_reject_calibration_macro_f1": float("nan"),
            "stage2_reject_flagged": 0,
        }

    X_flag = X_val[flagged_idx]
    stage2.eval().to(device)
    preds, confs = [], []
    for i in range(0, len(X_flag), 512):
        xb = torch.from_numpy(X_flag[i:i + 512]).to(device)
        logits, *_ = stage2(xb, exclude_self=False)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    stage2_preds = np.concatenate(preds)
    stage2_conf = np.concatenate(confs)

    candidate_thresholds = np.unique(np.quantile(stage2_conf, np.linspace(0.0, 1.0, n_grid)))
    lut = np.full(max(new_to_orig.keys()) + 1 if new_to_orig else 0, -1, dtype=np.int64)
    for k, v in new_to_orig.items():
        lut[k] = v

    best_f1, best_t = -1.0, 0.0
    for t in candidate_thresholds:
        pred = np.full(len(X_val), benign_label, dtype=np.int64)
        keep = stage2_conf >= t
        if keep.any():
            pred[flagged_idx[keep]] = lut[stage2_preds[keep]]
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    pred = np.full(len(X_val), benign_label, dtype=np.int64)
    keep = stage2_conf >= best_t
    if keep.any():
        pred[flagged_idx[keep]] = lut[stage2_preds[keep]]

    metrics = {
        "stage2_reject_threshold": best_t,
        "stage2_reject_calibration_macro_f1": best_f1,
        "stage2_reject_flagged": int(flagged_idx.size),
        "stage2_reject_kept": int(keep.sum()),
    }
    return best_t, metrics
