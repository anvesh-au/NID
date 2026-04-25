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

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

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

    @torch.no_grad()
    def stage1_scores(self, X: np.ndarray, batch_size: int = 1024,
                      device: str = "cpu") -> np.ndarray:
        return compute_anomaly_scores(self.vae, X, beta=self.beta,
                                      batch_size=batch_size, device=device)

    @torch.no_grad()
    def predict_array(self, X: np.ndarray, batch_size: int = 512,
                      device: str = "cpu") -> dict:
        """Run end-to-end inference. Returns dict with:
        - preds:           (N,) int labels in the ORIGINAL label space
        - stage1_scores:   (N,) anomaly scores
        - stage1_attack:   (N,) bool — flagged as attack by Stage 1
        - stage2_preds:    (M,) Stage-2 predictions (attack label-space) for M flagged samples
        - stage2_idx:      (M,) row indices in X that were forwarded to Stage 2
        """
        scores = self.stage1_scores(X, batch_size=batch_size, device=device)
        flagged = scores >= self.threshold

        preds = np.full(len(X), self.benign_label, dtype=np.int64)
        stage2_preds = np.empty(0, dtype=np.int64)
        flagged_idx = np.where(flagged)[0]

        if flagged_idx.size > 0:
            X_flag = X[flagged_idx]
            self.stage2.eval().to(device)
            chunks = []
            for i in range(0, len(X_flag), batch_size):
                xb = torch.from_numpy(X_flag[i:i + batch_size]).to(device)
                logits, *_ = self.stage2(xb, exclude_self=False)
                chunks.append(logits.argmax(-1).cpu().numpy())
            stage2_preds = np.concatenate(chunks)
            # Remap attack-only labels back to the original label space.
            lut = self._lut.cpu().numpy()
            preds[flagged_idx] = lut[stage2_preds]

        return {
            "preds": preds,
            "stage1_scores": scores,
            "stage1_attack": flagged,
            "stage2_preds": stage2_preds,
            "stage2_idx": flagged_idx,
        }
