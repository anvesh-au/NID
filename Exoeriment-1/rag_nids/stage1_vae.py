"""Stage 1: VAE-based BENIGN anomaly detector.

Train on BENIGN flows only. At inference, samples with high reconstruction +
KL loss are anomalous (attack candidates) and forwarded to Stage 2.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:                                    # pragma: no cover
    _HAS_MLFLOW = False


def _active_run() -> bool:
    return _HAS_MLFLOW and mlflow.active_run() is not None


class FlowVAE(nn.Module):
    """Symmetric VAE for tabular flow features.

    encoder: input_dim → hidden → hidden → 2 × latent_dim (μ, logσ²)
    decoder: latent_dim → hidden → hidden → input_dim
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden = hidden

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_mu = nn.Linear(hidden, latent_dim)
        self.head_logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),     nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.head_mu(h), self.head_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Per-sample ELBO-style score: recon MSE + β·KL. Higher = more anomalous."""
        self.eval()
        recon, mu, logvar = self.forward(x)
        recon_err = ((recon - x) ** 2).mean(dim=-1)                                 # (B,)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)             # (B,)
        return recon_err + beta * kl / x.size(-1)


def _vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
              logvar: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_err = F.mse_loss(recon, x, reduction="none").sum(dim=-1).mean()
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon_err + beta * kl, recon_err, kl


def train_vae(
    X_benign: np.ndarray,
    input_dim: int,
    latent_dim: int = 16, hidden: int = 128,
    epochs: int = 30, batch_size: int = 512, lr: float = 1e-3, beta: float = 1.0,
    device: str = "cpu",
    val_frac: float = 0.1, patience: Optional[int] = None, min_delta: float = 1e-4,
    seed: int = 0,
) -> FlowVAE:
    """Train FlowVAE on BENIGN only. Optional held-out val split for early stopping.

    Early stopping on val ELBO (lower is better). Restores best weights.
    """
    import copy

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_benign))
    if patience is not None and val_frac > 0:
        n_val = max(1, int(len(perm) * val_frac))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        X_val = X_benign[val_idx]
    else:
        fit_idx = perm
        X_val = None

    X_fit = X_benign[fit_idx]
    vae = FlowVAE(input_dim=input_dim, latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_fit)),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(epochs):
        vae.train()
        t_loss = t_rec = t_kl = 0.0
        n = 0
        for (xb,) in tqdm(loader, desc=f"vae ep{ep+1}/{epochs}", leave=False):
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss, rec, kl = _vae_loss(recon, xb, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = xb.size(0)
            t_loss += loss.item() * bs; t_rec += rec.item() * bs; t_kl += kl.item() * bs
            n += bs

        loss_avg = t_loss / max(n, 1)
        rec_avg = t_rec / max(n, 1)
        kl_avg = t_kl / max(n, 1)

        val_loss = float("nan")
        if X_val is not None:
            val_loss = _eval_vae_loss(vae, X_val, beta, batch_size, device)

        msg = f"[vae] ep {ep+1} loss={loss_avg:.4f} recon={rec_avg:.4f} kl={kl_avg:.4f}"
        if X_val is not None:
            msg += f" | val_loss={val_loss:.4f}"
        print(msg)

        if _active_run():
            metrics = {"vae/loss": loss_avg, "vae/recon": rec_avg, "vae/kl": kl_avg}
            if X_val is not None:
                metrics["vae/val_loss"] = val_loss
            mlflow.log_metrics(metrics, step=ep)

        if patience is not None and X_val is not None:
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = copy.deepcopy(vae.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[vae] early stop @ ep {ep+1}  best val_loss={best_val:.4f}")
                    break

    if best_state is not None:
        vae.load_state_dict(best_state)
    return vae


@torch.no_grad()
def _eval_vae_loss(vae: FlowVAE, X: np.ndarray, beta: float,
                   batch_size: int, device: str) -> float:
    vae.eval()
    total = 0.0
    n = 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        recon, mu, logvar = vae(xb)
        loss, *_ = _vae_loss(recon, xb, mu, logvar, beta)
        total += loss.item() * xb.size(0); n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def compute_anomaly_scores(vae: FlowVAE, X: np.ndarray, beta: float = 1.0,
                           batch_size: int = 1024, device: str = "cpu") -> np.ndarray:
    """Per-sample anomaly score across X. Higher = more anomalous."""
    vae.eval().to(device)
    scores = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        scores.append(vae.anomaly_score(xb, beta=beta).cpu().numpy())
    return np.concatenate(scores)


def calibrate_threshold(
    vae: FlowVAE, X_val: np.ndarray, y_val: np.ndarray, benign_label: int,
    target_recall: float = 0.95, beta: float = 1.0, device: str = "cpu",
) -> Tuple[float, dict]:
    """Pick θ on a labelled mixed val slice that achieves `target_recall` on attacks.

    Strategy: rank all val samples by anomaly score descending; choose θ s.t. the
    top-fraction captures >=target_recall of true attacks. Returns (θ, metrics).
    """
    scores = compute_anomaly_scores(vae, X_val, beta=beta, device=device)
    is_attack = (y_val != benign_label)
    n_attacks = int(is_attack.sum())
    if n_attacks == 0:
        raise ValueError("Val set has zero attack samples — cannot calibrate threshold.")

    # Sort attack scores descending; pick the score at the target_recall percentile
    attack_scores = np.sort(scores[is_attack])[::-1]
    k = max(1, int(np.ceil(target_recall * n_attacks)))
    threshold = float(attack_scores[k - 1])

    pred_attack = scores >= threshold
    tp = int((pred_attack & is_attack).sum())
    fp = int((pred_attack & ~is_attack).sum())
    fn = int((~pred_attack & is_attack).sum())
    tn = int((~pred_attack & ~is_attack).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "stage1_threshold": threshold,
        "stage1_target_recall": target_recall,
        "stage1_precision": precision,
        "stage1_recall": recall,
        "stage1_f1": f1,
        "stage1_tp": tp, "stage1_fp": fp, "stage1_fn": fn, "stage1_tn": tn,
    }
    print(f"[stage1] θ={threshold:.4f}  prec={precision:.4f}  rec={recall:.4f}  f1={f1:.4f}  "
          f"(tp={tp} fp={fp} fn={fn} tn={tn})")
    return threshold, metrics
