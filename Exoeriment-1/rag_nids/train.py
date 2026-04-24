"""Training loops: (1) encoder with SupCon + CE, (2) cross-attention head over frozen encoder.

MLflow is called when an active run is present; otherwise these functions run untouched.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# pytorch-metric-learning: stable, well-tested contrastive losses — no reason to reimplement SupCon.
from pytorch_metric_learning import losses
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:                                    # pragma: no cover
    _HAS_MLFLOW = False

from .classifier import CrossAttentionHead
from .data import CICDataset, class_weights
from .encoder import EncoderWithAuxHead, FlowEncoder
from .index import FlowIndex
from .pipeline import RAGNIDS


def _active_run() -> bool:
    return _HAS_MLFLOW and mlflow.active_run() is not None


def _sampler(y: np.ndarray) -> WeightedRandomSampler:
    w = class_weights(y)
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


def pretrain_scarf(
    X: np.ndarray, encoder: FlowEncoder,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3,
    corruption_rate: float = 0.6, temperature: float = 0.5,
    device: str = "cpu",
) -> FlowEncoder:
    """SCARF self-supervised pretraining (Bahri et al., ICLR 2022).

    Create two views of each flow by corrupting a random p% of features with
    draws from each feature's marginal empirical distribution. Train the encoder
    + a lightweight projection head with NT-Xent so both views align in latent space.
    The projection head is discarded after pretraining.
    """
    encoder.train().to(device)
    projector = nn.Sequential(
        nn.Linear(encoder.embed_dim, encoder.embed_dim), nn.ReLU(),
        nn.Linear(encoder.embed_dim, encoder.embed_dim),
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()),
                            lr=lr, weight_decay=1e-4)

    X_t = torch.from_numpy(X).to(device)         # full train set on device: cheap marginal sampling
    N, D = X_t.shape
    steps = max(1, N // batch_size)
    col_idx = torch.arange(D, device=device)

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total, n = 0.0, 0
        for step in tqdm(range(steps), desc=f"scarf ep{ep+1}/{epochs}", leave=False):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            x = X_t[idx]                                                         # (B, D)

            # Corrupt p% of features with draws from each feature's marginal distribution
            mask = torch.rand(x.size(0), D, device=device) < corruption_rate     # (B, D) bool
            rand_rows = torch.randint(0, N, (x.size(0), D), device=device)       # (B, D)
            corrupt = X_t[rand_rows, col_idx.expand_as(rand_rows)]               # (B, D)
            x_corr = torch.where(mask, corrupt, x)

            z1 = projector(encoder(x))
            z2 = projector(encoder(x_corr))
            z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)

            # NT-Xent: 2B samples, positives are (i, i+B)
            z = torch.cat([z1, z2], dim=0)                                       # (2B, d)
            sim = (z @ z.T) / temperature
            sim.fill_diagonal_(-1e9)
            B = x.size(0)
            targets = torch.cat([torch.arange(B, 2 * B, device=device),
                                 torch.arange(0, B, device=device)])
            loss = F.cross_entropy(sim, targets)

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * B; n += B

        loss_avg = total / max(n, 1)
        print(f"[scarf] ep {ep+1} ntxent_loss={loss_avg:.4f}")
        if _active_run():
            mlflow.log_metrics({"scarf/ntxent_loss": loss_avg}, step=ep)

    return encoder


def train_encoder(
    X: np.ndarray, y: np.ndarray, num_classes: int,
    embed_dim: int = 64, epochs: int = 20, batch_size: int = 512,
    lr: float = 1e-3, supcon_weight: float = 1.0, ce_weight: float = 0.3,
    temperature: float = 0.1, device: str = "cpu",
    ce_class_weights: Optional[torch.Tensor] = None,
    init_encoder: Optional[FlowEncoder] = None,
) -> FlowEncoder:
    # FIX: accept a pre-initialized encoder (e.g. SCARF-pretrained) instead of always starting fresh
    encoder = init_encoder if init_encoder is not None else FlowEncoder(input_dim=X.shape[1], embed_dim=embed_dim)
    encoder = encoder.to(device)
    model = EncoderWithAuxHead(encoder, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    supcon = losses.SupConLoss(temperature=temperature)

    cw = ce_class_weights.to(device) if ce_class_weights is not None else None
    loader = DataLoader(CICDataset(X, y), batch_size=batch_size,
                        sampler=_sampler(y), num_workers=0, drop_last=True)

    model.train()
    for ep in range(epochs):
        t_sup, t_ce, t_tot, n = 0.0, 0.0, 0.0, 0
        for xb, yb in tqdm(loader, desc=f"encoder ep{ep+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            z, logits = model(xb)
            l_sup = supcon(z, yb)
            l_ce = F.cross_entropy(logits, yb, weight=cw)
            loss = supcon_weight * l_sup + ce_weight * l_ce
            opt.zero_grad(); loss.backward(); opt.step()
            bs = xb.size(0)
            t_sup += l_sup.item() * bs; t_ce += l_ce.item() * bs
            t_tot += loss.item() * bs; n += bs
        sched.step()

        sup_avg, ce_avg, tot_avg = t_sup / n, t_ce / n, t_tot / n
        print(f"[encoder] ep {ep+1} supcon={sup_avg:.4f} ce={ce_avg:.4f} total={tot_avg:.4f}")
        if _active_run():
            mlflow.log_metrics({
                "encoder/supcon_loss": sup_avg,
                "encoder/ce_loss": ce_avg,
                "encoder/total_loss": tot_avg,
                "encoder/lr": sched.get_last_lr()[0],
            }, step=ep)

    return encoder


@torch.no_grad()
def build_index(encoder: FlowEncoder, X: np.ndarray, y: np.ndarray,
                use_hnsw: bool = False, batch_size: int = 4096, device: str = "cpu") -> FlowIndex:
    encoder.eval().to(device)
    embs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        embs.append(encoder(xb).cpu().numpy())
    embs = np.concatenate(embs, axis=0).astype(np.float32)

    index = FlowIndex(embed_dim=encoder.embed_dim, use_hnsw=use_hnsw)
    index.add(embs, y)
    return index


@torch.no_grad()
def _val_macro_f1(model: RAGNIDS, X: np.ndarray, y: np.ndarray,
                  batch_size: int = 512, device: str = "cpu") -> float:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits, *_ = model(xb, exclude_self=False)
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return float(f1_score(y, preds, average="macro", zero_division=0))


def train_head(
    encoder: FlowEncoder, index: FlowIndex, X: np.ndarray, y: np.ndarray,
    num_classes: int, k: int = 10, n_heads: int = 4,
    epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cpu",
    val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ce_class_weights: Optional[torch.Tensor] = None,
    loss_name: str = "ce",   # "ce" | "focal"
    focal_gamma: float = 2.0,
) -> CrossAttentionHead:
    head = CrossAttentionHead(encoder.embed_dim, num_classes, n_heads=n_heads).to(device)
    model = RAGNIDS(encoder, head, index, k=k).to(device)

    # Freeze encoder during head training — simpler, avoids destabilizing the index geometry mid-training.
    for p in encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(CICDataset(X, y), batch_size=batch_size,
                        sampler=_sampler(y), num_workers=0, drop_last=True)
    cw = ce_class_weights.to(device) if ce_class_weights is not None else None

    def _loss(logits, targets):
        if loss_name == "focal":
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            ce = F.nll_loss(logp, targets, weight=cw, reduction="none")
            pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            return ((1 - pt) ** focal_gamma * ce).mean()
        return F.cross_entropy(logits, targets, weight=cw)

    head.train()
    for ep in range(epochs):
        total, correct, n = 0.0, 0, 0
        for xb, yb in tqdm(loader, desc=f"head ep{ep+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits, *_ = model(xb, exclude_self=True)
            loss = _loss(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
            correct += (logits.argmax(-1) == yb).sum().item()
            n += xb.size(0)

        train_loss, train_acc = total / n, correct / n
        val_f1 = _val_macro_f1(model, val[0], val[1], device=device) if val else float("nan")
        print(f"[head] ep {ep+1} loss={train_loss:.4f} acc={train_acc:.4f} val_macro_f1={val_f1:.4f}")
        if _active_run():
            metrics = {"head/train_loss": train_loss, "head/train_acc": train_acc}
            if val is not None:
                metrics["head/val_macro_f1"] = val_f1
            mlflow.log_metrics(metrics, step=ep)

    return head
