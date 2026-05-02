"""Continual-learning session pipeline for evolving attack sets.

This module implements a session-wise experimental loop:

1. load one session at a time
2. split that session into train/test
3. expand the label space when new classes appear
4. warm-start the encoder/head from the previous session
5. retain a replay buffer of exemplars from old classes
6. retrain and evaluate per session

The goal is to support old-class retention and new-class introduction without
replacing the single-session training path.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional runtime dependency in some environments
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False

from .classifier import CrossAttentionHead
from .data import ce_class_weights, load_cic_ids2017_frame
from .encoder import FlowEncoder
from .index import FlowIndex
from .pipeline import RAGNIDS
from .train import build_index, train_encoder, train_head


@dataclass
class SessionSpec:
    name: str
    csv_dir: str
    subsample: Optional[int] = None


@dataclass
class SessionResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    train_rows: int
    test_rows: int
    num_classes: int
    per_class: pd.DataFrame
    label_names: list[str]


@dataclass
class FeaturePreprocessor:
    feature_names: list[str]
    scaler: object

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        X_aligned = X_df.reindex(columns=self.feature_names, fill_value=0.0)
        return self.scaler.transform(X_aligned.values).astype(np.float32)


class LabelSpace:
    """Append-only label map for continual sessions."""

    def __init__(self):
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: list[str] = []

    def add_many(self, labels: np.ndarray | list[str]) -> None:
        for label in labels:
            if label not in self.label_to_id:
                self.label_to_id[label] = len(self.id_to_label)
                self.id_to_label.append(label)

    def encode(self, labels: np.ndarray | list[str]) -> np.ndarray:
        self.add_many(labels)
        return np.asarray([self.label_to_id[label] for label in labels], dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.id_to_label)


class ReplayBuffer:
    """Retained exemplars from old sessions."""

    def __init__(self):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def append(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        if self.X is None:
            self.X = X.astype(np.float32, copy=True)
            self.y = y.astype(np.int64, copy=True)
            return
        self.X = np.concatenate([self.X, X.astype(np.float32, copy=False)], axis=0)
        self.y = np.concatenate([self.y, y.astype(np.int64, copy=False)], axis=0)

    def cap_per_class(self, max_per_class: int, seed: int) -> None:
        if self.X is None or self.y is None or self.y.size == 0:
            return
        if max_per_class <= 0:
            self.X = None
            self.y = None
            return
        rng = np.random.default_rng(seed)
        keep: list[int] = []
        for cls in np.unique(self.y):
            cls_idx = np.where(self.y == cls)[0]
            if len(cls_idx) <= max_per_class:
                keep.extend(cls_idx.tolist())
            else:
                keep.extend(rng.choice(cls_idx, size=max_per_class, replace=False).tolist())
        keep_idx = np.asarray(sorted(keep), dtype=np.int64)
        self.X = self.X[keep_idx]
        self.y = self.y[keep_idx]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.y is None:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        return self.X, self.y


def _load_manifest(path: str | Path) -> list[SessionSpec]:
    payload = json.loads(Path(path).read_text())
    sessions = payload["sessions"] if isinstance(payload, dict) else payload
    out: list[SessionSpec] = []
    for i, entry in enumerate(sessions):
        out.append(SessionSpec(
            name=entry.get("name", f"session_{i+1}"),
            csv_dir=entry["csv_dir"],
            subsample=entry.get("subsample"),
        ))
    return out


def _fit_preprocessor(csv_dir: str, subsample: Optional[int], seed: int) -> tuple[FeaturePreprocessor, np.ndarray, np.ndarray]:
    X_df, y_raw, feature_names = load_cic_ids2017_frame(csv_dir, subsample=subsample, seed=seed)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_df.values)
    return FeaturePreprocessor(feature_names=feature_names, scaler=scaler), X_df, y_raw


def _load_with_preprocessor(
    csv_dir: str,
    preprocessor: FeaturePreprocessor,
    subsample: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_df, y_raw, _ = load_cic_ids2017_frame(csv_dir, subsample=subsample, seed=seed)
    return preprocessor.transform(X_df), y_raw


def _safe_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:
        # Tiny session slices can fail stratification when one class is too small.
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def _expand_linear(layer: nn.Linear, new_out_features: int) -> nn.Linear:
    if layer.out_features == new_out_features:
        return copy.deepcopy(layer)
    new_layer = nn.Linear(layer.in_features, new_out_features, bias=layer.bias is not None)
    with torch.no_grad():
        new_layer.weight.zero_()
        new_layer.weight[: layer.out_features].copy_(layer.weight)
        if layer.bias is not None:
            new_layer.bias.zero_()
            new_layer.bias[: layer.out_features].copy_(layer.bias)
    return new_layer


def _expand_encoder_aux(encoder: FlowEncoder, aux: nn.Linear, new_num_classes: int) -> nn.Linear:
    expanded = _expand_linear(aux, new_num_classes)
    return expanded.to(next(encoder.parameters()).device)


def _expand_head(head: CrossAttentionHead, new_num_classes: int) -> CrossAttentionHead:
    old_num_classes = head.cls.out_features
    if old_num_classes == new_num_classes:
        return copy.deepcopy(head)
    embed_dim = head.label_embed.embedding_dim
    n_heads = head.decoder.layers[0].self_attn.num_heads
    ffn_dim = head.decoder.layers[0].linear1.out_features
    new_head = CrossAttentionHead(embed_dim, new_num_classes, n_heads=n_heads, ffn_dim=ffn_dim)
    new_head.neighbor_proj.load_state_dict(head.neighbor_proj.state_dict())
    new_head.decoder.load_state_dict(head.decoder.state_dict())
    with torch.no_grad():
        new_head.label_embed.weight.zero_()
        new_head.label_embed.weight[: old_num_classes].copy_(head.label_embed.weight)
        new_head.cls.weight.zero_()
        new_head.cls.weight[: old_num_classes].copy_(head.cls.weight)
        if head.cls.bias is not None:
            new_head.cls.bias.zero_()
            new_head.cls.bias[: old_num_classes].copy_(head.cls.bias)
    return new_head


@torch.no_grad()
def _evaluate_session(
    model: RAGNIDS,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[dict, pd.DataFrame]:
    model.eval().to(device)
    preds: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits, *_ = model(xb, exclude_self=False)
        preds.append(logits.argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)

    acc = accuracy_score(y, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=np.arange(len(label_names)), zero_division=0
    )
    per_class = pd.DataFrame({
        "label": label_names,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
    })
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "preds": y_pred,
    }
    return metrics, per_class


def _train_session_model(
    encoder: Optional[FlowEncoder],
    head: Optional[CrossAttentionHead],
    train_X: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
    embed_dim: int,
    k: int,
    device: str,
    enc_epochs: int,
    head_epochs: int,
    enc_lr: float,
    head_lr: float,
    n_heads: int,
    supcon_weight: float,
    ce_weight: float,
    temperature: float,
    loss_name: str,
    focal_gamma: float,
    ce_class_weights: Optional[torch.Tensor],
    enc_patience: Optional[int],
    head_patience: Optional[int],
    seed: int,
) -> tuple[FlowEncoder, CrossAttentionHead, FlowIndex, RAGNIDS]:
    init_encoder = encoder
    init_aux = None
    if encoder is not None and head is not None:
        # Warm-start the auxiliary classifier with the previous class space.
        init_aux = nn.Linear(encoder.embed_dim, num_classes, bias=True).to(device)
        with torch.no_grad():
            init_aux.weight.zero_()
            init_aux.bias.zero_()
            old_n = min(head.cls.out_features, num_classes)
            init_aux.weight[: old_n].copy_(head.cls.weight[: old_n])
            if head.cls.bias is not None:
                init_aux.bias[: old_n].copy_(head.cls.bias[: old_n])

    trained_encoder = train_encoder(
        train_X, train_y, num_classes=num_classes,
        embed_dim=embed_dim, epochs=enc_epochs, lr=enc_lr,
        supcon_weight=supcon_weight, ce_weight=ce_weight,
        temperature=temperature, device=device,
        ce_class_weights=ce_class_weights, init_encoder=init_encoder,
        init_aux_head=init_aux, patience=enc_patience, val_frac=0.0, seed=seed,
    )

    index = build_index(
        trained_encoder, train_X, train_y,
        use_hnsw=False, device=device, faiss_device="cpu",
    )

    init_head = _expand_head(head, num_classes) if head is not None else None
    trained_head = train_head(
        trained_encoder, index, train_X, train_y, num_classes=num_classes,
        k=k, n_heads=n_heads, epochs=head_epochs, lr=head_lr, device=device,
        val=None, ce_class_weights=ce_class_weights, loss_name=loss_name,
        focal_gamma=focal_gamma, patience=head_patience, init_head=init_head,
    )

    model = RAGNIDS(trained_encoder, trained_head, index, k=k).to(device)
    return trained_encoder, trained_head, index, model


def run_continual_sessions(
    manifest_path: str | Path,
    device: str = "cpu",
    test_size: float = 0.2,
    embed_dim: int = 64,
    k: int = 10,
    enc_epochs: int = 10,
    head_epochs: int = 5,
    enc_lr: float = 1e-3,
    head_lr: float = 1e-3,
    supcon_weight: float = 1.0,
    ce_weight: float = 0.3,
    temperature: float = 0.1,
    n_heads: int = 4,
    loss_name: str = "ce",
    focal_gamma: float = 2.0,
    replay_per_class: int = 50,
    enc_patience: Optional[int] = None,
    head_patience: Optional[int] = None,
    seed: int = 0,
    output_dir: Optional[str | Path] = None,
) -> list[SessionResult]:
    sessions = _load_manifest(manifest_path)
    if not sessions:
        raise ValueError("Session manifest did not contain any sessions")

    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    label_space = LabelSpace()
    replay = ReplayBuffer()
    preprocessor: Optional[FeaturePreprocessor] = None
    encoder: Optional[FlowEncoder] = None
    head: Optional[CrossAttentionHead] = None
    history_tests: list[tuple[str, np.ndarray, np.ndarray]] = []
    results: list[SessionResult] = []

    for session_idx, session in enumerate(sessions):
        print(f"[session] loading {session.name} from {session.csv_dir}")
        if preprocessor is None:
            preprocessor, X_df, y_raw = _fit_preprocessor(session.csv_dir, session.subsample, seed)
            X = preprocessor.transform(X_df)
        else:
            X, y_raw = _load_with_preprocessor(session.csv_dir, preprocessor, session.subsample, seed)
        label_space.add_many(y_raw)
        y = label_space.encode(y_raw)

        X_tr, X_te, y_tr, y_te = _safe_split(X, y, test_size=test_size, seed=seed + session_idx)
        replay_X, replay_y = replay.as_arrays()
        if replay_y.size > 0:
            train_X = np.concatenate([X_tr, replay_X], axis=0)
            train_y = np.concatenate([y_tr, replay_y], axis=0)
        else:
            train_X, train_y = X_tr, y_tr

        num_classes = label_space.num_classes
        ce_w = ce_class_weights(train_y, num_classes=num_classes)
        print(f"[session] {session.name}: train={len(train_X)} test={len(X_te)} classes={num_classes}")

        encoder, head, index, model = _train_session_model(
            encoder, head, train_X, train_y, num_classes=num_classes, embed_dim=embed_dim,
            k=k, device=device, enc_epochs=enc_epochs, head_epochs=head_epochs,
            enc_lr=enc_lr, head_lr=head_lr, n_heads=n_heads, supcon_weight=supcon_weight,
            ce_weight=ce_weight, temperature=temperature, loss_name=loss_name,
            focal_gamma=focal_gamma, ce_class_weights=ce_w,
            enc_patience=enc_patience, head_patience=head_patience, seed=seed + session_idx,
        )

        metrics, per_class = _evaluate_session(model, X_te, y_te, label_space.id_to_label, device=device)
        result = SessionResult(
            name=session.name,
            accuracy=metrics["accuracy"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_macro=metrics["f1_macro"],
            train_rows=len(train_X),
            test_rows=len(X_te),
            num_classes=num_classes,
            per_class=per_class,
            label_names=label_space.id_to_label.copy(),
        )
        results.append(result)

        print(
            f"[session] {session.name}: accuracy={result.accuracy:.4f} "
            f"precision_macro={result.precision_macro:.4f} "
            f"recall_macro={result.recall_macro:.4f} f1_macro={result.f1_macro:.4f}"
        )
        print(per_class.to_string(index=False))

        if _HAS_MLFLOW and mlflow.active_run() is not None:  # pragma: no cover
            mlflow.log_metrics({
                f"session/{session.name}/accuracy": result.accuracy,
                f"session/{session.name}/precision_macro": result.precision_macro,
                f"session/{session.name}/recall_macro": result.recall_macro,
                f"session/{session.name}/f1_macro": result.f1_macro,
            })

        if out_root is not None:
            session_dir = out_root / session.name
            session_dir.mkdir(parents=True, exist_ok=True)
            per_class.to_csv(session_dir / "per_class_metrics.csv", index=False)
            pd.DataFrame([{
                "session": session.name,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "num_classes": result.num_classes,
                "accuracy": result.accuracy,
                "precision_macro": result.precision_macro,
                "recall_macro": result.recall_macro,
                "f1_macro": result.f1_macro,
            }]).to_csv(session_dir / "summary.csv", index=False)

        if history_tests:
            print(f"[session] retention check after {session.name}")
            rows = []
            for past_name, past_X, past_y in history_tests:
                past_metrics, _ = _evaluate_session(
                    model, past_X, past_y, label_space.id_to_label, device=device
                )
                print(f"  prior {past_name}: acc={past_metrics['accuracy']:.4f} "
                      f"f1={past_metrics['f1_macro']:.4f}")
                rows.append({
                    "session": session.name,
                    "prior_session": past_name,
                    "accuracy": past_metrics["accuracy"],
                    "f1_macro": past_metrics["f1_macro"],
                })
            if out_root is not None and rows:
                pd.DataFrame(rows).to_csv(session_dir / "retention.csv", index=False)

        # Retain a balanced replay buffer from the current session for the next one.
        replay.append(X_tr, y_tr)
        replay.cap_per_class(replay_per_class, seed=seed + session_idx)

        # Keep a copy of the current session test set for optional future evaluation hooks.
        history_tests.append((session.name, X_te, y_te))

    if out_root is not None:
        summary = pd.DataFrame([{
            "session": r.name,
            "train_rows": r.train_rows,
            "test_rows": r.test_rows,
            "num_classes": r.num_classes,
            "accuracy": r.accuracy,
            "precision_macro": r.precision_macro,
            "recall_macro": r.recall_macro,
            "f1_macro": r.f1_macro,
        } for r in results])
        summary.to_csv(out_root / "session_summary.csv", index=False)

    return results
