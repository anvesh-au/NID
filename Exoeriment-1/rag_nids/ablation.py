"""Ablation runners for continual and full-dataset experiments."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from .continual import (
    _fit_preprocessor,
    _load_manifest,
    _load_with_preprocessor,
    _safe_split,
    LabelSpace,
    ReplayBuffer,
    run_continual_sessions,
)
from .data import ce_class_weights, load_cic_ids2017
from .encoder import FlowEncoder
from .index import FlowIndex
from .pipeline import RAGNIDS
from .train import build_index, train_encoder, train_head

try:  # pragma: no cover
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False


@dataclass
class AblationRow:
    mode: str
    model: str
    session: str
    seed: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    train_seconds: float
    infer_seconds: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    return {
        "accuracy": float(acc),
        "precision_macro": float(np.mean(p[present])) if np.any(present) else 0.0,
        "recall_macro": float(np.mean(r[present])) if np.any(present) else 0.0,
        "f1_macro": float(np.mean(f1[present])) if np.any(present) else 0.0,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _write_eval_artifacts(
    base_dir: Path,
    model: str,
    session: str,
    label_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_dir = base_dir / model / session
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(label_names))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "label": label_names,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": s,
    })
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
    cm_counts = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_rates = pd.DataFrame(rates, index=label_names, columns=label_names)
    cm_counts.index.name = "true"; cm_counts.columns.name = "pred"
    cm_rates.index.name = "true"; cm_rates.columns.name = "pred"
    cm_counts.to_csv(out_dir / "confusion_matrix_counts.csv")
    cm_rates.to_csv(out_dir / "confusion_matrix_rates.csv")


@torch.no_grad()
def _encode_array(encoder: FlowEncoder, X: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    encoder.eval().to(device)
    outs: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        outs.append(encoder(xb).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def _predict_majority(
    encoder: FlowEncoder,
    index: FlowIndex,
    X: np.ndarray,
    k: int,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    encoder.eval().to(device)
    preds: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        z = encoder(xb).cpu().numpy().astype(np.float32)
        sims, _, n_labels = index.search(z, k=k)
        # Similarity-weighted vote with floor shift for negative sims.
        min_sims = np.min(sims, axis=1, keepdims=True)
        weights = sims - min_sims + 1e-6
        out = np.zeros((len(z), num_classes), dtype=np.float32)
        for row in range(len(z)):
            np.add.at(out[row], n_labels[row], weights[row])
        preds.append(np.argmax(out, axis=1))
    return np.concatenate(preds, axis=0).astype(np.int64)


def run_continual_ablation(
    manifest_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    faiss_device: str = "cpu",
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
    seed: int = 0,
    recency_alpha: float = 0.0,
    encoder_first_session_only: bool = False,
) -> pd.DataFrame:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Baseline A1 (attention head): reuse existing runner and convert summary rows.
    attn_results = run_continual_sessions(
        manifest_path=manifest_path,
        device=device,
        test_size=test_size,
        embed_dim=embed_dim,
        k=k,
        enc_epochs=enc_epochs,
        head_epochs=head_epochs,
        enc_lr=enc_lr,
        head_lr=head_lr,
        supcon_weight=supcon_weight,
        ce_weight=ce_weight,
        temperature=temperature,
        n_heads=n_heads,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        recency_alpha=recency_alpha,
        replay_per_class=replay_per_class,
        faiss_device=faiss_device,
        encoder_first_session_only=encoder_first_session_only,
        seed=seed,
        output_dir=out_root / "continual_attention",
    )
    rows: list[AblationRow] = []
    for r in attn_results:
        # Weighted F1 can be reconstructed from per-class table supports/f1.
        total = float(r.per_class["support"].sum())
        fw = float((r.per_class["f1"] * r.per_class["support"]).sum() / max(total, 1.0))
        rows.append(AblationRow(
            mode="continual", model="ann_attention", session=r.name, seed=seed,
            accuracy=r.accuracy, precision_macro=r.precision_macro, recall_macro=r.recall_macro,
            f1_macro=r.f1_macro, f1_weighted=fw, train_seconds=np.nan, infer_seconds=np.nan,
        ))
        attn_dir = out_root / "artifacts"
        (attn_dir / "ann_attention" / r.name).mkdir(parents=True, exist_ok=True)
        r.per_class.to_csv(attn_dir / "ann_attention" / r.name / "per_class_metrics.csv", index=False)
        r.confusion_rates.to_csv(attn_dir / "ann_attention" / r.name / "confusion_matrix_rates.csv")

    # A2 (majority vote) in same continual protocol.
    sessions = _load_manifest(manifest_path)
    label_space = LabelSpace()
    replay = ReplayBuffer()
    preprocessor = None
    encoder: Optional[FlowEncoder] = None
    index: Optional[FlowIndex] = None

    for session_idx, session in enumerate(sessions):
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
        do_train_encoder = (session_idx == 0) or (not encoder_first_session_only)

        t0 = time.time()
        if do_train_encoder:
            encoder = train_encoder(
                train_X, train_y, num_classes=num_classes,
                embed_dim=embed_dim, epochs=enc_epochs, lr=enc_lr,
                supcon_weight=supcon_weight, ce_weight=ce_weight,
                temperature=temperature, device=device,
                ce_class_weights=ce_w, init_encoder=encoder,
                init_aux_head=None, patience=None, val_frac=0.0, seed=seed + session_idx,
            )
        if encoder is None:
            raise ValueError("encoder was not initialized")
        if index is None or do_train_encoder:
            index = build_index(
                encoder, train_X, train_y, use_hnsw=False, device=device, faiss_device=faiss_device
            )
        else:
            append_ix = build_index(
                encoder, X_tr, y_tr, use_hnsw=False, device=device, faiss_device=faiss_device
            )
            index.add(append_ix.embeddings, append_ix.labels)
        train_secs = time.time() - t0

        t1 = time.time()
        y_pred = _predict_majority(encoder, index, X_te, k=k, num_classes=num_classes, device=device)
        infer_secs = time.time() - t1
        m = _metrics(y_te, y_pred, labels=np.arange(num_classes))
        rows.append(AblationRow(
            mode="continual", model="ann_majority_vote", session=session.name, seed=seed,
            accuracy=m["accuracy"], precision_macro=m["precision_macro"], recall_macro=m["recall_macro"],
            f1_macro=m["f1_macro"], f1_weighted=m["f1_weighted"], train_seconds=train_secs, infer_seconds=infer_secs,
        ))
        _write_eval_artifacts(
            out_root / "artifacts", "ann_majority_vote", session.name,
            label_space.id_to_label.copy(), y_te, y_pred
        )

        replay.append(X_tr, y_tr)
        replay.cap_per_class(replay_per_class, seed=seed + session_idx)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_root / "ablation_continual_summary.csv", index=False)
    return df


def run_full_ablation(
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    test_size: float = 0.2,
    subsample: Optional[int] = None,
    device: str = "cpu",
    faiss_device: str = "cpu",
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
    recency_alpha: float = 0.0,
) -> pd.DataFrame:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    X, y, _, _, label_enc = load_cic_ids2017(data_dir, subsample=subsample, seed=seed)
    label_names = list(label_enc.classes_)
    n_classes = int(np.max(y)) + 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    labels = np.arange(n_classes)
    rows: list[AblationRow] = []

    # B1: plain XGBoost
    if not _HAS_XGBOOST:
        raise ImportError("xgboost is required for full ablation; please install xgboost.")
    t0 = time.time()
    xgb_plain = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_plain.fit(X_tr, y_tr)
    tr_secs = time.time() - t0
    t1 = time.time()
    p = xgb_plain.predict(X_te)
    inf_secs = time.time() - t1
    m = _metrics(y_te, p, labels=labels)
    rows.append(AblationRow("full", "plain_xgboost", "full_dataset", seed, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["f1_weighted"], tr_secs, inf_secs))
    _write_eval_artifacts(out_root / "artifacts", "plain_xgboost", "full_dataset", label_names, y_te, p)

    # shared encoder for B2/B3/B4
    cw = ce_class_weights(y_tr, num_classes=n_classes)
    t0 = time.time()
    encoder = train_encoder(
        X_tr, y_tr, num_classes=n_classes, embed_dim=embed_dim, epochs=enc_epochs, lr=enc_lr,
        supcon_weight=supcon_weight, ce_weight=ce_weight, temperature=temperature, device=device,
        ce_class_weights=cw, patience=None, val_frac=0.0, seed=seed,
    )
    enc_secs = time.time() - t0
    Z_tr = _encode_array(encoder, X_tr, device=device)
    Z_te = _encode_array(encoder, X_te, device=device)

    # B2: encoder + linear classifier
    t0 = time.time()
    lin = LogisticRegression(max_iter=1000, n_jobs=None, multi_class="multinomial")
    lin.fit(Z_tr, y_tr)
    tr_secs = enc_secs + (time.time() - t0)
    t1 = time.time()
    p = lin.predict(Z_te)
    inf_secs = time.time() - t1
    m = _metrics(y_te, p, labels=labels)
    rows.append(AblationRow("full", "encoder_linear", "full_dataset", seed, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["f1_weighted"], tr_secs, inf_secs))
    _write_eval_artifacts(out_root / "artifacts", "encoder_linear", "full_dataset", label_names, y_te, p)

    # B3: encoder + xgboost
    t0 = time.time()
    xgb_emb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_emb.fit(Z_tr, y_tr)
    tr_secs = enc_secs + (time.time() - t0)
    t1 = time.time()
    p = xgb_emb.predict(Z_te)
    inf_secs = time.time() - t1
    m = _metrics(y_te, p, labels=labels)
    rows.append(AblationRow("full", "encoder_xgboost", "full_dataset", seed, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["f1_weighted"], tr_secs, inf_secs))
    _write_eval_artifacts(out_root / "artifacts", "encoder_xgboost", "full_dataset", label_names, y_te, p)

    # B4: encoder + retrieval + attention head
    t0 = time.time()
    index = build_index(encoder, X_tr, y_tr, use_hnsw=False, device=device, faiss_device=faiss_device)
    head = train_head(
        encoder, index, X_tr, y_tr, num_classes=n_classes, k=k, n_heads=n_heads,
        epochs=head_epochs, lr=head_lr, device=device, val=(X_te, y_te),
        ce_class_weights=cw, loss_name=loss_name, focal_gamma=focal_gamma, patience=None,
        recency_alpha=recency_alpha,
    )
    model = RAGNIDS(encoder, head, index, k=k, recency_alpha=recency_alpha).to(device)
    tr_secs = enc_secs + (time.time() - t0)
    t1 = time.time()
    pred = []
    with torch.no_grad():
        for i in range(0, len(X_te), 4096):
            xb = torch.from_numpy(X_te[i:i + 4096]).to(device)
            logits, *_ = model(xb, exclude_self=False)
            pred.append(logits.argmax(-1).cpu().numpy())
    p = np.concatenate(pred, axis=0)
    inf_secs = time.time() - t1
    m = _metrics(y_te, p, labels=labels)
    rows.append(AblationRow("full", "encoder_retrieval_attention", "full_dataset", seed, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"], m["f1_weighted"], tr_secs, inf_secs))
    _write_eval_artifacts(out_root / "artifacts", "encoder_retrieval_attention", "full_dataset", label_names, y_te, p)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_root / "ablation_full_summary.csv", index=False)
    (out_root / "ablation_config.json").write_text(json.dumps({
        "seed": seed,
        "test_size": test_size,
        "subsample": subsample,
        "embed_dim": embed_dim,
        "k": k,
    }, indent=2))
    return df
