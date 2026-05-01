from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from .data import SessionData, load_cic_ids2017_sessions
from .infer import confusion_matrix
from .lifecycle import dataset_hash, log_and_register, mark_staging, promote_if_better, save_pipeline
from .pipeline import RAGNIDS
from .train import build_index, pretrain_scarf, train_encoder, train_head
from .encoder import FlowEncoder


def _concat_sessions(sessions: list[SessionData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [session.X for session in sessions]
    ys = [session.y for session in sessions]
    session_ids = [np.full(len(session.y), idx, dtype=np.int32) for idx, session in enumerate(sessions)]
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(session_ids)


def _index_snapshot(model: RAGNIDS) -> dict[str, int]:
    stats = model.index.stats()
    return {
        "index_total": stats.total,
        "index_pinned": stats.pinned,
        "index_writeback": stats.writeback,
    }


@torch.no_grad()
def _evaluate_session_stream(
    model: RAGNIDS,
    session: SessionData,
    label_names: list[str],
    session_id: int,
    device: str,
    batch_size: int,
    benign_class_id: int | None,
    writeback_threshold: float,
    cm_out_dir: str | None = None,
) -> dict:
    model.eval().to(device)
    preds, trues = [], []
    added = 0
    attack_mask = None

    for start in range(0, len(session.y), batch_size):
        stop = min(start + batch_size, len(session.y))
        xb = torch.from_numpy(session.X[start:stop]).to(device)
        yb = session.y[start:stop]
        logits, z, _, _, _ = model(xb, exclude_self=False)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        pred_np = pred.cpu().numpy()
        conf_np = conf.cpu().numpy()
        z_np = z.detach().cpu().numpy()

        preds.append(pred_np)
        trues.append(yb)
        if attack_mask is None:
            attack_mask = np.ones(len(label_names), dtype=bool)
            if benign_class_id is not None:
                attack_mask[benign_class_id] = False
        for row_idx in range(len(pred_np)):
            if model.index.writeback(
                embedding=z_np[row_idx],
                label=int(pred_np[row_idx]),
                min_confidence=writeback_threshold,
                confidence=float(conf_np[row_idx]),
                is_attack=bool(attack_mask[pred_np[row_idx]]),
                session_id=session_id,
            ):
                added += 1

    preds_arr = np.concatenate(preds)
    trues_arr = np.concatenate(trues)
    macro_f1 = float(f1_score(trues_arr, preds_arr, average="macro", zero_division=0))
    print(classification_report(
        trues_arr, preds_arr,
        labels=range(len(label_names)), target_names=label_names,
        digits=4, zero_division=0,
    ))
    print(f"macro-F1: {macro_f1:.4f}")
    cm = confusion_matrix(trues_arr, preds_arr, label_names, out_dir=cm_out_dir)
    return {
        "macro_f1": macro_f1,
        "preds": preds_arr,
        "trues": trues_arr,
        "added": added,
        "confusion": cm,
    }


def run_session_pipeline(args, use_mlflow: bool) -> None:
    print(f"[data] loading session dataset from {args.data_dir}")
    sessions, feature_names, scaler, label_enc = load_cic_ids2017_sessions(
        args.data_dir, subsample=None if args.subsample <= 0 else args.subsample, seed=args.seed,
    )
    if len(sessions) < args.train_days + 1:
        raise ValueError(f"Need at least {args.train_days + 1} day sessions, found {len(sessions)}")

    label_names = list(label_enc.classes_)
    num_classes = len(label_names)
    for idx, session in enumerate(sessions):
        counts = np.bincount(session.y, minlength=num_classes)
        print(f"[session {idx}] {session.name}: rows={len(session.y)} class_counts={counts.tolist()}")

    train_sessions = sessions[:args.train_days]
    eval_sessions = sessions[args.train_days:]
    X_train_all, y_train_all, train_session_ids = _concat_sessions(train_sessions)
    print(f"[session] training on {[s.name for s in train_sessions]} and evaluating on {[s.name for s in eval_sessions]}")

    if use_mlflow:
        mlflow.log_params({
            "num_classes": num_classes,
            "num_features": X_train_all.shape[1],
            "dataset_hash": dataset_hash(X_train_all, y_train_all),
            "dataset_rows": int(sum(len(s.y) for s in sessions)),
            "pipeline_mode": "sessions",
        })

    cw = None
    if args.class_weighted_ce:
        counts = np.bincount(y_train_all, minlength=num_classes).astype(np.float32)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        cw = torch.from_numpy(w)

    init_encoder = None
    if args.scarf_epochs > 0:
        print("[scarf] pretraining encoder on train sessions")
        init_encoder = FlowEncoder(input_dim=X_train_all.shape[1], embed_dim=args.embed_dim)
        init_encoder = pretrain_scarf(
            X_train_all, init_encoder,
            epochs=args.scarf_epochs, lr=args.enc_lr,
            corruption_rate=args.scarf_corruption,
            temperature=args.scarf_temperature,
            device=args.device,
        )

    print("[train] encoder (session pipeline)")
    encoder = train_encoder(
        X_train_all, y_train_all, num_classes=num_classes,
        embed_dim=args.embed_dim, epochs=args.enc_epochs, lr=args.enc_lr,
        supcon_weight=args.supcon_weight, ce_weight=args.ce_weight,
        temperature=args.temperature, device=args.device,
        ce_class_weights=cw, init_encoder=init_encoder,
        patience=args.enc_patience, val_frac=args.enc_val_frac, seed=args.seed,
    )

    print("[index] building base session index")
    index = build_index(
        encoder, X_train_all, y_train_all,
        use_hnsw=args.hnsw, device=args.device, faiss_device=args.faiss_device,
        session_ids=train_session_ids,
    )

    X_head_fit, X_head_val, y_head_fit, y_head_val = train_test_split(
        X_train_all, y_train_all, test_size=args.session_val_frac, stratify=y_train_all, random_state=args.seed,
    ) if args.session_val_frac > 0 else (X_train_all, None, y_train_all, None)

    print("[train] head (session pipeline)")
    head = train_head(
        encoder, index, X_head_fit, y_head_fit, num_classes=num_classes,
        k=args.k, n_heads=args.n_heads, epochs=args.head_epochs, lr=args.head_lr,
        device=args.device, val=(X_head_val, y_head_val) if X_head_val is not None else None,
        ce_class_weights=cw, loss_name=args.loss_name, focal_gamma=args.focal_gamma,
        patience=args.head_patience,
    )
    model = RAGNIDS(encoder, head, index, k=args.k).to(args.device)

    benign_class_id = next((i for i, name in enumerate(label_names) if name.upper() == "BENIGN"), None)
    report_dir = Path(args.session_report_dir or Path("session_reports") / args.run_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    session_rows = []

    for eval_offset, session in enumerate(eval_sessions, start=args.train_days):
        min_session_id = max(0, eval_offset - args.session_window_days)
        evicted = model.index.evict_sessions_before(min_session_id)
        before = _index_snapshot(model)
        print(f"[session {eval_offset}] {session.name}: evicted={evicted} keep_sessions>={min_session_id}")
        day_dir = report_dir / f"{eval_offset:02d}_{session.name}"
        day_dir.mkdir(parents=True, exist_ok=True)
        res = _evaluate_session_stream(
            model, session, label_names,
            session_id=eval_offset,
            device=args.device,
            batch_size=args.session_eval_batch_size,
            benign_class_id=benign_class_id,
            writeback_threshold=args.session_writeback_threshold,
            cm_out_dir=str(day_dir),
        )
        after = _index_snapshot(model)
        print(
            f"[session {eval_offset}] {session.name}: macro_f1={res['macro_f1']:.4f} "
            f"added={res['added']} index_total={after['index_total']} "
            f"(pinned={after['index_pinned']} writeback={after['index_writeback']})"
        )
        session_rows.append({
            "session_id": eval_offset,
            "session_name": session.name,
            "rows": len(session.y),
            "macro_f1": res["macro_f1"],
            "evicted_before_eval": evicted,
            "added_writeback": res["added"],
            "index_total_before": before["index_total"],
            "index_total_after": after["index_total"],
            "index_pinned_before": before["index_pinned"],
            "index_pinned_after": after["index_pinned"],
            "index_writeback_before": before["index_writeback"],
            "index_writeback_after": after["index_writeback"],
            "min_session_kept": min_session_id,
        })
        if use_mlflow:
            mlflow.log_metric(f"session/{session.name}/macro_f1", res["macro_f1"])
            mlflow.log_metric(f"session/{session.name}/writeback_added", float(res["added"]))
            mlflow.log_metric(f"session/{session.name}/evicted", float(evicted))

    report_df = pd.DataFrame(session_rows)
    report_path = report_dir / "session_summary.csv"
    report_df.to_csv(report_path, index=False)
    print("\n[session report]")
    print(report_df.to_string(index=False))

    if use_mlflow:
        mlflow.log_artifact(str(report_path), artifact_path="session_eval")
        mlflow.log_artifacts(str(report_dir), artifact_path="session_eval/details")
        macro_f1 = float(report_df["macro_f1"].mean()) if not report_df.empty else float("nan")
        art_dir = report_dir / "pipeline"
        save_pipeline(art_dir, encoder, head, model.index, label_enc, scaler, feature_names, args.k, num_classes)
        uri = log_and_register(art_dir, macro_f1=macro_f1, register=True)
        print(f"[mlflow] model uri = {uri}")
        run = mlflow.active_run()
        if run is not None:
            mark_staging(run.info.run_id)
            promoted = promote_if_better(threshold=args.promote_threshold)
            if promoted:
                print(f"[mlflow] promoted version {promoted} to Production")
