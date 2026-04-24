"""RAG-NIDS end-to-end driver with MLflow tracking.

Usage:
  python main.py --data_dir path/to/CIC-IDS2017/MachineLearningCVE
"""
import argparse
import os
import random
import re
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from rag_nids import RAGNIDS
from rag_nids.data import class_weights, load_cic_ids2017
from rag_nids.infer import evaluate, explain
from rag_nids.lifecycle import (
    ensure_experiment, dataset_hash, log_and_register, mark_staging, promote_if_better,
    save_pipeline,
)
from rag_nids.train import build_index, train_encoder, train_head


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _pkg_versions() -> dict:
    pkgs = ["torch", "numpy", "pandas", "scikit-learn", "faiss-cpu",
            "pytorch-metric-learning", "mlflow"]
    out = {}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "missing"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--subsample", type=int, default=50_000)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--enc_epochs", type=int, default=10)
    ap.add_argument("--head_epochs", type=int, default=5)
    ap.add_argument("--enc_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--supcon_weight", type=float, default=1.0)
    ap.add_argument("--ce_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--loss_name", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--class_weighted_ce", action="store_true")
    ap.add_argument("--hnsw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="poc")
    ap.add_argument("--promote_threshold", type=float, default=0.80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_experiment()

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        mlflow.log_params({f"pkg.{k}": v for k, v in _pkg_versions().items()})

        print(f"[data] loading {args.data_dir}")
        X, y, feats, scaler, label_enc = load_cic_ids2017(args.data_dir, subsample=args.subsample,
                                                          seed=args.seed)
        label_names = list(label_enc.classes_)
        num_classes = len(label_names)
        print(f"[data] X={X.shape}  classes={num_classes}  -> {label_names}")
        mlflow.log_params({
            "num_classes": num_classes, "num_features": X.shape[1],
            "dataset_hash": dataset_hash(X, y), "dataset_rows": int(X.shape[0]),
        })

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=args.seed)

        # print(f"[data] X_tr={X_tr.shape}  y_tr={y_tr.shape}  X_te={X_te.shape}  y_te={y_te.shape}")

        cw = None
        if args.class_weighted_ce:
            # print("[train] class-weighted CE")
            counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
            w = counts.sum() / (num_classes * np.maximum(counts, 1))
            cw = torch.from_numpy(w)

        print("[train] encoder")
        encoder = train_encoder(
            X_tr, y_tr, num_classes=num_classes,
            embed_dim=args.embed_dim, epochs=args.enc_epochs, lr=args.enc_lr,
            supcon_weight=args.supcon_weight, ce_weight=args.ce_weight,
            temperature=args.temperature, device=args.device,
            ce_class_weights=cw,
        )

        print("[index] building")
        index = build_index(encoder, X_tr, y_tr, use_hnsw=args.hnsw, device=args.device)
        mlflow.log_params({"index_total": index.stats().total,
                           "index_type": "hnsw" if args.hnsw else "flat"})

        print("[train] cross-attention head")
        head = train_head(
            encoder, index, X_tr, y_tr, num_classes=num_classes,
            k=args.k, n_heads=args.n_heads, epochs=args.head_epochs, lr=args.head_lr,
            device=args.device, val=(X_te, y_te),
            ce_class_weights=cw, loss_name=args.loss_name, focal_gamma=args.focal_gamma,
        )

        model = RAGNIDS(encoder, head, index, k=args.k).to(args.device)

        print("[eval] test set")
        res = evaluate(model, X_te, y_te, label_names, device=args.device)
        preds, trues = res["preds"], res["trues"]
        macro_f1 = res["macro_f1"]

        # Per-class F1 + confusion matrix artifacts
        f1_per = f1_score(trues, preds, average=None, zero_division=0, labels=range(num_classes))
        # FIX: sanitize class names — CIC-IDS2017 has non-ASCII chars (e.g. Web Attack \ufffd XSS)
        # that mlflow rejects in metric keys.
        def _clean(n: str) -> str:
            return re.sub(r"[^A-Za-z0-9_\-. ]+", "_", n).strip("_")
        for i, name in enumerate(label_names):
            mlflow.log_metric(f"f1/{_clean(name)}", float(f1_per[i]))

        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd
            pd.DataFrame(confusion_matrix(trues, preds, labels=range(num_classes)),
                         index=label_names, columns=label_names).to_csv(f"{tmp}/confusion_matrix.csv")
            pd.DataFrame({"label": label_names, "f1": f1_per}).to_csv(
                f"{tmp}/per_class_f1.csv", index=False)
            pd.DataFrame({"train": np.bincount(y_tr, minlength=num_classes),
                          "test": np.bincount(y_te, minlength=num_classes)},
                         index=label_names).to_csv(f"{tmp}/class_distribution.csv")
            mlflow.log_artifacts(tmp, artifact_path="eval")

            # Persist full pipeline + register pyfunc
            art_dir = os.path.join(tmp, "pipeline")
            save_pipeline(art_dir, encoder, head, index, label_enc, scaler,
                          feats, args.k, num_classes)
            uri = log_and_register(art_dir, macro_f1=macro_f1, register=True)
            print(f"[mlflow] model uri = {uri}")

        mark_staging(run.info.run_id)
        promoted = promote_if_better(threshold=args.promote_threshold)
        if promoted:
            print(f"[mlflow] promoted version {promoted} to Production")

        print("[explain] 3 test flows")
        explain(model, torch.from_numpy(X_te[:3]), label_names, device=args.device)


if __name__ == "__main__":
    main()
