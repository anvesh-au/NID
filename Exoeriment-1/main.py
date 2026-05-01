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
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from rag_nids import RAGNIDS
from rag_nids.data import class_weights, load_cic_ids2017
from rag_nids.encoder import FlowEncoder
from rag_nids.infer import evaluate, explain
from rag_nids.lifecycle import (
    ensure_experiment, dataset_hash, log_and_register, mark_staging, promote_if_better,
    save_pipeline,
)
from rag_nids.session_pipeline import run_session_pipeline
from rag_nids.train import build_index, pretrain_scarf, train_encoder, train_head


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def log_device_status(device: str) -> None:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    print(f"[device] requested={device} torch.cuda.is_available()={cuda_available} "
          f"torch.backends.mps.is_available()={mps_available}")
    if device == "cuda" and not cuda_available:
        print("[warning] CUDA was requested but is not available. Falling back may fail or run on CPU.")
    elif cuda_available and device != "cuda":
        print(f"[warning] CUDA is available but not selected; using '{device}' instead.")
    elif not cuda_available and device == "cpu":
        print("[warning] CUDA is not available; running on CPU.")


def _pkg_versions() -> dict:
    pkgs = ["torch", "numpy", "pandas", "scikit-learn", "faiss-cpu", "faiss-gpu",
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
    ap.add_argument("--pipeline_mode", choices=["standard", "sessions"], default="standard")
    ap.add_argument("--subsample", type=int, default=50_000,
                    help="Max rows to keep per run; use 0 for the full dataset")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of the dataset reserved for the test split")
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
    ap.add_argument("--scarf_epochs", type=int, default=0,
                    help="SCARF pretraining epochs (0 disables)")
    ap.add_argument("--scarf_corruption", type=float, default=0.6)
    ap.add_argument("--scarf_temperature", type=float, default=0.5)
    ap.add_argument("--hnsw", action="store_true")
    ap.add_argument("--enc_patience", type=int, default=None,
                    help="Early-stop encoder if val total-loss doesn't improve for N epochs (None=off)")
    ap.add_argument("--head_patience", type=int, default=None,
                    help="Early-stop head if val macro-F1 doesn't improve for N epochs (None=off)")
    ap.add_argument("--enc_val_frac", type=float, default=0.1,
                    help="Fraction of train carved out for encoder early-stopping val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="poc")
    ap.add_argument("--promote_threshold", type=float, default=0.80)
    ap.add_argument("--train_days", type=int, default=2,
                    help="For session mode: number of initial day sessions used for training")
    ap.add_argument("--session_window_days", type=int, default=2,
                    help="For session mode: retain only the last N sessions in the index")
    ap.add_argument("--session_val_frac", type=float, default=0.1,
                    help="For session mode: validation fraction carved from train days for head training")
    ap.add_argument("--session_eval_batch_size", type=int, default=256,
                    help="For session mode: streaming evaluation batch size")
    ap.add_argument("--session_writeback_threshold", type=float, default=0.95,
                    help="For session mode: min confidence required before a prediction is written back")
    ap.add_argument("--session_report_dir", default=None,
                    help="For session mode: directory where session reports and confusion matrices are saved")
    ap.add_argument("--no_mlflow", action="store_true",
                    help="Disable MLflow tracking, artifact logging, and registry updates")
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--faiss_device", choices=["cpu", "cuda"], default="cpu",
                    help="FAISS index device. 'cuda' requires a GPU-enabled faiss build.")
    args = ap.parse_args()
    subsample = None if args.subsample <= 0 else args.subsample
    use_mlflow = not args.no_mlflow

    set_seed(args.seed)
    log_device_status(args.device)
    if use_mlflow:
        ensure_experiment()

    run_ctx = mlflow.start_run(run_name=args.run_name) if use_mlflow else nullcontext()
    with run_ctx as run:
        if use_mlflow:
            mlflow.log_params(vars(args))
            mlflow.log_params({f"pkg.{k}": v for k, v in _pkg_versions().items()})

        if args.pipeline_mode == "sessions":
            run_session_pipeline(args, use_mlflow=use_mlflow)
            return

        print(f"[data] loading {args.data_dir}")
        X, y, feats, scaler, label_enc = load_cic_ids2017(args.data_dir, subsample=subsample,
                                                          seed=args.seed)
        label_names = list(label_enc.classes_)
        num_classes = len(label_names)
        print(f"[data] X={X.shape}  classes={num_classes}  -> {label_names}")
        if use_mlflow:
            mlflow.log_params({
                "num_classes": num_classes, "num_features": X.shape[1],
                "dataset_hash": dataset_hash(X, y), "dataset_rows": int(X.shape[0]),
            })

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed,
        )

        cw = None
        if args.class_weighted_ce:
            # print("[train] class-weighted CE")
            counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
            w = counts.sum() / (num_classes * np.maximum(counts, 1))
            cw = torch.from_numpy(w)

        init_encoder = None
        if args.scarf_epochs > 0:
            print("[scarf] pretraining encoder")
            init_encoder = FlowEncoder(input_dim=X_tr.shape[1], embed_dim=args.embed_dim)
            init_encoder = pretrain_scarf(
                X_tr, init_encoder,
                epochs=args.scarf_epochs, lr=args.enc_lr,
                corruption_rate=args.scarf_corruption,
                temperature=args.scarf_temperature,
                device=args.device,
            )

        print("[train] encoder (SupCon fine-tune)" if init_encoder else "[train] encoder")
        encoder = train_encoder(
            X_tr, y_tr, num_classes=num_classes,
            embed_dim=args.embed_dim, epochs=args.enc_epochs, lr=args.enc_lr,
            supcon_weight=args.supcon_weight, ce_weight=args.ce_weight,
            temperature=args.temperature, device=args.device,
            ce_class_weights=cw, init_encoder=init_encoder,
            patience=args.enc_patience, val_frac=args.enc_val_frac, seed=args.seed,
        )

        print("[index] building")
        print(f"[index] faiss_device={args.faiss_device}")
        index = build_index(
            encoder, X_tr, y_tr,
            use_hnsw=args.hnsw, device=args.device, faiss_device=args.faiss_device,
        )
        if use_mlflow:
            mlflow.log_params({"index_total": index.stats().total,
                               "index_type": "hnsw" if args.hnsw else "flat",
                               "faiss_device_effective": index.faiss_device})

        print("[train] cross-attention head")
        head = train_head(
            encoder, index, X_tr, y_tr, num_classes=num_classes,
            k=args.k, n_heads=args.n_heads, epochs=args.head_epochs, lr=args.head_lr,
            device=args.device, val=(X_te, y_te),
            ce_class_weights=cw, loss_name=args.loss_name, focal_gamma=args.focal_gamma,
            patience=args.head_patience,
        )

        model = RAGNIDS(encoder, head, index, k=args.k).to(args.device)

        print("[eval] test set")
        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd
            res = evaluate(model, X_te, y_te, label_names, device=args.device,
                           cm_out_dir=tmp)  # writes confusion_matrix_{counts,rates}.csv
            preds, trues = res["preds"], res["trues"]
            macro_f1 = res["macro_f1"]

            # Per-class F1 — sanitize names for MLflow (rejects non-ASCII in metric keys).
            f1_per = f1_score(trues, preds, average=None, zero_division=0,
                              labels=range(num_classes))
            pd.DataFrame({"label": label_names, "f1": f1_per}).to_csv(
                f"{tmp}/per_class_f1.csv", index=False)
            pd.DataFrame({"train": np.bincount(y_tr, minlength=num_classes),
                          "test": np.bincount(y_te, minlength=num_classes)},
                         index=label_names).to_csv(f"{tmp}/class_distribution.csv")
            if use_mlflow:
                def _clean(n: str) -> str:
                    return re.sub(r"[^A-Za-z0-9_\-. ]+", "_", n).strip("_")
                for i, name in enumerate(label_names):
                    mlflow.log_metric(f"f1/{_clean(name)}", float(f1_per[i]))
                mlflow.log_artifacts(tmp, artifact_path="eval")

                # Persist full pipeline + register pyfunc
                art_dir = os.path.join(tmp, "pipeline")
                save_pipeline(art_dir, encoder, head, index, label_enc, scaler,
                              feats, args.k, num_classes)
                uri = log_and_register(art_dir, macro_f1=macro_f1, register=True)
                print(f"[mlflow] model uri = {uri}")

        if use_mlflow:
            mark_staging(run.info.run_id)
            promoted = promote_if_better(threshold=args.promote_threshold)
            if promoted:
                print(f"[mlflow] promoted version {promoted} to Production")

        print("[explain] 3 test flows")
        explain(model, torch.from_numpy(X_te[:3]), label_names, device=args.device)


if __name__ == "__main__":
    main()
