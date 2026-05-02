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

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False

from rag_nids import RAGNIDS
from rag_nids.continual import run_continual_sessions
from rag_nids.data import ce_class_weights, load_cic_ids2017
from rag_nids.encoder import FlowEncoder
from rag_nids.infer import evaluate, explain
from rag_nids.lifecycle import (
    ensure_experiment, dataset_hash, log_and_register, mark_staging, promote_if_better,
    save_pipeline,
)
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
    ap.add_argument("--data_dir", default=None,
                    help="Directory containing CIC-IDS2017 CSVs for the single-session pipeline")
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
    ap.add_argument("--session_manifest", default=None,
                    help="JSON manifest describing continual-learning sessions")
    ap.add_argument("--session_output_dir", default=None,
                    help="Optional directory for per-session CSV artifacts")
    ap.add_argument("--replay_per_class", type=int, default=50,
                    help="Replay exemplars retained per class across sessions")
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
    if args.data_dir is None and args.session_manifest is None:
        ap.error("either --data_dir or --session_manifest must be provided")
    subsample = None if args.subsample <= 0 else args.subsample
    use_mlflow = (not args.no_mlflow) and _HAS_MLFLOW
    if not args.no_mlflow and not _HAS_MLFLOW:
        print("[warning] mlflow is not installed; proceeding with --no_mlflow behavior.")

    set_seed(args.seed)
    log_device_status(args.device)
    if use_mlflow:
        ensure_experiment()

    run_ctx = mlflow.start_run(run_name=args.run_name) if use_mlflow else nullcontext()
    with run_ctx as run:
        if use_mlflow:
            mlflow.log_params(vars(args))
            mlflow.log_params({f"pkg.{k}": v for k, v in _pkg_versions().items()})

        if args.session_manifest is not None:
            results = run_continual_sessions(
                args.session_manifest,
                device=args.device,
                test_size=args.test_size,
                embed_dim=args.embed_dim,
                k=args.k,
                enc_epochs=args.enc_epochs,
                head_epochs=args.head_epochs,
                enc_lr=args.enc_lr,
                head_lr=args.head_lr,
                supcon_weight=args.supcon_weight,
                ce_weight=args.ce_weight,
                temperature=args.temperature,
                n_heads=args.n_heads,
                loss_name=args.loss_name,
                focal_gamma=args.focal_gamma,
                replay_per_class=args.replay_per_class,
                enc_patience=args.enc_patience,
                head_patience=args.head_patience,
                seed=args.seed,
                output_dir=args.session_output_dir,
            )
            print("[session] summary")
            for r in results:
                print(f"  {r.name}: acc={r.accuracy:.4f} prec={r.precision_macro:.4f} "
                      f"rec={r.recall_macro:.4f} f1={r.f1_macro:.4f}")
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
            cw = ce_class_weights(y_tr, num_classes=num_classes)

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
