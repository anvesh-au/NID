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
from rag_nids.data import (
    class_weights, load_cic_ids2017, remap_attack_labels, split_benign_attack,
)
from rag_nids.encoder import FlowEncoder
from rag_nids.infer import evaluate, evaluate_two_stage, explain
from rag_nids.lifecycle import (
    ensure_experiment, dataset_hash, log_and_register, mark_staging, promote_if_better,
    save_pipeline, save_two_stage_pipeline, log_and_register_two_stage,
)
from rag_nids.stage1_vae import calibrate_threshold, train_vae
from rag_nids.train import build_index, pretrain_scarf, train_encoder, train_head
from rag_nids.two_stage import TwoStageNIDS


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


def _run_two_stage(args, X_tr, y_tr, X_te, y_te, label_names, label_enc,
                   feats, scaler, num_classes, run):
    """Two-stage pipeline: VAE BENIGN-filter → RAG attack-classifier.

    Train Stage 1 on BENIGN-only train rows, calibrate θ on a labelled mixed
    val slice, train Stage 2 RAGNIDS on attack-only train rows (label space
    excludes BENIGN), then evaluate three tiers on the held-out test set.
    """
    benign_label = int(np.where(np.array(label_names) == "BENIGN")[0][0])
    print(f"[two-stage] benign_label={benign_label}  ({label_names[benign_label]})")

    # ---- Stage 1: BENIGN VAE ----
    X_b_tr, X_a_tr, y_a_tr = split_benign_attack(X_tr, y_tr, benign_label)
    print(f"[two-stage] train: benign={len(X_b_tr)} attack={len(X_a_tr)}")

    print("[stage1] training VAE on BENIGN")
    vae = train_vae(
        X_b_tr, input_dim=X_tr.shape[1],
        latent_dim=args.vae_latent_dim, hidden=args.vae_hidden,
        epochs=args.vae_epochs, batch_size=args.vae_batch_size, lr=args.vae_lr,
        beta=args.vae_beta, device=args.device,
        patience=args.vae_patience, val_frac=0.1, seed=args.seed,
    )

    # Calibrate θ on a labelled mixed val slice carved from train (stratified).
    val_frac = 0.1
    val_idx = train_test_split(
        np.arange(len(X_tr)), test_size=val_frac, stratify=y_tr,
        random_state=args.seed,
    )[1]
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    print(f"[stage1] calibrating θ on val slice (n={len(X_val)}) "
          f"target_recall={args.vae_threshold_recall}")
    threshold, s1_metrics = calibrate_threshold(
        vae, X_val, y_val, benign_label=benign_label,
        target_recall=args.vae_threshold_recall, beta=args.vae_beta,
        device=args.device,
    )
    mlflow.log_metrics({k: float(v) for k, v in s1_metrics.items()
                        if isinstance(v, (int, float))})

    # ---- Stage 2: attack-only RAGNIDS ----
    y_a_tr_new, attack_names, orig_to_new = remap_attack_labels(
        y_a_tr, label_names, benign_label,
    )
    new_to_orig = {v: k for k, v in orig_to_new.items()}
    n_attack_classes = len(attack_names)
    print(f"[stage2] attack-only label space: {n_attack_classes} classes -> {attack_names}")
    mlflow.log_params({"n_attack_classes": n_attack_classes,
                       "stage1_threshold": float(threshold)})

    cw2 = None
    if args.class_weighted_ce:
        counts = np.bincount(y_a_tr_new, minlength=n_attack_classes).astype(np.float32)
        w = counts.sum() / (n_attack_classes * np.maximum(counts, 1))
        cw2 = torch.from_numpy(w)

    init_encoder = None
    if args.scarf_epochs > 0:
        print("[stage2/scarf] pretraining encoder on attack-only flows")
        init_encoder = FlowEncoder(input_dim=X_a_tr.shape[1], embed_dim=args.embed_dim)
        init_encoder = pretrain_scarf(
            X_a_tr, init_encoder, epochs=args.scarf_epochs, lr=args.enc_lr,
            corruption_rate=args.scarf_corruption, temperature=args.scarf_temperature,
            device=args.device,
        )

    print("[stage2] training encoder on attack-only flows")
    encoder = train_encoder(
        X_a_tr, y_a_tr_new, num_classes=n_attack_classes,
        embed_dim=args.embed_dim, epochs=args.enc_epochs, lr=args.enc_lr,
        supcon_weight=args.supcon_weight, ce_weight=args.ce_weight,
        temperature=args.temperature, device=args.device,
        ce_class_weights=cw2, init_encoder=init_encoder,
        patience=args.enc_patience, val_frac=args.enc_val_frac, seed=args.seed,
    )

    print("[stage2] building attack-only FAISS index")
    index = build_index(encoder, X_a_tr, y_a_tr_new,
                        use_hnsw=args.hnsw, device=args.device)
    mlflow.log_params({"index_total": index.stats().total,
                       "index_type": "hnsw" if args.hnsw else "flat"})

    # Build a held-out attack-val slice for head early stopping.
    # Use the BENIGN-included test slice for end-to-end eval — but the head's
    # val signal must be attack-only macro-F1, not 15-class.
    X_a_te, y_a_te = X_te[y_te != benign_label], y_te[y_te != benign_label]
    y_a_te_new = np.array([orig_to_new[int(v)] for v in y_a_te], dtype=np.int64)

    print("[stage2] training cross-attention head")
    head = train_head(
        encoder, index, X_a_tr, y_a_tr_new, num_classes=n_attack_classes,
        k=args.k, n_heads=args.n_heads, epochs=args.head_epochs, lr=args.head_lr,
        device=args.device, val=(X_a_te, y_a_te_new),
        ce_class_weights=cw2, loss_name=args.loss_name, focal_gamma=args.focal_gamma,
        patience=args.head_patience,
    )

    stage2 = RAGNIDS(encoder, head, index, k=args.k).to(args.device)
    two_stage = TwoStageNIDS(
        vae=vae, threshold=threshold, stage2=stage2,
        benign_label=benign_label, new_to_orig=new_to_orig, beta=args.vae_beta,
    ).to(args.device)

    print("[two-stage] evaluating on test set")
    with tempfile.TemporaryDirectory() as tmp:
        import pandas as pd
        res = evaluate_two_stage(
            two_stage, X_te, y_te, label_names,
            benign_label=benign_label, attack_class_names=attack_names,
            device=args.device, cm_out_dir=tmp,
        )
        mlflow.log_metrics({k: float(v) for k, v in res["metrics"].items()
                            if isinstance(v, (int, float)) and not np.isnan(v)})

        # Per-class F1 in original label space
        f1_per = f1_score(res["trues"], res["preds"], average=None, zero_division=0,
                          labels=range(num_classes))
        def _clean(n: str) -> str:
            return re.sub(r"[^A-Za-z0-9_\-. ]+", "_", n).strip("_")
        for i, name in enumerate(label_names):
            mlflow.log_metric(f"f1/{_clean(name)}", float(f1_per[i]))

        pd.DataFrame({"label": label_names, "f1": f1_per}).to_csv(
            f"{tmp}/per_class_f1.csv", index=False)
        mlflow.log_artifacts(tmp, artifact_path="eval")

        # Persist + register the two-stage pipeline
        art_dir = os.path.join(tmp, "pipeline")
        save_two_stage_pipeline(
            art_dir, vae=vae, threshold=threshold,
            encoder=encoder, head=head, index=index,
            label_encoder=label_enc, scaler=scaler, feature_names=feats,
            k=args.k, num_classes=num_classes,
            n_attack_classes=n_attack_classes, attack_class_names=attack_names,
            benign_label=benign_label, new_to_orig=new_to_orig,
            vae_latent_dim=args.vae_latent_dim, vae_hidden=args.vae_hidden,
            vae_beta=args.vae_beta,
        )
        uri = log_and_register_two_stage(art_dir, e2e_macro_f1=res["macro_f1"],
                                         register=True)
        print(f"[mlflow] two-stage model uri = {uri}")

    mark_staging(run.info.run_id)


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
    # ----- Two-stage architecture -----
    ap.add_argument("--two_stage", action="store_true",
                    help="Enable VAE-filter (Stage 1) + RAG-attack-classifier (Stage 2)")
    ap.add_argument("--vae_epochs", type=int, default=20)
    ap.add_argument("--vae_latent_dim", type=int, default=16)
    ap.add_argument("--vae_hidden", type=int, default=128)
    ap.add_argument("--vae_beta", type=float, default=1.0)
    ap.add_argument("--vae_lr", type=float, default=1e-3)
    ap.add_argument("--vae_batch_size", type=int, default=512)
    ap.add_argument("--vae_patience", type=int, default=None)
    ap.add_argument("--vae_threshold_recall", type=float, default=0.95,
                    help="Target Stage-1 attack recall when calibrating θ")
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    ap.add_argument("--device", default=_default_device())
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

        if args.two_stage:
            _run_two_stage(args, X_tr, y_tr, X_te, y_te, label_names, label_enc,
                           feats, scaler, num_classes, run)
            return

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
        index = build_index(encoder, X_tr, y_tr, use_hnsw=args.hnsw, device=args.device)
        mlflow.log_params({"index_total": index.stats().total,
                           "index_type": "hnsw" if args.hnsw else "flat"})

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
            def _clean(n: str) -> str:
                return re.sub(r"[^A-Za-z0-9_\-. ]+", "_", n).strip("_")
            for i, name in enumerate(label_names):
                mlflow.log_metric(f"f1/{_clean(name)}", float(f1_per[i]))

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
