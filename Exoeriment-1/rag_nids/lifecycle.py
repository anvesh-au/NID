"""MLflow model lifecycle: pyfunc wrapper, registration, and staging promotion.

mlflow: used directly — it already covers experiment tracking, artifacts, and the
Model Registry. No additional framework is warranted.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from .classifier import CrossAttentionHead
from .encoder import FlowEncoder
from .index import FlowIndex, PINNED
from .pipeline import RAGNIDS

EXPERIMENT_NAME = "rag_nids"
REGISTERED_MODEL = "rag_nids_pipeline"


# ---------------------------------------------------------------- artifact IO
def _save_index(index: FlowIndex, path: Path) -> None:
    faiss.write_index(index.to_cpu_index(), str(path / "faiss.index"))
    np.savez(path / "index_meta.npz",
             embeddings=index.embeddings,
             labels=index.labels, timestamps=index.timestamps, source=index.source,
             session_ids=index.session_ids,
             use_hnsw=index.use_hnsw, faiss_device=index.faiss_device)


def _load_index(path: Path, embed_dim: int) -> FlowIndex:
    meta = np.load(path / "index_meta.npz")
    use_hnsw = bool(meta["use_hnsw"]) if "use_hnsw" in meta.files else False
    ix = FlowIndex(embed_dim=embed_dim, use_hnsw=use_hnsw, faiss_device="cpu")
    ix.index = faiss.read_index(str(path / "faiss.index"))
    ix.embeddings = meta["embeddings"] if "embeddings" in meta.files else ix.index.reconstruct_n(0, ix.index.ntotal)
    ix.labels = meta["labels"]; ix.timestamps = meta["timestamps"]; ix.source = meta["source"]
    if "session_ids" in meta.files:
        ix.session_ids = meta["session_ids"]
    return ix


def save_pipeline(
    artifact_dir: str | Path,
    encoder: FlowEncoder,
    head: CrossAttentionHead,
    index: FlowIndex,
    label_encoder,
    scaler,
    feature_names: list[str],
    k: int,
    num_classes: int,
) -> Path:
    """Dump every component needed to rebuild the pipeline from scratch."""
    d = Path(artifact_dir); d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), d / "encoder.pt")
    torch.save(head.state_dict(), d / "head.pt")
    _save_index(index, d)
    with open(d / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(d / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (d / "config.json").write_text(json.dumps({
        "input_dim": len(feature_names),
        "embed_dim": encoder.embed_dim,
        "num_classes": num_classes,
        "k": k,
        "feature_names": feature_names,
    }))
    return d


# ---------------------------------------------------------------- pyfunc wrapper
class RAGNIDSWrapper(mlflow.pyfunc.PythonModel):
    """Single-call loader/inference wrapper. Input: DataFrame or ndarray of raw features."""

    def load_context(self, context):
        root = Path(context.artifacts["pipeline"])
        cfg = json.loads((root / "config.json").read_text())
        self.cfg = cfg

        encoder = FlowEncoder(input_dim=cfg["input_dim"], embed_dim=cfg["embed_dim"])
        encoder.load_state_dict(torch.load(root / "encoder.pt", map_location="cpu"))
        head = CrossAttentionHead(cfg["embed_dim"], cfg["num_classes"])
        head.load_state_dict(torch.load(root / "head.pt", map_location="cpu"))
        index = _load_index(root, cfg["embed_dim"])

        with open(root / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(root / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        self.model = RAGNIDS(encoder, head, index, k=cfg["k"]).eval()

    def predict(self, context, model_input: Any) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            X = model_input[self.cfg["feature_names"]].values.astype(np.float32)
        else:
            X = np.asarray(model_input, dtype=np.float32)
        X = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            preds = self.model.predict(torch.from_numpy(X))
        return pd.DataFrame([{
            "label": self.label_encoder.inverse_transform([p.label])[0],
            "confidence": p.confidence,
            "neighbor_labels": self.label_encoder.inverse_transform(p.neighbor_labels).tolist(),
            "neighbor_sims": p.neighbor_sims.tolist(),
        } for p in preds])


# ---------------------------------------------------------------- MLflow glue
def ensure_experiment() -> str:
    mlflow.set_experiment(EXPERIMENT_NAME)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp.experiment_id


def dataset_hash(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(X.tobytes()); h.update(y.tobytes())
    return h.hexdigest()[:12]


def log_and_register(
    artifact_dir: Path,
    macro_f1: float,
    register: bool = True,
) -> Optional[str]:
    """Log the saved artifact dir as a pyfunc model; optionally register it."""
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RAGNIDSWrapper(),
        artifacts={"pipeline": str(artifact_dir)},
        registered_model_name=REGISTERED_MODEL if register else None,
    )
    mlflow.log_metric("test_macro_f1", macro_f1)
    return model_info.model_uri


# ---------------------------------------------------------------- lifecycle API
def list_runs(max_results: int = 25) -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs([exp.experiment_id],
                              order_by=["metrics.test_macro_f1 DESC"],
                              max_results=max_results)


def promote_if_better(threshold: float = 0.80) -> Optional[str]:
    """Promote the latest Staging version to Production if macro-F1 ≥ threshold
    AND it beats the current Production version."""
    client = MlflowClient()
    try:
        staging = client.get_latest_versions(REGISTERED_MODEL, stages=["Staging"])
    except mlflow.exceptions.RestException:
        return None
    if not staging:
        return None
    cand = staging[0]
    cand_f1 = float(client.get_run(cand.run_id).data.metrics.get("test_macro_f1", -1))
    if cand_f1 < threshold:
        print(f"[lifecycle] staging v{cand.version} f1={cand_f1:.4f} < threshold {threshold}")
        return None

    prod = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])
    prod_f1 = float(client.get_run(prod[0].run_id).data.metrics.get("test_macro_f1", -1)) if prod else -1

    if cand_f1 > prod_f1:
        client.transition_model_version_stage(
            REGISTERED_MODEL, cand.version, "Production",
            archive_existing_versions=True,
        )
        print(f"[lifecycle] promoted v{cand.version} f1={cand_f1:.4f} (was {prod_f1:.4f})")
        return cand.version
    print(f"[lifecycle] v{cand.version} f1={cand_f1:.4f} did not beat prod f1={prod_f1:.4f}")
    return None


def mark_staging(run_id: str) -> Optional[str]:
    """Attach the model version produced by `run_id` to the Staging stage."""
    client = MlflowClient()
    for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        if v.run_id == run_id:
            client.transition_model_version_stage(REGISTERED_MODEL, v.version, "Staging")
            return v.version
    return None


def load_production_model():
    """Single-call loader for the current Production model."""
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/Production")
