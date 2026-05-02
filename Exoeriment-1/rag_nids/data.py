"""CIC-IDS2017 loading & preprocessing.

sklearn.preprocessing for StandardScaler + LabelEncoder — standard tabular stack, no custom impls.
"""
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

LEAKY_COLS = {
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port", "Timestamp", "SimillarHTTP", "Fwd Header Length.1",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    drop = [c for c in df.columns if c in LEAKY_COLS]
    return df.drop(columns=drop, errors="ignore")


def _load_cic_ids2017_frame(
    csv_dir: str | Path,
    label_col: str = "Label",
    subsample: int | None = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and clean CIC-IDS2017 into a numeric feature frame plus raw labels."""
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")

    frames = [_clean_columns(pd.read_csv(f, low_memory=False)) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[label_col] = df[label_col].astype(str).str.strip()

    if subsample is not None and len(df) > subsample:
        # FIX: previous formula gave minority classes floor(0)->max(1,0)=1 sample,
        # causing minority-class collapse at POC scale. Guarantee a per-class floor.
        min_per_class = max(20, subsample // (10 * df[label_col].nunique()))
        df = df.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), max(min_per_class, subsample * len(g) // len(df))),
                random_state=seed,
            )
        )

    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])
    # Protocol one-hot if present as small-cardinality int
    if "Protocol" in X_df.columns and X_df["Protocol"].nunique() < 20:
        X_df = pd.get_dummies(X_df, columns=["Protocol"], prefix="proto")

    X_df = X_df.select_dtypes(include=[np.number]).astype(np.float32)
    return X_df, y_raw


def load_cic_ids2017_frame(
    csv_dir: str | Path,
    label_col: str = "Label",
    subsample: int | None = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return the cleaned numeric feature frame, raw labels, and feature names."""
    X_df, y_raw = _load_cic_ids2017_frame(csv_dir, label_col=label_col, subsample=subsample, seed=seed)
    return X_df, y_raw, list(X_df.columns)


def load_cic_ids2017(
    csv_dir: str | Path,
    label_col: str = "Label",
    subsample: int | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list[str], StandardScaler, LabelEncoder]:
    """Load all CIC-IDS2017 CSVs in `csv_dir`, return (X, y, feature_names, scaler, label_enc)."""
    X_df, y_raw, feature_names = load_cic_ids2017_frame(
        csv_dir, label_col=label_col, subsample=subsample, seed=seed
    )

    scaler = StandardScaler().fit(X_df.values)
    X = scaler.transform(X_df.values).astype(np.float32)

    label_enc = LabelEncoder().fit(y_raw)
    y = label_enc.transform(y_raw).astype(np.int64)

    return X, y, feature_names, scaler, label_enc


class CICDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for WeightedRandomSampler."""
    counts = np.bincount(y)
    w = 1.0 / np.maximum(counts, 1)
    return torch.from_numpy(w[y].astype(np.float32))


def ce_class_weights(y: np.ndarray, num_classes: int | None = None) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy losses."""
    counts = np.bincount(y, minlength=num_classes) if num_classes is not None else np.bincount(y)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / max(len(weights), 1)
    return torch.from_numpy(weights.astype(np.float32))
