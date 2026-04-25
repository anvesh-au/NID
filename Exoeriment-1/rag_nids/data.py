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


def load_cic_ids2017(
    csv_dir: str | Path,
    label_col: str = "Label",
    subsample: int | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list[str], StandardScaler, LabelEncoder]:
    """Load all CIC-IDS2017 CSVs in `csv_dir`, return (X, y, feature_names, scaler, label_enc)."""
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

    y_raw = df[label_col].values
    X_df = df.drop(columns=[label_col])
    # Protocol one-hot if present as small-cardinality int
    if "Protocol" in X_df.columns and X_df["Protocol"].nunique() < 20:
        X_df = pd.get_dummies(X_df, columns=["Protocol"], prefix="proto")

    X_df = X_df.select_dtypes(include=[np.number]).astype(np.float32)
    feature_names = list(X_df.columns)

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


# ---------------------------------------------------------------- two-stage helpers
def split_benign_attack(
    X: np.ndarray, y: np.ndarray, benign_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) into BENIGN-only features and attack (X, y) pair.

    Returns (X_benign, X_attack, y_attack). Stage 1 trains on X_benign;
    Stage 2 trains on (X_attack, remap(y_attack)).
    """
    is_b = (y == benign_label)
    return X[is_b], X[~is_b], y[~is_b]


def remap_attack_labels(
    y_attack: np.ndarray, original_classes: list[str], benign_label: int,
) -> Tuple[np.ndarray, list[str], dict]:
    """Drop BENIGN from the label space; remap remaining labels to 0..N-2.

    Returns (y_remapped, attack_class_names, orig_to_new_map). orig_to_new_map
    is a dict mapping original integer label -> new integer label. BENIGN is
    deliberately omitted from the map.
    """
    attack_orig = [i for i in range(len(original_classes)) if i != benign_label]
    orig_to_new = {orig: new for new, orig in enumerate(attack_orig)}
    y_remap = np.array([orig_to_new[int(v)] for v in y_attack], dtype=np.int64)
    attack_names = [original_classes[i] for i in attack_orig]
    return y_remap, attack_names, orig_to_new
