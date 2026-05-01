"""CIC-IDS2017 loading & preprocessing.

sklearn.preprocessing for StandardScaler + LabelEncoder — standard tabular stack, no custom impls.
"""
from dataclasses import dataclass
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
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class SessionData:
    name: str
    X: np.ndarray
    y: np.ndarray


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    drop = [c for c in df.columns if c in LEAKY_COLS]
    return df.drop(columns=drop, errors="ignore")


def _subsample_df(df: pd.DataFrame, label_col: str, subsample: int | None, seed: int) -> pd.DataFrame:
    if subsample is None or len(df) <= subsample:
        return df
    min_per_class = max(20, subsample // (10 * df[label_col].nunique()))
    return df.groupby(label_col, group_keys=False).apply(
        lambda g: g.sample(
            min(len(g), max(min_per_class, subsample * len(g) // len(df))),
            random_state=seed,
        )
    )


def _prepare_day_frame(df: pd.DataFrame, label_col: str, subsample: int | None, seed: int) -> pd.DataFrame:
    df = _clean_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[label_col] = df[label_col].astype(str).str.strip()
    return _subsample_df(df, label_col=label_col, subsample=subsample, seed=seed)


def _group_cic_day_files(csv_dir: Path) -> dict[str, list[Path]]:
    groups = {day: [] for day in DAY_ORDER}
    for path in sorted(csv_dir.glob("*.csv")):
        for day in DAY_ORDER:
            if day.lower() in path.name.lower():
                groups[day].append(path)
                break
    return {day: files for day, files in groups.items() if files}


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

    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = _prepare_day_frame(df, label_col=label_col, subsample=subsample, seed=seed)

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


def load_cic_ids2017_sessions(
    csv_dir: str | Path,
    label_col: str = "Label",
    subsample: int | None = None,
    seed: int = 0,
    max_days: int = 5,
) -> Tuple[list[SessionData], list[str], StandardScaler, LabelEncoder]:
    """Load CIC-IDS2017 split into day sessions with one shared scaler/label encoder."""
    csv_dir = Path(csv_dir)
    day_files = _group_cic_day_files(csv_dir)
    if not day_files:
        raise FileNotFoundError(f"No CIC-IDS2017 day CSVs found in {csv_dir}")

    selected_days = list(day_files.keys())[:max_days]
    frames = []
    day_lengths: list[tuple[str, int]] = []
    for offset, day in enumerate(selected_days):
        day_df = pd.concat(
            [pd.read_csv(path, low_memory=False) for path in day_files[day]],
            ignore_index=True,
        )
        day_df = _prepare_day_frame(day_df, label_col=label_col, subsample=subsample, seed=seed + offset)
        frames.append(day_df)
        day_lengths.append((day, len(day_df)))

    df = pd.concat(frames, ignore_index=True)
    y_raw = df[label_col].values
    X_df = df.drop(columns=[label_col])
    if "Protocol" in X_df.columns and X_df["Protocol"].nunique() < 20:
        X_df = pd.get_dummies(X_df, columns=["Protocol"], prefix="proto")

    X_df = X_df.select_dtypes(include=[np.number]).astype(np.float32)
    feature_names = list(X_df.columns)
    scaler = StandardScaler().fit(X_df.values)
    X_all = scaler.transform(X_df.values).astype(np.float32)

    label_enc = LabelEncoder().fit(y_raw)
    y_all = label_enc.transform(y_raw).astype(np.int64)

    sessions: list[SessionData] = []
    start = 0
    for day, length in day_lengths:
        stop = start + length
        sessions.append(SessionData(name=day, X=X_all[start:stop], y=y_all[start:stop]))
        start = stop

    return sessions, feature_names, scaler, label_enc


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
