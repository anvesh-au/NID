"""FAISS-backed flow index with write-back + TTL eviction."""
import time
from dataclasses import dataclass

import faiss
import numpy as np


PINNED = 0
WRITEBACK = 1


@dataclass
class IndexStats:
    total: int
    pinned: int
    writeback: int


def _faiss_gpu_available() -> bool:
    return all(hasattr(faiss, name) for name in ("StandardGpuResources", "index_cpu_to_gpu"))


class FlowIndex:
    def __init__(self, embed_dim: int, use_hnsw: bool = False, ttl_seconds: float = 7 * 24 * 3600,
                 max_writeback: int = 200_000, faiss_device: str = "cpu"):
        self.embed_dim = embed_dim
        self.use_hnsw = use_hnsw
        self.faiss_device = "cpu"
        self._gpu_resources = None
        self.index = self._new_index()
        self._set_faiss_device(faiss_device)
        self.embeddings = np.empty((0, embed_dim), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int32)
        self.timestamps = np.empty(0, dtype=np.float64)
        self.source = np.empty(0, dtype=np.int8)
        self.ttl = ttl_seconds
        self.max_writeback = max_writeback

    def _new_index(self):
        if self.use_hnsw:
            index = faiss.IndexHNSWFlat(self.embed_dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
            return index
        return faiss.IndexFlatIP(self.embed_dim)

    def _set_faiss_device(self, faiss_device: str) -> None:
        if faiss_device == "cuda":
            if not _faiss_gpu_available():
                print("[warning] FAISS GPU was requested but this faiss build has no CUDA support; using CPU index.")
                self.faiss_device = "cpu"
                return
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self.index)
                self.faiss_device = "cuda"
                print("[index] using FAISS GPU index on CUDA:0")
            except Exception as exc:
                self._gpu_resources = None
                self.faiss_device = "cpu"
                print(f"[warning] FAISS GPU was requested but could not be initialized ({exc}); using CPU index.")
            return
        self.faiss_device = "cpu"

    def to_cpu_index(self):
        if self.faiss_device == "cuda":
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    # ---- building ----
    def add(self, embeddings: np.ndarray, labels: np.ndarray, source: int = PINNED):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)
        self.labels = np.concatenate([self.labels, labels.astype(np.int32)])
        now = time.time()
        self.timestamps = np.concatenate([self.timestamps, np.full(len(labels), now)])
        self.source = np.concatenate([self.source, np.full(len(labels), source, dtype=np.int8)])

    # ---- query ----
    def search(self, query: np.ndarray, k: int = 10):
        query = np.ascontiguousarray(query, dtype=np.float32)
        k = min(k, self.index.ntotal)  # FIX: FAISS returns -1 padding when k > ntotal
        sims, idx = self.index.search(query, k)
        # FIX: clamp any residual -1 entries before fancy-indexing labels
        safe_idx = np.where(idx >= 0, idx, 0)
        neighbor_labels = self.labels[safe_idx]
        return sims, idx, neighbor_labels

    def reconstruct_batch(self, idx: np.ndarray) -> np.ndarray:
        safe_idx = np.where(idx >= 0, idx, 0)
        return self.embeddings[safe_idx]

    # ---- write-back ----
    def writeback(self, embedding: np.ndarray, label: int, min_confidence: float,
                  confidence: float, is_attack: bool):
        if not (is_attack and confidence >= min_confidence):
            return False
        self.add(embedding[None, :], np.array([label]), source=WRITEBACK)
        self._cap_writeback()
        return True

    # ---- maintenance ----
    def _cap_writeback(self):
        wb_mask = self.source == WRITEBACK
        n_wb = int(wb_mask.sum())
        if n_wb <= self.max_writeback:
            return
        wb_idx = np.where(wb_mask)[0]
        order = np.argsort(self.timestamps[wb_idx])
        drop = wb_idx[order[: n_wb - self.max_writeback]]
        self._rebuild_dropping(drop)

    def evict_expired(self):
        now = time.time()
        expired = (self.source == WRITEBACK) & ((now - self.timestamps) > self.ttl)
        if expired.any():
            self._rebuild_dropping(np.where(expired)[0])

    def _rebuild_dropping(self, drop_indices: np.ndarray):
        """FAISS IndexFlat doesn't support remove_ids cleanly for HNSW — rebuild."""
        keep = np.ones(self.labels.size, dtype=bool)
        keep[drop_indices] = False
        kept_emb = self.embeddings[keep]
        self.index = self._new_index()
        self._set_faiss_device(self.faiss_device)
        self.index.add(np.ascontiguousarray(kept_emb, dtype=np.float32))
        self.embeddings = kept_emb
        self.labels = self.labels[keep]
        self.timestamps = self.timestamps[keep]
        self.source = self.source[keep]

    def stats(self) -> IndexStats:
        return IndexStats(
            total=int(self.labels.size),
            pinned=int((self.source == PINNED).sum()),
            writeback=int((self.source == WRITEBACK).sum()),
        )
