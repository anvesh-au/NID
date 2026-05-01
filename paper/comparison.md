Comparison Set

  | Baseline / Model | Source / Inspiration | What You Should Run | Primary Metrics | Why It Matters |
  |---|---|---|---|---|
  | Logistic Regression | Arcos-Argudo et al. 2025 (https://www.mdpi.com/1999-4893/18/12/749), Al Lail et al. 2023
  (https://www.mdpi.com/1999-5903/15/7/243) | Multiclass LR on your exact preprocessing and split/session setup | macro-F1, per-class F1, benign
  FPR | Strong transparent linear baseline |
  | Random Forest | Al Lail et al. 2023 (https://www.mdpi.com/1999-5903/15/7/243), Maseer et al. 2021
  (https://doi.org/10.1109/ACCESS.2021.3056614) | RF with tuned depth/trees on same split | macro-F1, weighted F1, inference cost | Strong clas
  sical tabular baseline |
  | XGBoost or ExtraTrees | common CICIDS2017 literature; also justified by newer ensemble papers | Gradient-boosted/tree-ensemble baseline |
  macro-F1, minority recall, benign FPR | Likely strongest non-neural conventional baseline |
  | MLP | common DL baseline in CICIDS2017 papers | Plain feedforward network without retrieval | macro-F1, weighted F1 | Tests whether
  retrieval beats plain parametric DL |
  | AE + LR or AE + RF | Abdulhammed et al. 2019 (https://www.mdpi.com/2079-9292/8/3/322), Arcos-Argudo et al. 2025
  (https://www.mdpi.com/1999-4893/18/12/749) | Autoencoder embeddings, then train shallow classifier | macro-F1, AUC if useful | Representation-
  learning baseline without retrieval |
  | kNN on raw/scaled features | classical nonparametric baseline | kNN on normalized CIC features | macro-F1, minority precision/recall | Shows
  whether retrieval alone on raw space is enough |
  | Encoder + Linear Head | internal ablation | Your trained encoder, no retrieval, just classification head | macro-F1, per-class F1 | Isolates
  retrieval benefit |
  | Encoder + kNN retrieval vote | internal ablation | Your encoder embeddings + nearest-neighbor majority vote | macro-F1, minority recall |
  Isolates learned embedding + retrieval without attention |
  | Static RAG-NIDS | your method minus refresh | Full retrieval-conditioned model with fixed memory/index | macro-F1, session stability |
  Baseline for adaptive memory claim |
  | Adaptive RAG-NIDS | your full method | Retrieval-conditioned model with eviction/refresh across sessions | macro-F1, drift robustness,
  memory efficiency | Main proposed method |

  Paper Mapping

  You can organize the experimental claims into three blocks:

  1. Does retrieval help IDS classification?
     Compare:

  - MLP
  - Encoder + Linear Head
  - Encoder + kNN retrieval vote
  - Static RAG-NIDS

  2. Does adaptive memory help beyond static retrieval?
     Compare:

  - Static RAG-NIDS
  - Adaptive RAG-NIDS

  3. Can the method stay competitive with less retained training data?
     Compare at 40%, 50%, 80%, and full training:

  - Random Forest
  - XGBoost/ExtraTrees
  - Static RAG-NIDS
  - Adaptive RAG-NIDS

  Metrics To Report

  Do not lead with accuracy. Given your current results, the table order should be:

  - Macro-F1
  - Per-class F1
  - Macro precision
  - Macro recall
  - Benign false positive rate
  - Minority-class precision/recall for Bot, Infiltration, Web Attack - Brute Force, Web Attack - XSS, SQL Injection
  - Memory size / retained exemplars
  - Inference time per sample or per batch
  - Session-to-session delta in macro-F1 for refresh experiments

  Most Important Ablations

  These are the ones that will make the paper convincing:

  - No retrieval vs retrieval
  - Raw-feature kNN vs embedding-space retrieval
  - Static memory vs eviction-refresh memory
  - Full-data training vs 40/50/80% training with refresh
  - Single-session evaluation vs multi-session evaluation

  How To Frame The Novelty Against Prior Work

  - Against Maseer et al. 2021 (https://doi.org/10.1109/ACCESS.2021.3056614) and Al Lail et al. 2023 (https://www.mdpi.com/1999-5903/15/7/243):
    you are not just benchmarking static classifiers; you are testing retrieval-conditioned decision-making under imbalance.
  - Against Abdulhammed et al. 2019 (https://www.mdpi.com/2079-9292/8/3/322): you are not stopping at representation learning; you use an exter
    nal memory during inference.
  - Against Li et al. 2024 (https://www.mdpi.com/1424-8220/24/7/2122): you are not only improving embeddings with contrastive learning; you are
    using retrieval as a first-class part of the classifier.
  - Against D’hooge et al. 2020 (https://doi.org/10.1016/j.jisa.2020.102564), Abdulrahman and Ibrahem 2021
    (https://doi.org/10.24996/ijs.2021.62.1.30), and Guo et al. 2025 (https://www.mdpi.com/1999-5903/17/10/456): your adaptive memory is the pr
    actical mechanism for handling evolving sessions and drift without fully retraining from scratch.

  Recommended Final Experiment Table Structure In The Paper

  Table 1: Main performance on standard split

  - LR, RF, XGBoost/ExtraTrees, MLP, AE+LR, Encoder+Linear, Encoder+kNN, Static RAG, Adaptive RAG

  Table 2: Low-data regime

  - same key models at 40%, 50%, 80%, 100% train fraction

  Table 3: Session-wise / drift evaluation

  - Static RAG vs Adaptive RAG across sessions
  - report macro-F1, minority F1, benign FPR, memory size

  Table 4: Ablation

  - remove retrieval
  - remove attention
  - freeze memory
  - disable eviction
  - disable refresh
