# Experiment Section Draft: RAG-NIDS

## 1. Experimental Objective

The experiments are designed to evaluate whether `RAG-NIDS`, a single-stage retrieval-augmented intrusion detection system, improves class-balanced network intrusion detection compared with conventional static classifiers and non-retrieval neural baselines.

The evaluation should focus on three central questions:

1. Does retrieval-conditioned classification improve performance over purely parametric models?
2. Does retrieval in a learned embedding space outperform raw-feature nearest-neighbor retrieval?
3. Can an adaptive retrieval memory preserve performance when training data retention is reduced or traffic is evaluated across multiple sessions?

The experiments should not be framed primarily as an accuracy contest. `CIC-IDS2017` is heavily imbalanced, so the main evidence should come from macro-F1, per-class F1, minority-class behavior, and false positive behavior on benign traffic.


## 2. Dataset

The primary dataset is `CIC-IDS2017`, a flow-based intrusion detection dataset containing benign traffic and multiple attack categories. The task is multiclass classification over the original traffic labels.

The expected label set includes:

- `BENIGN`
- `Bot`
- `DDoS`
- `DoS GoldenEye`
- `DoS Hulk`
- `DoS Slowhttptest`
- `DoS slowloris`
- `FTP-Patator`
- `Heartbleed`
- `Infiltration`
- `PortScan`
- `SSH-Patator`
- `Web Attack - Brute Force`
- `Web Attack - Sql Injection`
- `Web Attack - XSS`

The paper should explicitly state that the dataset is highly imbalanced and that some classes contain very few examples. This matters because accuracy and weighted-F1 can appear strong even when rare attack classes are poorly detected.


## 3. Preprocessing Protocol

All models should use the same preprocessing pipeline for a fair comparison.

The preprocessing pipeline should include:

- loading and merging all dataset CSV files
- removing leakage-prone identifiers such as flow IDs, IP addresses, ports, and timestamps
- replacing infinite values and dropping invalid rows
- one-hot encoding low-cardinality protocol features when present
- retaining numeric features only
- standardizing features using `StandardScaler`
- encoding class labels using `LabelEncoder`

When subsampling is used, the sampling protocol should preserve minority classes using a class-aware minimum floor.


## 4. Evaluation Protocol

### 4.1 Static Split Evaluation

The baseline evaluation should use a stratified train-test split. The current project uses a configurable `test_size`, with `0.2` as the default test fraction.

This evaluation answers:

- how well each model performs under a standard static supervised-learning setting
- whether retrieval improves over equivalent non-retrieval baselines

### 4.2 Reduced Training Data Evaluation

To test memory efficiency and retained-data behavior, evaluate the method under multiple retained training fractions:

- `40%`
- `50%`
- `80%`
- `100%`

For each fraction, train and evaluate the selected baselines using the same test protocol.

This evaluation supports the claim that `RAG-NIDS` can remain competitive even when less initial training data is retained, especially if adaptive memory refresh is later applied.

### 4.3 Multi-Session Evaluation

The paper should include a session-wise experiment if the data and scripts are available.

The intended setup is:

1. divide traffic into ordered or pseudo-ordered sessions
2. initialize the model and memory on an initial session or initial training subset
3. evaluate on subsequent sessions
4. compare static memory against adaptive memory refresh

The goal is to test whether the retrieval database remains useful as traffic changes and whether refresh mechanisms help preserve performance.


## 5. Models and Baselines

The baseline set should contain both external-style baselines and internal ablations. The most defensible results will come from re-running these baselines under the same preprocessing and split protocol rather than copying numbers from prior papers.

### 5.1 Classical Machine Learning Baselines

Use these as static tabular baselines:

- `Logistic Regression`
- `Random Forest`
- `ExtraTrees` or `XGBoost`

These baselines are important because tree ensembles are often strong on tabular IDS data.

### 5.2 Neural Baselines

Use these to test whether retrieval is better than a plain neural classifier:

- `MLP`
- `Autoencoder + classifier`
- `Encoder + linear head`

The `Encoder + linear head` baseline is especially important because it isolates the value of retrieval. It uses the same learned representation family but removes neighbor-conditioned inference.

### 5.3 Retrieval Baselines

Use these to test whether the proposed retrieval design matters:

- `kNN` on standardized raw features
- `kNN` over learned encoder embeddings
- `Encoder + kNN majority vote`

These baselines separate simple nearest-neighbor behavior from the full retrieval-conditioned classifier.

### 5.4 Proposed Method Variants

Evaluate:

- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`, if the session refresh loop is used

`Static RAG-NIDS` uses a fixed memory built from retained training embeddings. `Adaptive RAG-NIDS` updates the retrieval memory over time through writeback and eviction.


## 6. Ablation Studies

Ablations should identify which part of the method is responsible for performance.

Recommended ablations:

- remove retrieval and use only the encoder classifier
- use raw-feature kNN instead of learned embedding retrieval
- use embedding-space kNN vote instead of cross-attention fusion
- use static memory instead of adaptive memory
- vary memory size
- vary number of retrieved neighbors `k`
- disable optional SCARF pretraining
- compare cross-entropy and focal loss for the classifier head

The most important ablations are retrieval-related. Optional pretraining and encoder block variants should be treated as secondary.


## 7. Metrics

The primary metrics should be:

- macro-F1
- macro precision
- macro recall
- per-class F1

Secondary metrics should include:

- accuracy
- weighted-F1
- confusion matrix
- benign false positive rate
- minority-class precision and recall

For adaptive memory experiments, additionally report:

- macro-F1 per session
- session-to-session performance change
- number of memory insertions
- number of evictions
- final memory size
- pinned vs writeback memory counts


## 8. Minority-Class Analysis

The paper should separately analyze the most difficult minority classes. Based on current results, the important classes are:

- `Bot`
- `Infiltration`
- `Web Attack - Brute Force`
- `Web Attack - Sql Injection`
- `Web Attack - XSS`

For each, report:

- precision
- recall
- F1
- confusion patterns

This is necessary because high overall performance can hide poor behavior on rare attacks.


## 9. Current Result Snapshot

The current observed `RAG-NIDS` run produced:

- `accuracy = 0.9791`
- `macro precision = 0.6487`
- `macro recall = 0.8848`
- `macro-F1 = 0.6812`
- `weighted-F1 = 0.9861`

This pattern suggests that the model achieves strong overall performance and strong recall across many attack classes, but precision is weak for some rare labels. This should be discussed as a retrieval sensitivity tradeoff rather than ignored.

The current confusion matrix indicates:

- high performance on majority and common attack classes
- strong recall but low precision for some rare classes
- false positives from benign traffic into rare attack categories
- instability for extremely small-support classes

This motivates memory curation, adaptive refresh, and careful false positive analysis.


## 10. Expected Tables

### Table 1: Main Static Split Results

Rows:

- `Logistic Regression`
- `Random Forest`
- `ExtraTrees` or `XGBoost`
- `MLP`
- `Autoencoder + classifier`
- `Encoder + linear head`
- `Raw-feature kNN`
- `Embedding-space kNN`
- `Static RAG-NIDS`

Columns:

- accuracy
- weighted-F1
- macro-F1
- macro precision
- macro recall
- benign false positive rate

### Table 2: Per-Class F1

Rows:

- all class labels

Columns:

- selected baselines
- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`, if available

### Table 3: Reduced Training Data

Rows:

- `40%`
- `50%`
- `80%`
- `100%`

Columns:

- selected baselines
- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`, if available
- macro-F1
- weighted-F1
- memory size

### Table 4: Session-Wise Evaluation

Rows:

- session IDs

Columns:

- static memory macro-F1
- adaptive memory macro-F1
- benign false positive rate
- memory insertions
- memory evictions
- writeback memory size

### Table 5: Ablation Study

Rows:

- no retrieval
- raw-feature kNN
- embedding-space kNN
- retrieval-conditioned head
- static memory
- adaptive memory

Columns:

- macro-F1
- macro precision
- macro recall
- minority-class average F1


## 11. Expected Figures

Suggested figures:

- architecture diagram of `RAG-NIDS`
- embedding and retrieval pipeline diagram
- confusion matrix heatmap
- macro-F1 vs retained training fraction
- session-wise macro-F1 curve
- memory size over sessions
- retrieved-neighbor explanation example


## 12. Result Discussion Template

The results should be discussed around the following claims:

1. Retrieval-conditioned inference improves over non-retrieval baselines if `Static RAG-NIDS` outperforms `MLP`, `Encoder + linear head`, and simple kNN variants on macro-F1.
2. Learned embedding retrieval is useful if embedding-space retrieval outperforms raw-feature kNN.
3. Cross-attention fusion is useful if full `RAG-NIDS` outperforms embedding-space kNN majority voting.
4. Adaptive memory is useful if session-wise performance is more stable than static memory or if similar performance is preserved with lower retained training data.
5. The main remaining weakness is false positives into rare classes if minority precision remains low.


## 13. Draft-Ready Experiment Paragraph

`We evaluate RAG-NIDS on the CIC-IDS2017 multiclass intrusion detection task using a shared preprocessing pipeline across all baselines. Because the dataset is highly imbalanced, we emphasize macro-F1, macro precision, macro recall, per-class F1, and benign false positive rate rather than relying only on accuracy. The proposed model is compared against classical tabular baselines, neural baselines, raw-feature nearest-neighbor retrieval, embedding-space nearest-neighbor retrieval, and an encoder-only classifier. We further evaluate reduced training-data settings and, where applicable, session-wise adaptive memory refresh to test whether retrieval memory can preserve performance under bounded retained data and evolving traffic conditions.`

