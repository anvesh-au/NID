# Method Section Draft: RAG-NIDS

## 1. Method Overview

We propose `RAG-NIDS`, a single-stage retrieval-augmented network intrusion detection framework for multiclass flow classification. The method combines a learned flow encoder, a retrieval memory built over labeled historical flows, and a retrieval-conditioned classifier head. Instead of predicting from the input flow alone, `RAG-NIDS` first maps the query flow into an embedding space, retrieves similar flows from memory, and then conditions the final classification decision on both the query embedding and the retrieved evidence.

The architecture is designed for imbalanced intrusion detection settings in which purely parametric classifiers can become dominated by majority classes such as `BENIGN`, while rare attack classes remain difficult to separate. By explicitly using local neighbors in a learned embedding space, the proposed framework aims to improve class-balanced recognition and provide exemplar-based evidence for each prediction.


## 2. Problem Formulation

Let the dataset be denoted by:

- `X = {x_i}_{i=1}^N`, where each `x_i in R^F` is a flow-level feature vector
- `Y = {y_i}_{i=1}^N`, where each `y_i in {1, ..., C}` is a class label over `C` traffic categories

Given a query flow `x_q`, the goal is to predict its label `y_q` by using both:

- a learned representation of the query flow
- a memory of previously observed labeled flows

Unlike a conventional classifier that estimates `p(y_q | x_q)` directly from fixed model parameters, our method estimates the class distribution conditioned on a retrieval memory `M`:

`p(y_q | x_q, M)`

where `M` contains stored embeddings and labels of historical flows.


## 3. Data Preprocessing

All experiments are conducted on flow-based tabular traffic data derived from `CIC-IDS2017`. Each CSV file is loaded and merged into a unified dataset. To reduce leakage and remove non-generalizable identifiers, fields such as flow IDs, IP addresses, ports, timestamps, and known problematic columns are discarded. Infinite values are converted to missing values and removed, and only numeric features are retained for modeling.

If the `Protocol` field is present and has small cardinality, it is one-hot encoded. The remaining numeric features are standardized using `StandardScaler`, and class labels are encoded as integer IDs. The preprocessing pipeline also supports class-aware subsampling for smaller-scale experiments while enforcing a minimum number of examples per class so that minority attack categories are not collapsed during sampling.


## 4. Flow Representation Learning

### 4.1 Encoder Architecture

Each input flow is mapped to a dense embedding using a residual GLU-based multilayer perceptron. Let `f_theta` denote the encoder. For an input flow `x`, the encoder produces:

`z = f_theta(x), z in R^d`

where `d` is the embedding dimension.

The encoder consists of:

- an input projection from feature space into a hidden representation
- a stack of residual gated linear unit blocks
- a final projection into the embedding space
- `L2` normalization of the final embedding

The GLU-based residual blocks allow feature-wise gating while preserving stable gradient flow. The `L2` normalization is important because similarity search is performed using inner-product comparisons in the embedding space.

### 4.2 Supervised Contrastive Objective

The encoder is trained to produce embeddings that preserve class structure. To do this, we optimize the encoder using supervised contrastive learning together with an auxiliary classification loss. The supervised contrastive term pulls examples of the same class closer together and pushes examples from different classes apart. This is particularly important because retrieval quality depends directly on whether the embedding space forms semantically meaningful neighborhoods.

An auxiliary linear classification head is attached during encoder training to stabilize optimization and preserve class-discriminative information. The encoder objective is:

`L_enc = lambda_supcon * L_supcon + lambda_ce * L_ce`

where:

- `L_supcon` is the supervised contrastive loss
- `L_ce` is the auxiliary cross-entropy loss
- `lambda_supcon` and `lambda_ce` control the contribution of each term

### 4.3 Imbalance-Aware Sampling

To reduce the dominance of majority classes during encoder training, mini-batches are formed using inverse-frequency weighted sampling. This ensures that minority attack classes are presented more frequently during optimization and helps prevent the embedding space from collapsing around the majority `BENIGN` class.

### 4.4 Optional Self-Supervised Pretraining

The implementation supports optional SCARF-style self-supervised pretraining before supervised contrastive fine-tuning. This step is treated as an optional enhancement rather than the core contribution of the method.


## 5. Retrieval Memory Construction

After training the encoder, all retained training flows are encoded into the embedding space and inserted into a retrieval memory index. Let the memory be defined as:

`M = {(z_i, y_i)}_{i=1}^n`

where `z_i` is the stored embedding of a historical flow and `y_i` is its class label.

The current implementation uses a FAISS-based index for efficient nearest-neighbor retrieval. Exact retrieval is supported through `IndexFlatIP`, and approximate retrieval can optionally be enabled through HNSW. The index can operate on CPU or GPU depending on system configuration.

Retrieval is performed in the learned embedding space rather than the raw feature space. This is important because the encoder is explicitly optimized to organize flows in a semantically meaningful geometry for class-aware neighborhood search.


## 6. Retrieval-Conditioned Classification

### 6.1 Neighbor Retrieval

Given a query flow `x_q`, the system first computes the query embedding:

`z_q = f_theta(x_q)`

The top-`k` nearest neighbors are then retrieved from memory:

`N_q = Retrieve(z_q, M) = {(z_(q,1), y_(q,1)), ..., (z_(q,k), y_(q,k))}`

where each neighbor consists of:

- its stored embedding
- its class label
- its similarity to the query

### 6.2 Label-Aware Memory Tokens

The classifier does not use the retrieved neighbors through simple majority voting. Instead, each neighbor is converted into a memory token that combines both feature and label information. For each neighbor label `y_(q,i)`, a learnable label embedding `e(y_(q,i))` is obtained. This label embedding is concatenated with the neighbor embedding `z_(q,i)` and then projected back into the common embedding dimension:

`m_i = W_m [z_(q,i) || e(y_(q,i))]`

where `m_i` is the memory token for the `i`-th retrieved neighbor.

This design allows the model to use the retrieved labels as part of the reasoning process rather than treating neighbors as unlabeled context.

### 6.3 Cross-Attention Fusion

To produce a final decision, the query embedding is treated as the target token and the retrieved memory tokens are treated as context. A transformer decoder performs cross-attention from the query to the memory:

`h_q = CrossAttention(z_q, {m_i}_{i=1}^k)`

This enables the model to:

- reweight neighbors differently for each query
- use both similarity structure and neighbor labels
- form a prediction from local evidence rather than only a global boundary

The final class logits are then computed as:

`l_q = W_c h_q + b_c`

and the predicted class probabilities are obtained through:

`p(y_q | x_q, M) = softmax(l_q)`


## 7. Training Procedure

The training process is divided into two stages.

### 7.1 Stage 1: Encoder Optimization

The encoder is trained first using the contrastive-plus-auxiliary objective described above. Early stopping can be applied using a held-out validation split from the training data, monitored through validation loss. If enabled, self-supervised pretraining is performed before this supervised stage.

### 7.2 Stage 2: Retrieval-Conditioned Head Optimization

Once the encoder is trained, all retained training flows are encoded and inserted into the retrieval memory. The encoder is then frozen, and the retrieval-conditioned cross-attention head is optimized separately.

For each training batch:

1. the query samples are encoded using the frozen encoder
2. neighbors are retrieved from the memory
3. self-neighbors can be excluded to avoid trivial memorization during training
4. the cross-attention head produces class logits
5. the head is optimized using either cross-entropy or focal loss

Validation macro-F1 is used for early stopping of the head when enabled.

This staged training strategy avoids instability that could arise if the retrieval space were continuously changing while the classifier head was simultaneously learning to attend over it.


## 8. Adaptive Memory Mechanism

In addition to supporting a static memory index, the implemented retrieval layer supports adaptive memory maintenance. Stored items are divided into:

- pinned entries originating from the initial indexed training set
- writeback entries inserted later during operation

The memory interface supports:

- insertion of new high-confidence attack samples into writeback memory
- time-based expiration of writeback entries
- bounded capacity for writeback memory
- index rebuilding after eviction

Conceptually, this turns the memory into a refreshable retrieval database rather than a fully static reference set. This feature is central to the broader project motivation because it creates a path for the system to remain aligned with more recent traffic patterns and to operate under bounded retained-data conditions.

At the current stage of implementation, the adaptive writeback and eviction logic is available in the index layer, while the main training script primarily evaluates the core static train-index-predict pipeline. Session-wise experiments that explicitly exercise memory refresh over time should therefore be described as an extension built on top of the proposed architecture.


## 9. Inference and Explanation

At inference time, the system outputs not only a predicted class and confidence score, but also the IDs, labels, and similarity scores of the retrieved neighbors used for the decision. This provides an exemplar-based explanation mechanism. In security settings, such case-based evidence is useful because an analyst can inspect whether a prediction is supported by semantically similar historical flows rather than relying only on opaque logits.


## 10. Discussion of Design Rationale

The proposed architecture is motivated by three considerations.

First, intrusion detection is highly imbalanced, and a purely parametric classifier may overfit to dominant traffic classes. By explicitly retrieving neighbors from memory, the model can use local evidence for rare attack types that might otherwise be overwhelmed by global class priors.

Second, learned embeddings provide a structured similarity space that is more suitable for neighbor search than raw tabular features. The supervised contrastive objective aligns the representation learning stage with the retrieval stage.

Third, a refreshable memory is operationally attractive because traffic patterns can evolve over time. While the core paper focuses on the single-stage `RAG-NIDS` architecture, the memory design also supports future or extended evaluation under multi-session and drift-aware conditions.


## 11. Distinction from Alternative Approaches

The proposed solution should be distinguished clearly from several nearby classes of methods.

Compared with a conventional MLP classifier, `RAG-NIDS` uses explicit external memory during inference.

Compared with raw-feature kNN, `RAG-NIDS` retrieves in a learned embedding space and uses a trainable retrieval-conditioned classifier rather than majority voting.

Compared with an encoder followed by a linear classifier, `RAG-NIDS` conditions each decision on query-specific local neighbors.

Compared with a static memory lookup system, `RAG-NIDS` includes support for bounded writeback and eviction, making it compatible with adaptive memory refresh.


## 12. Concise Draft-Ready Closing Paragraph

`In summary, RAG-NIDS formulates intrusion detection as retrieval-augmented classification over learned flow embeddings. A residual GLU-based encoder maps flows into a normalized embedding space, a FAISS memory index stores labeled historical examples, and a cross-attention head predicts labels by conditioning on both the query embedding and the retrieved neighbors. This design combines discriminative representation learning with local exemplar evidence, offering a practical architecture for imbalanced multiclass intrusion detection and providing a foundation for adaptive memory refresh under evolving traffic conditions.`

