# Proposed Solution: RAG-NIDS Architecture

## Purpose of This Document

This document is a focused reference for the proposed solution used in the current solution branch of the project. It is written in the style of a paper's `Proposed Solution` or `Methodology` section and is intended to help:

- the project author while drafting the paper
- an LLM that needs architecture-specific context to generate a first draft

This document describes the actual single-stage `RAG-NIDS` pipeline implemented in the current branch, with emphasis on:

- flow preprocessing
- flow embedding learning
- retrieval memory/index
- retrieval-conditioned classification
- training strategy
- adaptive memory support


## Solution Overview

The proposed system is a single-stage retrieval-augmented intrusion detection framework for multiclass flow classification. The core idea is to classify a network flow using not only a learned representation of the query flow, but also a set of similar labeled flows retrieved from an indexed memory.

The architecture consists of four main stages:

1. preprocess tabular flow records from `CIC-IDS2017`
2. learn a discriminative embedding for each flow using a supervised contrastive encoder
3. store encoded training flows in a FAISS-based retrieval index
4. classify a query flow by combining its embedding with the embeddings and labels of retrieved neighbors using a cross-attention classifier head

This makes the system retrieval-augmented rather than purely parametric. The classifier does not rely only on a fixed decision boundary learned during training. Instead, it conditions the prediction on a local neighborhood of similar historical flows at inference time.


## Problem Setup

Let the dataset be:

- `X in R^(N x F)`: `N` flows with `F` numeric features
- `y in {1, ..., C}`: multiclass intrusion labels over `C` traffic classes

The goal is to learn a model that maps a query flow `x_q` to a class label `y_q`, while explicitly using a memory of historical labeled flows.

Unlike a conventional classifier that predicts from `x_q` alone, the proposed method first maps `x_q` into an embedding space, retrieves the `k` most similar stored flows, and then predicts using both the query representation and the retrieved evidence.


## Data Preprocessing

The preprocessing pipeline follows the implementation in the current branch.

### Input format

The system reads all CSV files from the `CIC-IDS2017` directory and merges them into a single tabular dataset.

### Feature cleanup

Columns that are likely to cause leakage or are unsuitable for direct modeling are removed, including identifiers and address fields such as:

- `Flow ID`
- `Source IP`
- `Destination IP`
- `Source Port`
- `Destination Port`
- `Timestamp`
- `SimillarHTTP`
- `Fwd Header Length.1`

### Missing and invalid values

The preprocessing stage:

- replaces `+inf` and `-inf` with `NaN`
- drops rows containing invalid values

### Label cleanup

The `Label` column is stripped and standardized as the multiclass target.

### Protocol handling

If the `Protocol` field is present and has low cardinality, it is one-hot encoded.

### Feature normalization

All numeric features are standardized using `StandardScaler`.

### Label encoding

String labels are mapped to integer class IDs using `LabelEncoder`.

### Subsampling behavior

The loader supports a controlled subsample mode for smaller experiments. The subsampling logic is class-aware and enforces a per-class floor so that minority attack classes are not accidentally collapsed to near-zero representation.


## Architecture Summary

The architecture has three learned or semi-learned components:

1. `FlowEncoder`
2. `FlowIndex`
3. `CrossAttentionHead`

These are combined inside the end-to-end `RAGNIDS` inference pipeline.


## Component 1: Flow Encoder

### Objective

The encoder maps each standardized flow feature vector into a dense embedding space where semantically similar flows are close to one another.

### Structure

The current branch uses a residual GLU-based MLP encoder rather than a plain feedforward MLP.

The encoder consists of:

- an input projection from raw feature dimension to a hidden dimension
- a stack of residual `GLUBlock`s
- a final linear projection into the embedding space
- `L2` normalization of the output embedding

### GLU block design

Each `GLUBlock` applies:

- `LayerNorm`
- linear projection to `2d`
- split into value and gate branches
- sigmoid gating
- residual connection
- dropout

This design allows the encoder to perform feature-wise gating while preserving stable optimization through residual paths.

### Embedding output

For an input flow `x`, the encoder produces:

`z = f_theta(x) in R^d`

where `d` is the embedding dimension.

The final embedding is normalized to unit length, which is consistent with similarity search using inner product and helps stabilize contrastive training.


## Component 2: Retrieval Memory and FAISS Index

### Objective

The retrieval module stores encoded historical flows so that each query can access a local memory of similar examples at inference time.

### Stored items

For each stored flow, the index maintains:

- flow embedding
- class label
- timestamp
- source flag indicating whether the sample is part of the original pinned memory or added later through writeback

### Index type

The current implementation supports:

- `IndexFlatIP` for exact inner-product search
- `IndexHNSWFlat` for approximate search when enabled

The FAISS index can run on:

- CPU
- CUDA, if a GPU-enabled FAISS build is available

### Similarity search

Given a query embedding `z_q`, the index returns:

- top-`k` similarity scores
- neighbor indices
- neighbor labels

The system uses the learned embedding space as the retrieval space. This is important: retrieval is not done over raw features but over learned, normalized flow embeddings.

### Memory reconstruction

After retrieval, the model reconstructs or gathers the neighbor embeddings from cached stored vectors so they can be fed into the classifier head.


## Component 3: Retrieval-Conditioned Cross-Attention Head

### Objective

The classifier head predicts the class of a query flow using both:

- the query embedding
- retrieved neighbor embeddings
- retrieved neighbor labels

This is what makes the method retrieval-conditioned rather than retrieval-only.

### Inputs

For each batch, the head receives:

- `query_z`: query embedding of shape `(B, D)`
- `neighbor_z`: retrieved neighbor embeddings of shape `(B, K, D)`
- `neighbor_labels`: integer labels of shape `(B, K)`

### Label-aware memory tokens

Neighbor labels are embedded using a learnable label embedding table.

For each retrieved neighbor, the model concatenates:

- neighbor embedding
- corresponding label embedding

This concatenated vector is projected back into the common embedding dimension. The result is a memory token that represents both feature-level and class-level neighbor information.

### Cross-attention mechanism

The query embedding is treated as the target token, and the retrieved neighbor tokens are treated as memory tokens. A transformer decoder performs cross-attention from the query to the retrieved memory.

This allows the model to:

- focus on the most relevant retrieved examples
- weigh neighbors differently
- use both neighbor content and neighbor labels when forming the decision

### Output layer

The decoder output corresponding to the query is passed through a final linear classifier to produce multiclass logits.

### Why this is better than simple kNN voting

This design is more expressive than pure nearest-neighbor voting because:

- it does not assume all neighbors contribute equally
- it can learn query-specific weighting over the memory
- it conditions on both neighbor labels and embeddings
- it remains trainable end-to-end at the head level


## End-to-End Inference Pipeline

The full `RAGNIDS` pipeline performs the following steps for a query batch:

1. encode the query flows into normalized embeddings
2. retrieve the top-`k` nearest neighbors from the FAISS index
3. gather the corresponding neighbor embeddings and labels
4. pass the query embedding and neighbor tokens to the cross-attention head
5. produce class logits
6. convert logits to probabilities and final predictions

The prediction object can also expose:

- predicted label
- confidence
- retrieved neighbor IDs
- retrieved neighbor labels
- retrieved similarity scores

This makes the system naturally explainable at the instance level.


## Training Strategy

The model is trained in two stages rather than fully end-to-end over all components at once.

### Stage 1: Encoder training

The encoder is trained first using a combination of:

- supervised contrastive loss
- auxiliary cross-entropy loss

#### Why supervised contrastive learning

The contrastive objective encourages flows from the same class to cluster together and flows from different classes to separate in embedding space. This is especially important because the retrieval module depends directly on the quality of the learned neighborhood structure.

#### Why add auxiliary cross-entropy

A small auxiliary classifier is attached to the encoder during training. The cross-entropy term stabilizes optimization and encourages the embeddings to remain discriminative for multiclass classification under heavy imbalance.

#### Sampling strategy

The encoder is trained using a `WeightedRandomSampler`, which oversamples minority classes through inverse-frequency weighting. This is important because the embedding space should not be dominated entirely by `BENIGN` or other large classes.

#### Optional pretraining

The current branch also supports optional SCARF self-supervised pretraining before supervised contrastive fine-tuning. This is an optional enhancement, not the core identity of the method.

### Stage 2: Retrieval-conditioned head training

Once the encoder is trained, the training flows are encoded and inserted into the FAISS memory.

Then:

- the encoder is frozen
- the cross-attention head is trained on top of retrieval outputs

For each training batch:

1. the query flows are encoded
2. neighbors are retrieved from the memory
3. self-neighbor exclusion is applied during head training to reduce trivial memorization
4. the cross-attention head predicts logits
5. the head is optimized using either standard cross-entropy or focal loss

This staged training avoids destabilizing the embedding geometry while the retrieval-conditioned head is learning to use the memory.


## Loss Functions

### Encoder loss

The encoder optimization objective is:

`L_encoder = lambda_supcon * L_supcon + lambda_ce * L_ce`

where:

- `L_supcon` is supervised contrastive loss
- `L_ce` is auxiliary cross-entropy

### Head loss

The head is trained using:

- standard cross-entropy, or
- focal loss when class imbalance handling needs to be stronger


## Validation and Early Stopping

The current branch supports explicit early stopping for both training stages.

### Encoder early stopping

A validation split is carved out from the training data. The encoder is monitored using validation total loss.

### Head early stopping

The retrieval-conditioned head is monitored using validation macro-F1.

This is aligned with the paper's evaluation priorities, because macro-F1 is more informative than accuracy under severe class imbalance.


## Adaptive Memory Support

The retrieval database in the current branch is not a passive static index. The `FlowIndex` implementation includes support for memory refresh through:

- writeback insertion
- time-based expiration
- bounded writeback capacity
- index rebuilding after eviction

### Pinned vs writeback entries

Stored memory entries can come from two sources:

- `PINNED`: original indexed training flows
- `WRITEBACK`: new flows inserted later through adaptive update logic

### Writeback behavior

The index can insert new examples into memory when certain conditions are met, such as:

- the sample is considered an attack
- the prediction confidence exceeds a threshold

This creates a path for the system to evolve its memory over time.

### Eviction behavior

The memory supports:

- TTL-based expiration for writeback entries
- hard capacity limits on writeback memory
- rebuild-based removal of expired or excess entries

### Important implementation note

The current branch already implements refresh-capable memory behavior at the index layer. However, the top-level training and evaluation driver currently focuses on the main static train-index-evaluate loop. A full session-wise deployment loop that exercises writeback and eviction over time is a natural extension of the current architecture rather than the only currently executed experiment path.

This distinction should be stated clearly in the paper if needed.


## Why the Architecture Fits the IDS Setting

This design is appropriate for intrusion detection for several reasons.

### Local similarity matters

Attack flows may not be best modeled by only a global decision boundary. Retrieval lets the system use local evidence from similar historical cases.

### Minority classes benefit from memory

Rare attack types are often underrepresented in purely parametric training. A retrieval mechanism can expose relevant neighbors directly during inference.

### Interpretability is valuable

Security decisions benefit from case-based evidence. The architecture naturally provides neighboring exemplars and similarity scores.

### Adaptation is practical

A refreshable memory allows the deployed system to evolve more cheaply than a full end-to-end retraining pipeline.


## Mathematical Summary

At a high level, for a query flow `x_q`:

1. compute query embedding

`z_q = f_theta(x_q)`

2. retrieve top-`k` nearest stored flows

`N_q = {(z_i, y_i)}_(i=1)^k = Retrieve(z_q, M)`

3. build memory tokens using neighbor embeddings and label embeddings

`m_i = W_m [z_i || e(y_i)]`

4. use cross-attention to condition the query on the retrieved memory

`h_q = CrossAttention(z_q, {m_i}_{i=1}^k)`

5. predict class logits

`p(y | x_q, M) = softmax(W_c h_q)`

This formulation makes the label prediction explicitly dependent on both the query and the external memory `M`.


## Distinguishing Characteristics of the Proposed Solution

The proposed solution should be distinguished from simpler alternatives.

### Compared with a plain MLP classifier

- uses explicit memory at inference time
- is not purely parametric
- supports exemplar-based reasoning

### Compared with raw-feature kNN

- retrieves in a learned embedding space
- uses a trainable retrieval-conditioned classifier instead of majority voting

### Compared with embedding plus linear head

- incorporates neighbor context dynamically at inference time
- can reweight retrieved evidence per query

### Compared with static memory lookup

- includes support for memory refresh through writeback and eviction


## What to Emphasize in the Paper

When converting this into a paper section, emphasize:

- retrieval is a core inference mechanism, not a post-processing step
- the learned embedding space is optimized specifically to support retrieval quality
- the classifier uses both neighbor content and neighbor labels
- the memory is designed to be refreshable and bounded
- the architecture is intended for imbalanced and evolving traffic

Do not over-emphasize:

- training utilities such as MLflow
- implementation-side convenience features
- optional pretraining as the main contribution


## Suggested Paper Section Structure

This document can be converted into a `Proposed Solution` section using the following subsection layout:

1. Problem Formulation
2. Data Preprocessing
3. Flow Representation Learning
4. Retrieval Memory Construction
5. Retrieval-Conditioned Classification
6. Training Strategy
7. Adaptive Memory Refresh Mechanism
8. Inference and Explanation


## Draft-Ready Summary Paragraph

Use this paragraph as drafting material:

`The proposed solution, RAG-NIDS, is a single-stage retrieval-augmented intrusion detection architecture for tabular network traffic classification. First, each flow is transformed into a normalized embedding using a residual GLU-based encoder trained with supervised contrastive and auxiliary cross-entropy objectives. The resulting embeddings of labeled training flows are stored in a FAISS-based memory index. At inference time, the system retrieves the top-k nearest neighbors of a query flow in embedding space and combines the query representation with the retrieved neighbor embeddings and labels through a cross-attention classifier head. This design enables the model to make predictions using both learned global structure and local exemplar evidence. In addition, the memory layer supports bounded writeback and eviction, providing a foundation for adaptive refresh under evolving traffic conditions.` 

