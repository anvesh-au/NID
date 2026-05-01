# Introduction Draft: RAG-NIDS

## Draft Introduction

Network intrusion detection systems are required to distinguish benign traffic from a wide range of attack behaviors under highly imbalanced and evolving traffic conditions. Flow-based intrusion detection datasets such as `CIC-IDS2017` reflect this difficulty: benign traffic dominates the distribution, several attack classes are rare, and some attack categories contain only a small number of examples. Under these conditions, overall accuracy can be misleading. A classifier may perform well on majority classes while failing to detect or correctly classify minority attack types that are operationally important.

Most supervised intrusion detection methods formulate the task as standard multiclass classification. In this setting, a model maps each flow directly to a label using fixed learned parameters. While this approach is simple and effective for common classes, it has two important limitations. First, the learned decision boundary can be dominated by majority traffic patterns, making rare attacks difficult to preserve. Second, a purely parametric classifier does not directly expose the historical examples that support a prediction, limiting its usefulness for auditability and analyst trust.

In practical security workflows, intrusion decisions are often case-based. A suspicious flow is not only classified in isolation; it is compared against known traffic patterns, prior incidents, and similar observed behaviors. This motivates a retrieval-augmented formulation of intrusion detection. Rather than relying only on a fixed classifier, a retrieval-augmented IDS can classify a query flow using both its learned representation and a memory of similar labeled flows.

This work proposes `RAG-NIDS`, a single-stage retrieval-augmented network intrusion detection framework for imbalanced multiclass flow classification. The method first learns a normalized embedding space for network flows using supervised contrastive learning with an auxiliary classification objective. Labeled training flows are then stored in a FAISS-based retrieval memory. At inference time, the model retrieves the nearest historical flows for a query and uses a cross-attention classifier head to combine the query embedding with retrieved neighbor embeddings and labels.

This architecture differs from both conventional classifiers and simple nearest-neighbor methods. Unlike a plain neural classifier, `RAG-NIDS` uses an external memory during inference. Unlike raw-feature kNN, it retrieves in a learned embedding space optimized for class-aware similarity. Unlike majority-vote retrieval, it uses a trainable attention mechanism that can weigh retrieved evidence differently for each query.

The retrieval memory also provides a path toward adaptive operation. The current architecture supports a memory layer with pinned entries, writeback entries, time-based expiration, and bounded writeback capacity. This makes it possible to refresh stored examples over time and study whether retrieval memory can preserve performance under reduced retained training data and changing traffic sessions.

The central hypothesis of this work is that retrieval-conditioned classification over learned flow embeddings can improve class-balanced intrusion detection and provide useful exemplar-based evidence for predictions. The paper evaluates this hypothesis using macro-F1, per-class F1, macro precision, macro recall, confusion matrices, reduced training-data experiments, and session-wise memory evaluation where available.


## Motivation Points to Preserve

The introduction should preserve the following motivations:

- intrusion detection is highly imbalanced
- accuracy alone is not enough
- rare attacks matter operationally
- static classifiers can be dominated by majority classes
- retrieval gives local exemplar evidence
- adaptive memory can support evolving traffic conditions


## Gap Statement

Use this as the core gap:

`Existing IDS methods are commonly evaluated as static classifiers, but this formulation does not fully exploit the local similarity structure of flow traffic or provide explicit exemplar evidence at inference time. In addition, static models are poorly aligned with traffic environments where benign and attack behaviors can evolve across sessions.`


## Proposed Direction

Use this as the transition into the method:

`We address this gap by formulating intrusion detection as retrieval-augmented classification over learned flow embeddings. The proposed system stores labeled historical flows in an indexed memory and conditions each prediction on the query flow together with its retrieved neighbors.`


## Contribution List

A concise contribution list for the final introduction:

1. We propose `RAG-NIDS`, a single-stage retrieval-augmented intrusion detection framework that combines learned flow embeddings, FAISS-based memory retrieval, and a cross-attention classifier head.
2. We introduce a retrieval-conditioned classification mechanism that uses both neighbor embeddings and neighbor labels, making predictions dependent on local exemplar evidence rather than only fixed model parameters.
3. We study the method under imbalanced multiclass IDS evaluation using macro-F1, per-class F1, macro precision, macro recall, and confusion matrix analysis.
4. We examine reduced training-data and adaptive memory settings to evaluate whether refreshable retrieval memory can preserve performance under bounded retained data and evolving traffic sessions.
5. We provide exemplar-based prediction evidence through retrieved neighbor labels and similarity scores.


## Conservative Claim Version

Use this version if results are mixed or if baseline comparison is still incomplete:

`The results suggest that retrieval-augmented classification is a promising direction for imbalanced intrusion detection, particularly for improving minority-class sensitivity and providing interpretable neighbor evidence. The method also exposes a practical tradeoff between rare-class recall and false positives, motivating further study of memory curation and adaptive refresh policies.`


## Stronger Claim Version

Use this version only if final experiments support it:

`Across static, reduced-data, and session-wise evaluations, RAG-NIDS improves class-balanced intrusion detection compared with non-retrieval baselines while preserving high overall performance. The adaptive memory mechanism further improves stability under changing traffic conditions and enables competitive performance with reduced retained training data.`


## Draft Abstract Candidate

`Network intrusion detection is often evaluated as a static multiclass classification problem, despite strong class imbalance, rare attack categories, and evolving traffic distributions. We propose RAG-NIDS, a retrieval-augmented intrusion detection framework that classifies each flow using both a learned query representation and similar labeled examples retrieved from an indexed memory. The method learns normalized flow embeddings using supervised contrastive and auxiliary classification objectives, stores labeled embeddings in a FAISS retrieval index, and applies a cross-attention classifier head over query and neighbor tokens. The retrieval layer also supports bounded writeback and eviction, enabling adaptive memory refresh across traffic sessions. Experiments on CIC-IDS2017 compare RAG-NIDS with classical, neural, and retrieval baselines using macro-F1, per-class F1, macro precision, macro recall, and confusion matrix analysis. The proposed formulation provides a practical path toward class-balanced and interpretable intrusion detection with refreshable exemplar memory.`


## Opening Paragraph Alternative

`Intrusion detection systems operate in environments where attack traffic is sparse, heterogeneous, and continually changing. These properties make flow-based intrusion detection difficult to model as ordinary multiclass classification. In datasets such as CIC-IDS2017, benign flows dominate the distribution while several attack categories contain relatively few examples. As a result, high accuracy can mask weak performance on rare but important attacks. This motivates evaluation protocols and model designs that prioritize class-balanced behavior, minority-class detection, and adaptability over aggregate accuracy alone.`


## Final Introduction Checklist

Before using this as the final paper introduction, fill in:

- exact final baseline names
- exact final dataset split protocol
- exact final result summary
- whether adaptive memory experiments are fully executed or presented as architecture support plus preliminary evaluation
- final wording for the contribution list based on completed experiments

