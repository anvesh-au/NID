# RAG-NIDS Paper Knowledge Reference

## Purpose of This Document

This document is a project knowledge reference for writing a paper about the current `RAG-NIDS` approach in this repository. It is intended to be used by:

- the project author as a concise but complete reference while planning the paper
- an LLM as source context for drafting an introduction, method, experiments, results discussion, and abstract

This document reflects the direction agreed in discussion:

- the paper should focus on the single-stage `RAG-NIDS` approach
- the paper should not center on the alternative two-stage architecture explored in another branch
- the main novelty is retrieval-augmented intrusion detection with an adaptive memory/index, not just a stronger classifier


## Project Snapshot

### Working title

Suggested working title:

`RAG-NIDS: Retrieval-Augmented Network Intrusion Detection for Imbalanced and Evolving Traffic`

Alternative title:

`Adaptive Retrieval-Augmented Intrusion Detection with Refreshable Memory for CIC-IDS2017`


## One-Paragraph Project Summary

This project develops a retrieval-augmented network intrusion detection system for tabular flow-based traffic classification on `CIC-IDS2017`. Instead of relying only on a static parametric classifier, the model learns flow embeddings, stores labeled flow representations in an indexed memory, retrieves similar historical flows at inference time, and conditions prediction on both the query sample and its retrieved neighbors. The intended advantage is improved class-balanced intrusion detection, especially for minority attack classes, together with exemplar-based interpretability and an adaptive memory mechanism that can refresh stored examples over time through eviction and insertion. The adaptive memory mechanism is also expected to help the model remain useful under session shifts and attack/benign distribution drift without requiring full retraining from scratch.


## What the Paper Is About

The paper is about using retrieval augmentation as a first-class inference mechanism for intrusion detection.

It is not just about:

- applying a deep learning model to `CIC-IDS2017`
- improving a classifier with minor engineering changes
- getting high overall accuracy on an imbalanced benchmark

It is specifically about:

- learning a representation of network flows
- using an external memory of labeled historical examples
- retrieving similar flows during inference
- conditioning classification on retrieved evidence
- refreshing that memory over time to support smaller retained training sets and evolving traffic


## Core Problem Motivation

### Practical IDS problem

Network intrusion detection on datasets such as `CIC-IDS2017` is highly imbalanced and operationally difficult.

Key difficulties:

- `BENIGN` traffic dominates the dataset
- several attack classes are rare
- some rare attack types have very few examples
- a model can achieve high accuracy while still performing poorly on minority attacks
- attack traffic and benign traffic can shift over time across sessions
- a static model trained once may become stale

### Why plain multiclass classification is not enough

Many IDS papers treat the problem as standard multiclass classification. This is insufficient because:

- global decision boundaries are often dominated by majority classes
- rare attacks can be suppressed by the dominant `BENIGN` class
- static classifiers lack direct access to exemplar-level evidence at inference time
- adapting to new traffic patterns often requires retraining the model or rebuilding the pipeline

### Why retrieval is a natural fit

The central intuition behind this project is that intrusion detection should not rely only on a parametric decision boundary. Instead, it should use a memory of known traffic patterns.

Retrieval is appropriate for IDS because:

- network flows often form local similarity structure
- rare attack classes may be better identified by comparing a query to similar known examples
- exemplar-based evidence is useful in security settings
- retrieval makes decisions more inspectable than a pure black-box classifier
- a refreshable memory enables adaptation without fully retraining the model after each change


## Main Research Direction

The paper should focus on the following research direction:

`Can retrieval-augmented classification over learned flow embeddings improve class-balanced network intrusion detection and remain effective under reduced retained training data and evolving traffic distributions?`


## Main Research Questions

### RQ1: Retrieval benefit

Does retrieval-augmented inference improve intrusion detection compared with purely parametric baselines and embedding-only baselines?

### RQ2: Class imbalance

Does retrieval help preserve minority-class performance better than standard models under severe class imbalance?

### RQ3: Adaptive memory

Can a refreshable memory with eviction and insertion maintain competitive performance even when only a smaller portion of the original training data is retained?

### RQ4: Session drift and data evolution

Does the adaptive memory mechanism help the IDS remain robust across multiple sessions and evolving traffic patterns without full retraining?

### RQ5: Interpretability

Does retrieval provide useful case-based evidence that improves auditability of predictions?


## Central Thesis / Claim

The main claim the paper should support is:

`A retrieval-augmented intrusion detection architecture with learned flow embeddings and adaptive memory can improve class-balanced performance and provide exemplar-based interpretability while remaining competitive with reduced retained training data and evolving session distributions.`


## What Is Novel in This Project

The novelty should be framed carefully.

### Primary novelty

The paper's primary novelty is the use of retrieval augmentation as a core inference mechanism for flow-based intrusion detection on imbalanced multiclass traffic.

This includes:

- a learned embedding space for network flows
- indexed memory over labeled historical examples
- inference conditioned on retrieved neighbors
- attention or retrieval-conditioned aggregation rather than pure nearest-neighbor voting

### Secondary novelty

The project also introduces a practical adaptive-memory angle:

- the retrieval database is not treated as static
- stored exemplars can be refreshed through eviction and insertion
- this allows continued relevance of the memory under evolving traffic
- the method may remain effective even when retaining only part of the original training set

### Tertiary contribution

A third contribution is practical interpretability:

- predictions can be explained using retrieved similar flows
- neighbor labels and similarities provide evidence for the model's decision
- this is valuable for IDS auditing and security operations

### What should not be presented as the main novelty

These may improve results, but they are not the main scientific contribution:

- better training tricks alone
- early stopping
- stronger encoder blocks such as GLU residual layers
- self-supervised pretraining such as SCARF
- confusion matrix tooling
- MLflow integration

Those should be treated as implementation choices or secondary ablations, not the headline novelty.


## Project Scope to Keep

The paper should stay focused on the single-stage `RAG-NIDS` architecture currently aligned with the `exp4` direction.

The two-stage architecture explored in another branch should not be the focus of this paper.

If mentioned at all, it should be described only as an alternate exploratory path outside the scope of the present manuscript.


## Current Architecture Summary

Based on the current project direction, the single-stage pipeline is:

1. Load and preprocess flow-based traffic from `CIC-IDS2017`.
2. Learn tabular flow embeddings using an encoder trained with supervised contrastive loss plus auxiliary cross-entropy.
3. Build an index over stored training embeddings.
4. For a query flow, retrieve nearest labeled neighbors from memory.
5. Use a retrieval-conditioned head to combine the query representation and retrieved evidence.
6. Predict the final attack label.
7. Expose retrieved examples for explanation.
8. Refresh the retrieval database over time using eviction and insertion across sessions.

### Key model components

- `FlowEncoder`
- embedding index / retrieval memory
- retrieval-conditioned classifier head
- inference pipeline that uses retrieved neighbors
- adaptive memory with refresh behavior


## Important Implementation Identity

When writing the paper, keep the method identity stable:

- this is not simply kNN
- this is not simply a transformer over flows
- this is not simply a metric-learning classifier
- this is not only a memory-based cache

It is:

`a retrieval-augmented learned classifier for intrusion detection, with a refreshable memory of labeled examples`


## Dataset and Task Framing

### Dataset

Primary dataset:

- `CIC-IDS2017`

Canonical dataset citation to use:

- Sharafaldin et al., `Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization`

### Task

The task is multiclass flow-based intrusion detection with a heavy class imbalance.

Observed classes include:

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

### Dataset caveats to acknowledge

The paper should explicitly acknowledge:

- severe class imbalance
- some extremely tiny classes such as `Heartbleed`, `Infiltration`, `SQL Injection`
- overall accuracy can be misleading
- class grouping and split choices in prior papers may make direct metric comparison difficult


## Current Results Snapshot

The current reported results from the project discussion are:

- `accuracy = 0.9791`
- `macro-F1 = 0.6812`
- `weighted-F1 = 0.9861`
- `macro precision = 0.6487`
- `macro recall = 0.8848`

### Observed performance pattern

The model performs extremely well on major classes such as:

- `BENIGN`
- `DDoS`
- `DoS Hulk`
- `PortScan`
- `GoldenEye`

The model also achieves high recall on several minority classes, but precision drops sharply on rare attack categories.

Notable examples from the current results:

- `Bot`: high recall, low precision
- `Infiltration`: weak support and unstable performance
- `Web Attack - Brute Force`: recall much stronger than precision
- `Web Attack - XSS`: recall moderate, precision weak
- `Web Attack - Sql Injection`: very tiny support, unstable F1

### Interpretation of current performance

The current results suggest:

- the model is sensitive to rare attacks
- retrieval may help recover minority recall
- false positives remain a challenge
- the method may be trading precision for higher recall on scarce classes
- this creates a meaningful research story rather than undermining the paper

This is important: the paper should not pretend the model is perfect. Instead, it should show that the method changes the tradeoff in a way that is useful for IDS.


## How to Tell the Results Story

The results narrative should be:

- overall accuracy is high, but accuracy is not the primary contribution
- the important question is whether retrieval helps class-balanced detection
- current results indicate strong performance on major classes and meaningful sensitivity on rare attacks
- the main challenge is controlling false positives while preserving minority recall
- adaptive memory may help improve this tradeoff across sessions

### Good framing

Use wording like:

- `The proposed method improves or preserves minority-class sensitivity while maintaining strong overall performance.`
- `The retrieval mechanism enables the classifier to use exemplar-level evidence, which is particularly valuable for rare or evolving attack patterns.`
- `The adaptive memory design is intended to maintain relevance under evolving traffic distributions.`

### Bad framing

Avoid overclaiming with wording like:

- `state-of-the-art`
- `solves class imbalance`
- `perfectly handles drift`
- `outperforms all prior work`

Unless rigorous experiments and directly comparable protocols support those claims, they should not be used.


## Key Experimental Claims to Test

The paper should be built around testable claims.

### Claim 1

Retrieval-augmented inference performs better than purely parametric baselines on class-balanced intrusion detection metrics.

### Claim 2

Retrieval over learned embeddings is better than raw-space nearest-neighbor retrieval.

### Claim 3

The full retrieval-conditioned classifier is better than simple embedding plus nearest-neighbor voting.

### Claim 4

Adaptive memory refresh allows similar performance even when only `40%`, `50%`, or `80%` of training data is retained.

### Claim 5

Adaptive memory helps maintain robustness across multiple sessions under evolving traffic and possible distribution drift.

### Claim 6

Retrieved examples provide useful post hoc evidence for prediction auditing.


## Recommended Baselines

These baselines should be re-run under the same preprocessing and split/session protocol. Do not rely only on numbers copied from prior papers.

### External baseline families

- `Logistic Regression`
- `Random Forest`
- `XGBoost` or `ExtraTrees`
- `MLP`
- `Autoencoder + classifier`
- `kNN` on scaled raw features

### Internal ablations

- `Encoder + linear head`
- `Encoder + kNN vote in embedding space`
- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`

### Why each matters

- `Logistic Regression`: simple transparent baseline
- `Random Forest`: strong classical tabular baseline
- `XGBoost` or `ExtraTrees`: likely strongest conventional non-neural baseline
- `MLP`: standard deep classifier baseline
- `AE + classifier`: representation-learning baseline without retrieval
- `Raw-space kNN`: tests whether simple retrieval in input space is enough
- `Encoder + linear head`: isolates benefit of retrieval
- `Encoder + kNN vote`: isolates benefit of learned embedding plus retrieval without learned aggregation
- `Static RAG-NIDS`: baseline for the adaptive memory claim
- `Adaptive RAG-NIDS`: full proposed method


## Papers and Prior Work to Cite

These are useful anchors for the introduction, related work, and baseline framing.

### Dataset and dataset analysis

- Sharafaldin et al., `Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization`
- Panigrahi and Borah, `A detailed analysis of CICIDS2017 dataset for designing Intrusion Detection Systems`

### Broad benchmarking and classical baselines

- Maseer et al., `Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset`
- Al Lail et al., `Machine Learning for Network Intrusion Detection—A Comparative Study`

### Representation-learning / deep baselines

- Abdulhammed et al., `Features Dimensionality Reduction Approaches for Machine Learning Based Network Intrusion Detection`
- Li et al., `End-to-End Network Intrusion Detection Based on Contrastive Learning`

### Generalization, drift, and evolving threats

- D'hooge et al., `Inter-dataset generalization strength of supervised machine learning methods for intrusion detection`
- Abdulrahman and Ibrahem, `Intrusion Detection System Using Data Stream Classification`
- Guo et al., `Continual Learning for Intrusion Detection Under Evolving Network Threats`

### How to use prior work

Prior work should be used to support:

- class imbalance is a real problem
- static train/test accuracy is not the full story
- learned representations help, but are not enough on their own
- continual adaptation and drift handling are underexplored or difficult
- retrieval augmentation offers a distinct and defensible alternative


## Recommended Metrics

The paper should not lead with accuracy.

### Primary metrics

- `macro-F1`
- `per-class F1`
- `macro precision`
- `macro recall`

### Secondary metrics

- `weighted-F1`
- `accuracy`
- `confusion matrix`
- minority-class precision and recall
- `BENIGN` false positive rate

### Adaptive-memory metrics

- performance across sessions
- macro-F1 stability across sessions
- memory size
- retained exemplar count
- performance as a function of retained initial training fraction

### Efficiency metrics if available

- inference latency
- index lookup time
- memory footprint


## Why Accuracy Is Not Enough

This point should be made explicitly in the paper.

A model can achieve very high accuracy on `CIC-IDS2017` because:

- `BENIGN` dominates
- major attack classes dominate minority classes in total count
- rare classes contribute very little to the accuracy total

For this reason:

- macro-F1 is more informative
- per-class F1 is essential
- confusion matrices matter
- false positives on `BENIGN` should be reported


## Important Observations From Current Confusion Matrix

The current confusion matrix indicates:

- `BENIGN` is still occasionally confused with rare attack classes
- several rare web attack samples are mutually confused
- `Infiltration` remains unstable due to very small support
- the model is sensitive enough to find many rare attacks, but at the cost of precision

This should be used to motivate:

- memory curation
- eviction policy design
- session refresh strategy
- threshold calibration or neighbor aggregation improvements


## Adaptive Memory / Eviction Story

This is one of the strongest parts of the project and should be treated as a major contribution.

### Core idea

The retrieval memory is not static. It is refreshed over time through eviction and insertion.

This means:

- the model does not depend only on a frozen historical training set
- the example database can remain aligned with recent traffic
- the method can react to shifts in benign or attack behavior
- memory usage can remain bounded

### Why this matters scientifically

This supports two paper-level ideas:

1. Reduced data retention:
The system can achieve similar results even when only part of the original training set is retained, because memory contents can be refreshed.

2. Drift adaptation:
The system can respond to evolving traffic patterns by updating its retrieval database, rather than relying entirely on a stale static classifier.

### Existing observed project behavior to mention

Project discussion indicates:

- similar results were observed with `40%`, `50%`, and `80%` training data
- the retrieval database can refresh examples as required
- multi-session experiments are underway

This is highly valuable and should definitely be included in the paper if validated rigorously.


## Multi-Session Experiment Guidance

The paper should include a multi-session or sequential evaluation block if possible.

### Why it matters

A single random split only tests static generalization. It does not test:

- adaptation over time
- memory refresh behavior
- robustness to session-to-session distribution shift

### What to evaluate

Across sessions, compare:

- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`

Potential measurements:

- macro-F1 per session
- minority-class F1 per session
- benign false positive rate per session
- memory size over time
- number of inserted/evicted exemplars
- performance drift from early to late sessions

### What to claim if supported

If the evidence holds, the paper can claim:

- adaptive retrieval memory helps maintain competitive intrusion detection performance across evolving sessions
- the method reduces the need for full retraining under distribution change


## Suggested Experiment Table Structure

### Table 1: Main static split comparison

Include:

- `LR`
- `RF`
- `XGBoost` or `ExtraTrees`
- `MLP`
- `AE + classifier`
- `Encoder + linear head`
- `Encoder + embedding-space kNN`
- `Static RAG-NIDS`
- `Adaptive RAG-NIDS` if applicable in the same setting

Report:

- accuracy
- weighted-F1
- macro-F1
- macro precision
- macro recall

### Table 2: Low-data regime

Train with:

- `40%`
- `50%`
- `80%`
- `100%`

Compare:

- `RF`
- `XGBoost` or `ExtraTrees`
- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`

Report:

- macro-F1
- per-class macro recall
- memory size

### Table 3: Session-wise adaptive evaluation

Compare:

- `Static RAG-NIDS`
- `Adaptive RAG-NIDS`

Across sessions:

- macro-F1
- minority F1
- benign false positive rate
- memory update statistics

### Table 4: Ablation study

Remove or vary:

- retrieval
- learned aggregation head
- adaptive refresh
- eviction
- memory size


## Suggested Ablation Experiments

These ablations are important because they prove the source of gains.

### Retrieval ablations

- no retrieval
- raw-feature kNN
- embedding-space kNN
- full retrieval-conditioned head

### Memory ablations

- static memory
- adaptive memory
- different memory sizes
- different eviction policies if available

### Training/data ablations

- full training data
- `80%`
- `50%`
- `40%`

### Representation ablations

Optional, lower priority:

- with and without SCARF pretraining
- with and without stronger encoder blocks

These should not distract from the main retrieval story.


## Intended Paper Contribution List

This is a strong draft contribution list for the paper introduction:

1. We propose `RAG-NIDS`, a retrieval-augmented intrusion detection framework that classifies network flows using both learned flow embeddings and an indexed memory of labeled historical examples.
2. We show that retrieval-conditioned classification is a useful inductive bias for imbalanced multiclass intrusion detection, particularly when minority-class behavior is more informative than aggregate accuracy.
3. We introduce or study an adaptive memory mechanism with eviction and insertion that supports memory refresh under evolving traffic and enables strong performance even with reduced retained training data.
4. We provide exemplar-based interpretability through retrieved neighbors, making predictions more auditable than purely parametric baselines.


## Recommended Introduction Narrative

The introduction should follow this logic:

1. Intrusion detection remains difficult because datasets are imbalanced, attack behaviors evolve, and high accuracy can hide weak minority-class performance.
2. Most existing approaches are static classifiers that map a flow directly to a label using fixed parameters.
3. In practice, many IDS decisions could benefit from comparison to similar historical flows.
4. Retrieval augmentation provides a way to combine learned representations with explicit exemplar memory.
5. A refreshable memory is especially attractive for changing traffic conditions and bounded retention settings.
6. Therefore, the paper introduces `RAG-NIDS`.


## Recommended Related Work Structure

Organize related work into four categories:

1. Classical ML for IDS
2. Deep and representation-learning IDS
3. Memory, similarity, and retrieval-based methods
4. Drift adaptation, streaming IDS, and continual intrusion detection

The gap to highlight is:

- prior IDS work is rich in static classifiers
- there is less focus on retrieval-augmented inference with adaptive memory in tabular flow-based IDS


## Recommended Method Section Structure

The method section should cover:

1. Problem setup
2. Flow preprocessing
3. Learned flow encoder
4. Retrieval memory/index
5. Retrieval-conditioned classification
6. Adaptive memory refresh mechanism
7. Training objective
8. Inference and explanation behavior

### Important method writing guidance

The method should not read like a software manual. It should explain:

- what is stored in memory
- how neighbors are retrieved
- how retrieved information influences prediction
- how memory is updated across sessions
- why this design should help minority classes and drift robustness


## Recommended Results Discussion Angles

When discussing results, focus on:

- class-balanced performance
- rare-class behavior
- precision-recall tradeoff
- retrieval vs non-retrieval differences
- static vs adaptive memory behavior
- low-data retention findings

### If current result pattern persists

If the model continues to show:

- high minority recall
- lower minority precision

Then discuss this honestly as:

- a sensitivity-oriented retrieval effect
- potentially attractive for security screening where misses are costly
- an opportunity for improved memory curation or thresholding


## Risks, Weaknesses, and Honest Limitations

The paper should acknowledge likely limitations:

- results may depend strongly on the `CIC-IDS2017` split protocol
- extremely small classes make some metrics unstable
- retrieval systems may increase false positives if memory is poorly curated
- adaptive memory adds operational complexity
- evaluation on one dataset may limit external validity

If not addressed, reviewers may raise these anyway. It is better to handle them proactively.


## Claims to Avoid Unless Proven

Do not claim:

- state-of-the-art performance
- universal drift robustness
- real-time deployment readiness
- strong generalization to all network environments
- superiority over all prior work based only on non-comparable reported metrics


## Strongest Defensible Claims

The safest strong claims are:

- retrieval augmentation is useful for imbalanced multiclass intrusion detection
- adaptive memory makes the retrieval framework more practical under evolving traffic
- similar performance can be maintained under reduced retained training data when memory refresh is used
- retrieved neighbors improve interpretability and auditing value


## What to Include in the Paper

### Must include

- motivation around class imbalance and static-model limitations
- method overview with memory and retrieval
- baseline comparisons
- macro-F1 and per-class metrics
- confusion matrix analysis
- low-data retention experiment
- multi-session or drift-aware evaluation if available
- discussion of adaptive memory and refresh
- limitations

### Strongly recommended

- qualitative retrieval examples
- ablation on static vs adaptive memory
- memory size analysis
- discussion of minority precision-recall tradeoffs

### Optional

- implementation details like GLU or SCARF as secondary ablations
- engineering details around experiment tracking


## What the Abstract Should Eventually Emphasize

The abstract should likely emphasize:

- imbalanced intrusion detection is hard
- existing methods are mostly static classifiers
- `RAG-NIDS` uses learned embeddings plus retrieval over labeled memory
- the memory can be refreshed over time
- the approach improves class-balanced performance or remains competitive under reduced retained data
- the method provides exemplar-based interpretability


## Short Draft-Ready Novelty Statement

Use this as a drafting aid:

`The novelty of this work lies in treating intrusion detection as retrieval-augmented classification rather than purely parametric multiclass prediction. RAG-NIDS combines learned flow embeddings with an indexed memory of labeled historical traffic, allowing each prediction to be conditioned on similar prior examples. In addition, the retrieval memory is refreshable through eviction and insertion, enabling adaptation to evolving traffic patterns and competitive performance even when only a reduced portion of the original training data is retained.`


## Short Draft-Ready Research Hypothesis

Use this as a drafting aid:

`We hypothesize that retrieval-conditioned classification over learned flow embeddings yields better class-balanced intrusion detection than static parametric baselines, and that adaptive memory refresh helps preserve performance under reduced retained training data and across evolving traffic sessions.`


## Short Draft-Ready Contribution Paragraph

Use this as a drafting aid:

`This work presents RAG-NIDS, a retrieval-augmented network intrusion detection framework for imbalanced multiclass traffic classification. The method learns discriminative flow embeddings, retrieves similar labeled flows from an indexed memory at inference time, and conditions classification on both the query and the retrieved evidence. Unlike static retrieval-based systems, the proposed framework supports adaptive memory refresh through eviction and insertion, making it suitable for evolving traffic conditions and bounded-data retention settings. Experiments are designed to evaluate static classification performance, low-data retention behavior, and multi-session robustness, with a focus on macro-F1, per-class performance, and exemplar-based interpretability.`


## Instructions for an LLM Drafting the Paper

If this document is given to an LLM to draft the paper, the LLM should follow these instructions:

1. Write the paper about the single-stage `RAG-NIDS` architecture only.
2. Treat retrieval augmentation and adaptive memory as the core contribution.
3. Do not oversell auxiliary implementation details as the main novelty.
4. Emphasize macro-F1, per-class behavior, and low-data plus session-wise evaluation.
5. Explicitly discuss the tradeoff between minority recall and false positives.
6. Keep claims conservative unless directly backed by experiments.
7. Use prior papers mainly to frame the gap, not to make non-comparable leaderboard claims.


## Final Positioning Summary

This project should be positioned as:

`a retrieval-augmented, memory-adaptive intrusion detection framework for imbalanced and evolving network traffic`

This project should not be positioned as:

- just another deep learning IDS classifier
- only a contrastive learning paper
- only a nearest-neighbor paper
- only a drift-detection paper
- only a benchmark report on `CIC-IDS2017`

