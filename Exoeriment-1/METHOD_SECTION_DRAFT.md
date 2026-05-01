# Methodology Draft: Continual RAG-NIDS Sessions

## Overview

This experiment evaluates a continual-learning variant of the RAG-NIDS pipeline under session-wise traffic evolution. Each session contains benign flows plus a subset of attack types. The model is trained sequentially across sessions so that later sessions can introduce new attack classes while retaining performance on earlier classes.

## Session Protocol

We organize the CIC-IDS2017 CSVs into four chronological sessions:

1. Session 1:
   - `Monday-WorkingHours.pcap_ISCX.csv`
   - `Tuesday-WorkingHours.pcap_ISCX.csv`
   - initial classes: `BENIGN`, `FTP-Patator`, `SSH-Patator`
2. Session 2:
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - new classes: `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `Heartbleed`
3. Session 3:
   - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
   - new classes: `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS`, `Infiltration`
4. Session 4:
   - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
   - new classes: `Bot`, `DDoS`, `PortScan`

## Training Procedure

For each session, the pipeline:

1. loads only the CSVs assigned to that session
2. cleans and standardizes the feature space
3. splits the session into train and test partitions
4. expands the label space if unseen classes appear
5. warm-starts the encoder and retrieval-conditioned classifier from the previous session
6. mixes in a replay buffer of retained exemplars from prior sessions
7. retrains the model on the current session data plus replay
8. evaluates on the current session test split
9. optionally evaluates prior session test sets to measure forgetting

## Evaluation

We report the following metrics per session:

- accuracy
- macro precision
- macro recall
- macro F1
- per-class precision, recall, F1, and support

We also track retention on earlier session test sets after each new session. This gives a direct measure of old-class preservation instead of only current-session performance.

## Interpretation

This protocol is a continual-learning study, not a static benchmark run. The primary question is whether the retrieval-augmented architecture can absorb new attack classes while avoiding catastrophic forgetting on older ones.
