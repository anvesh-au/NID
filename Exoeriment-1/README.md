# RAG-NIDS Continual Learning Branch

This branch adds a session-based continual-learning pipeline on top of the existing single-session RAG-NIDS system.

## What Was Added

- A new continual-learning runner in `rag_nids/continual.py`
- A session manifest format for describing multiple data sessions
- Label-space expansion when new attack classes appear in later sessions
- Encoder and classifier warm-starting across sessions
- Replay buffering to retain old-class exemplars
- Per-session evaluation with:
  - accuracy
  - macro precision
  - macro recall
  - macro F1
  - per-class precision/recall/F1
- Retention checks against earlier session test sets to measure forgetting
- A sample manifest in `session_manifest.example.json`

The original single-session training path remains available and unchanged in spirit.

## How To Run

### 1. Single-session training

Use the existing pipeline when you want one train/test split over one dataset directory.

```bash
python main.py --data_dir /path/to/CIC-IDS2017/MachineLearningCVE
```

### 2. Continual-learning sessions

Use the new session pipeline when you want multiple sessions, each with its own train/test split.

```bash
python main.py --session_manifest session_manifest.example.json
```

Optional outputs:

```bash
python main.py \
  --session_manifest session_manifest.example.json \
  --session_output_dir outputs/sessions
```

## Session Manifest Format

The manifest is JSON with a `sessions` array. Each entry must provide:

- `name`: session name used in logs and output folders
- `csv_dir`: directory containing the CSV files for that session
- `subsample`: optional per-session row cap

Example:

```json
{
  "sessions": [
    {
      "name": "session_1",
      "csv_dir": "/path/to/session_1_csvs",
      "subsample": 50000
    },
    {
      "name": "session_2",
      "csv_dir": "/path/to/session_2_csvs",
      "subsample": 50000
    }
  ]
}
```

### Concrete Example From This Repo

The CSVs already present in `MachineLearningCVE` can be grouped into four sessions:

- `session_1`: `Monday-WorkingHours.pcap_ISCX.csv`, `Tuesday-WorkingHours.pcap_ISCX.csv`
  - classes: `BENIGN`, `FTP-Patator`, `SSH-Patator`
- `session_2`: `Wednesday-workingHours.pcap_ISCX.csv`
  - classes: `BENIGN`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `Heartbleed`
- `session_3`: `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`, `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
  - classes: `BENIGN`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS`, `Infiltration`
- `session_4`: `Friday-WorkingHours-Morning.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
  - classes: `BENIGN`, `Bot`, `DDoS`, `PortScan`

Concrete manifest:

```json
{
  "sessions": [
    {
      "name": "session_1",
      "csv_dir": "MachineLearningCVE/session_1",
      "subsample": 50000
    },
    {
      "name": "session_2",
      "csv_dir": "MachineLearningCVE/session_2",
      "subsample": 50000
    },
    {
      "name": "session_3",
      "csv_dir": "MachineLearningCVE/session_3",
      "subsample": 50000
    },
    {
      "name": "session_4",
      "csv_dir": "MachineLearningCVE/session_4",
      "subsample": 50000
    }
  ]
}
```

The easiest way to use this example is to copy the relevant CSVs into those four session folders, keeping only the files listed above in each folder.

## What The Continual Pipeline Does

For each session:

1. load the session CSVs
2. align the feature schema to the first session
3. split that session into train/test
4. expand the class mapping if new labels appear
5. warm-start the encoder and cross-attention head from the prior session
6. mix in replay examples from earlier sessions
7. retrain the model
8. evaluate on the current session test split
9. optionally evaluate retention on prior session test sets

## Expected Outputs

When `--session_output_dir` is set, the runner writes:

- `session_summary.csv`
- one folder per session
- `summary.csv` for the session metrics
- `per_class_metrics.csv`
- `retention.csv` when prior-session retention is available

## Notes

- The feature layout is anchored on the first session.
- New attack classes can be introduced in later sessions.
- The replay buffer is capped per class using `--replay_per_class`.
- You can still use MLflow with the same `--run_name` and tracking flags already supported by the project.

## Recommended Next Step

Prepare session-specific CSV directories and use `session_manifest.example.json` as the template, then run the continual mode command above.
