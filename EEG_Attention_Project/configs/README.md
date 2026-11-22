# configs/

- `paths.yaml`: links to existing 26,131 EEG epochs; no large data copied into this project.
- `baseline.yaml`: RT30 label protocol, subject split, CNN, normalization and training hyperparameters.
- `channels.yaml`: canonical 30 channels and candidate E1/E2 channel groups.

WARNING: `min_rt_s` / `max_rt_s` are intentionally `null`: no RT exclusion was agreed yet.
The 0.10s / 10s values only **flag** suspicious RTs. Do not interpret the current
RT30 label as validated ground-truth attention or use it for real-time inference.

For Windows, paths are YAML single quoted and may be adjusted to your actual disk.
From the project root (after extracting all three ZIPs):

    python -m pip install numpy pandas scipy scikit-learn PyYAML torch
    python -m src.labeling --project-root .
    python -m src.splits --project-root .

The last two commands regenerate the CSV files under data/ based on local metadata.
