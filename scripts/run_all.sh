#!/usr/bin/env bash
set -euo pipefail

python scripts/preprocess.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --variant mlp_full
python scripts/train.py --config configs/default.yaml --variant tcn_foot
python scripts/train.py --config configs/default.yaml --variant tcn_full_novis
python scripts/train.py --config configs/default.yaml --variant tcn_full
python scripts/train.py --config configs/default.yaml --variant tcn_full_clean
python scripts/evaluate_robustness.py --config configs/default.yaml
python scripts/refine_root.py --config configs/default.yaml
