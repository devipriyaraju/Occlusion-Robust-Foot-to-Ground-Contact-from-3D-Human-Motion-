.PHONY: install test preprocess train evaluate refine

install:
	pip install -e .

test:
	pytest -q

preprocess:
	python scripts/preprocess.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml --variant tcn_full

evaluate:
	python scripts/evaluate_robustness.py --config configs/default.yaml

refine:
	python scripts/refine_root.py --config configs/default.yaml
